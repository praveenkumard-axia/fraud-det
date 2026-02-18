#!/usr/bin/env python3
"""
Pod 3: Model Training (Optimized)
Trains XGBoost models using Polars for data loading and automated backend selection.
Optimizations:
- Polars for fast CPU data loading
- Single path execution (no redundant CPU vs GPU comparison)
- Automatic Triton config generation with dynamic batching
"""

import os
import sys
import gc
import json
import time
import logging
import psutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any

# CPU imports
import polars as pl
import numpy as np
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

def log_telemetry(rows, throughput, elapsed, cpu_cores, mem_gb, mem_percent, status="Running", preserve_total=False):
    """Write structured telemetry to logs for dashboard parsing."""
    try:
        # Read previous total if preserving
        previous_total = 0
        if preserve_total:
            # Parse last telemetry entry from logs to get previous row count
            try:
                with open("pipeline_report.txt", "r") as f:
                    lines = f.readlines()
                    for line in reversed(lines[-100:]):  # Check last 100 lines
                        if "[TELEMETRY]" in line and "stage=" in line:
                            # Parse the previous stage's row count
                            parts = line.split("|")
                            for part in parts:
                                if "rows=" in part:
                                    prev_rows = int(part.split("=")[1].strip())
                                    # Only preserve if from different stage
                                    if "stage=Model Train" not in line:
                                        previous_total = prev_rows
                                    break
                            break
            except:
                pass
        
        total_rows = (previous_total + rows) if preserve_total else rows
        telemetry = f"[TELEMETRY] stage=Model Train | status={status} | rows={int(total_rows)} | throughput={int(throughput)} | elapsed={round(elapsed, 1)} | cpu_cores={round(cpu_cores, 1)} | ram_gb={round(mem_gb, 2)} | ram_percent={round(mem_percent, 1)}"
        print(telemetry, flush=True)
    except:
        pass

# Check for GPU availability
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except Exception as e:
    print(f"WARNING: GPU libraries failed to import: {e}")
    GPU_AVAILABLE = False
FEATURE_COLUMNS = [
    'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
    'merch_zipcode', 'zip', 'amt_log', 'hour_of_day', 'day_of_week',
    'is_weekend', 'is_night', 'distance_km', 'category_encoded', 'state_encoded',
    'gender_encoded', 'city_pop_log', 'zip_region'
]

# Columns to exclude (IDs, raw text, etc.)
EXCLUDE_COLUMNS = [
    'transaction_id', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
    'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob',
    'trans_num', 'is_fraud'
]

from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry, push_to_gateway

# Prometheus Metrics
REGISTRY = CollectorRegistry()
TRAIN_SAMPS_GAUGE = Gauge("training_samples_per_sec", "Training throughput", registry=REGISTRY)
GPU_MEM_GAUGE = Gauge("gpu_memory_usage_bytes", "GPU memory usage", registry=REGISTRY)
PRECISION_GAUGE = Gauge("model_precision", "Model precision", registry=REGISTRY)
RECALL_GAUGE = Gauge("model_recall", "Model recall", registry=REGISTRY)

def atomic_write(path: Path, content: str = ""):
    """Enterprise-grade atomic write: tempfile -> fsync -> replace"""
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(dir_path), delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, str(path))

class ModelTrainer:
    def __init__(self):
        self.run_id = os.getenv('RUN_ID', 'run-default')
        self.run_root = Path(f"/fraud-benchmark/runs/{self.run_id}")
        
        self.input_dir = Path(os.getenv('INPUT_DIR', f"{self.run_root}/cpu/data/features"))
        self.output_path_cpu = Path(os.getenv('OUTPUT_DIR_CPU', f"{self.run_root}/cpu/models"))
        self.output_path_gpu = Path(os.getenv('OUTPUT_DIR_GPU', f"{self.run_root}/gpu/models"))
        
        self.output_path_cpu.mkdir(parents=True, exist_ok=True)
        self.output_path_gpu.mkdir(parents=True, exist_ok=True)
        
        self.gpu_mode = GPU_AVAILABLE
        self.max_wait = int(os.getenv('MAX_WAIT_SECONDS', '3600'))
        self.push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')

        # Preflight GPU check — fall back to CPU if CUDA device is unavailable
        if self.gpu_mode:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count == 0:
                    raise RuntimeError("No CUDA devices found")
                log.info(f"GPU preflight OK: {device_count} device(s) available")
            except Exception as e:
                log.warning(f"GPU preflight FAILED ({e}). Falling back to CPU.")
                self.gpu_mode = False
        
        log.info("=" * 70)
        log.info("Pod 3: Enterprise Model Training")
        log.info(f"RUN_ID:      {self.run_id}")
        log.info(f"Input:       {self.input_dir}")
        log.info(f"Backend:     {'GPU (L40)' if self.gpu_mode else 'CPU'}")
        log.info("=" * 70)

    def wait_for_upstream(self) -> bool:
        """Wait for _prep_complete marker."""
        marker = self.input_dir / "_prep_complete"
        log.info(f"Waiting for upstream marker: {marker}")
        
        start_wait = time.time()
        while not marker.exists():
            if (time.time() - start_wait) > self.max_wait:
                log.error("TIMEOUT: Upstream marker not found.")
                return False
            time.sleep(10)
        return True

    def prepare_data(self) -> Tuple[Any, Any, Any, Any, List[str]]:
        files = sorted(self.input_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No feature files in {self.input_dir}")
            
        # Fast load 1M samples
        # Fix: Pass explicit file list instead of glob string to avoid "expanded paths were empty" error
        # despite files existing (which we verified with `glob` above).
        file_paths = [str(f) for f in files]
        df = pl.scan_parquet(file_paths).head(1000000).collect()
        
        if self.gpu_mode:
            import cudf
            df = cudf.from_pandas(df.to_pandas())
        else:
            # Fallback to Pandas for .iloc compatibility if on CPU
            df = df.to_pandas()
        
        available_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
        split_idx = int(len(df) * 0.8)
        
        X_train = df.iloc[:split_idx][available_feats]
        y_train = df.iloc[:split_idx]['is_fraud']
        X_test = df.iloc[split_idx:][available_feats]
        y_test = df.iloc[split_idx:]['is_fraud']
        
        return X_train, y_train, X_test, y_test, available_feats

    def train(self, X_train, y_train):
        start = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc'],
            'tree_method': 'hist',
            'device': 'cuda' if self.gpu_mode else 'cpu',
        }
        
        # Retry loop: transient CUDA context issues can resolve on retry
        last_exc = None
        for attempt in range(1, 4):
            try:
                model = xgb.train(params, dtrain, num_boost_round=100)
                break
            except Exception as e:
                last_exc = e
                log.warning(f"Training attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)  # 5s, 10s backoff
        else:
            raise RuntimeError(f"All training attempts failed. Last error: {last_exc}")
        
        duration = time.time() - start
        
        throughput = len(X_train) / duration
        TRAIN_SAMPS_GAUGE.set(throughput)
        if self.gpu_mode:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                GPU_MEM_GAUGE.set(total - free)
            except Exception:
                pass
            
        return model, duration

    def save_model(self, model, feature_names, metrics):
        model_name = "fraud_xgboost"
        
        # Atomic save to both paths
        for base_path in [self.output_path_cpu, self.output_path_gpu]:
            model_dir = base_path / model_name
            version_dir = model_dir / "1"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Model File
            temp_model = version_dir / "xgboost.json.tmp"
            model.save_model(str(temp_model))
            os.rename(temp_model, version_dir / "xgboost.json")
            
            # 2. Features
            atomic_write(model_dir / "feature_names.json", json.dumps(feature_names))
            
            # 3. Status JSON
            status = {
                "stage": "model-train",
                "state": "completed",
                "run_id": self.run_id,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            atomic_write(model_dir / "_status.json", json.dumps(status))

def _run_training(trainer) -> tuple:
    """Run prepare_data + train, return (model, duration, X_test, y_test, feats)."""
    X_train, y_train, X_test, y_test, feats = trainer.prepare_data()
    # Validate we have actual rows
    n_rows = len(X_train) if hasattr(X_train, '__len__') else X_train.shape[0]
    if n_rows == 0:
        raise ValueError("Training dataset is empty — upstream prep may have produced no valid rows")
    log.info(f"Training on {n_rows} rows (gpu_mode={trainer.gpu_mode})")
    model, duration = trainer.train(X_train, y_train)
    return model, duration, X_test, y_test, feats


def main():
    from sklearn.metrics import precision_score, recall_score

    start_http_server(8000)
    trainer = ModelTrainer()

    if not trainer.wait_for_upstream():
        sys.exit(1)

    # --- Outer retry loop: wraps both prepare_data() and train() ---
    # Attempt 1-3 on current backend (GPU if available, CPU otherwise).
    # If all attempts fail, fall back to CPU and try once more.
    MAX_ATTEMPTS = 3
    model = duration = X_test = y_test = feats = None
    last_exc = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            model, duration, X_test, y_test, feats = _run_training(trainer)
            break
        except Exception as e:
            last_exc = e
            log.warning(f"Training attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)  # 5s, 10s backoff
    else:
        # All GPU attempts exhausted — try once on CPU
        if trainer.gpu_mode:
            log.warning("All GPU attempts failed. Falling back to CPU for final attempt.")
            trainer.gpu_mode = False
            try:
                model, duration, X_test, y_test, feats = _run_training(trainer)
            except Exception as e:
                log.error(f"CPU fallback also failed: {e}")
                sys.exit(1)
        else:
            log.error(f"All training attempts failed: {last_exc}")
            sys.exit(1)

    # --- Evaluate ---
    try:
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        predictions = [round(v) for v in y_pred]

        # Handle GPU tensors -> numpy for sklearn
        y_test_cpu = y_test
        if hasattr(y_test, "to_numpy"):
            y_test_cpu = y_test.to_numpy()
        elif hasattr(y_test, "values"):
            try:
                y_test_cpu = y_test.values.get()
            except Exception:
                pass

        metrics = {
            "precision": float(precision_score(y_test_cpu, predictions, zero_division=0)),
            "recall": float(recall_score(y_test_cpu, predictions, zero_division=0)),
            "samples_per_sec": len(X_test) / duration if duration > 0 else 0,
        }
        PRECISION_GAUGE.set(metrics["precision"])
        RECALL_GAUGE.set(metrics["recall"])
    except Exception as e:
        log.warning(f"Metrics evaluation failed (non-fatal): {e}")
        metrics = {"precision": 0.0, "recall": 0.0, "samples_per_sec": 0.0}

    trainer.save_model(model, feats, metrics)

    try:
        push_to_gateway(trainer.push_gateway, job='model-train',
                        grouping_key={'run_id': trainer.run_id}, registry=REGISTRY)
    except Exception as e:
        log.warning(f"PushGateway Warning (non-fatal): {e}")

    log.info(f"COMPLETE: Model trained in {duration:.2f}s | precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
