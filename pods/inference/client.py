import os
import sys
import json
import time
import psutil
import tempfile
import polars as pl
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry, push_to_gateway
import signal

STOP_FLAG = False

def signal_handler(signum, frame):
    global STOP_FLAG
    print("Shutdown signal received")
    STOP_FLAG = True

signal.signal(signal.SIGINT, signal_handler)

def log_telemetry(rows, tps, elapsed, cpu_cores, ram_gb, ram_percent, status="Running"):
    """Standardized Telemetry Format for Dashboard Parsing"""
    try:
        telemetry = (
            f"[TELEMETRY] stage=Inference | status={status} | rows={int(rows)} | "
            f"throughput={int(tps)} | elapsed={round(elapsed, 1)} | "
            f"cpu_cores={round(cpu_cores, 1)} | ram_gb={round(ram_gb, 2)} | ram_percent={round(ram_percent, 1)}"
        )
        print(telemetry, flush=True)
    except:
        pass

# Prometheus Metrics
REGISTRY = CollectorRegistry()
TPS_GAUGE = Gauge("inference_tps", "Inference throughput", registry=REGISTRY)
P95_GAUGE = Gauge("p95_latency_ms", "95th percentile latency", registry=REGISTRY)
P99_GAUGE = Gauge("p99_latency_ms", "99th percentile latency", registry=REGISTRY)
FRAUD_COUNTER = Counter("fraud_detected_total", "Total fraud detected", registry=REGISTRY)

class InferenceClient:
    def __init__(self):
        self.run_id = os.getenv('RUN_ID', 'run-default')
        self.run_root = Path(f"/fraud-benchmark/runs/{self.run_id}")
        
        # GPU Mode Check
        self.gpu_mode = os.getenv('EXECUTION_TYPE', 'cpu') == 'gpu'
        exec_type = "gpu" if self.gpu_mode else "cpu"
        
        # In benchmark, we use the models path as source of truth for metadata
        self.model_dir = Path(os.getenv('MODEL_DIR', f"{self.run_root}/{exec_type}/models/fraud_xgboost"))
        self.data_dir = Path(os.getenv('DATA_DIR', f"{self.run_root}/{exec_type}/data/features"))
        self.tmp_dir = Path(os.getenv('TMP_DIR', f"{self.run_root}/tmp"))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')
        self.max_wait = int(os.getenv('MAX_WAIT_SECONDS', '3600'))
        
        if self.gpu_mode:
            try:
                import cupy as cp
                import cudf
                cp.cuda.Device(0).use()
                self.log("GPU Context initialized for Inference.")
            except Exception as e:
                self.log(f"GPU check failed for Inference: {e}. Falling back to CPU (Polars).")
                self.gpu_mode = False
        
        # Load model immediately to fail fast
        self.bst = xgb.Booster()
        try:
             self.bst.load_model(str(self.model_dir / "1" / "xgboost.json"))
        except Exception as e:
             self.log(f"Warning: Model not found at init, will wait: {e}")

        log_telemetry(0, 0, 0, psutil.cpu_count(), 0, 0, status="Initializing")

    def atomic_write(self, path: Path, content: str = ""):
        """Enterprise-grade atomic write: tempfile -> fsync -> replace"""
        dir_path = path.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(dir_path), delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, str(path))

    def wait_for_model(self) -> bool:
        """Wait for the model and its metadata status file."""
        status_file = self.model_dir / "_status.json"
        log_telemetry(0, 0, 0, psutil.cpu_count(), 0, 0, status="Waiting for Model")
        
        start_wait = time.time()
        while not (self.model_dir / "1" / "xgboost.json").exists() or not status_file.exists():
            if (time.time() - start_wait) > self.max_wait:
                return False
            if (self.run_root / "STOP").exists(): return False
            time.sleep(10)
        
        # Reload model to be sure
        try:
            self.bst.load_model(str(self.model_dir / "1" / "xgboost.json"))
        except:
            return False
            
        return True

    def log(self, msg):
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

    def run_continuous(self):
        """Entry point for continuous inference loop."""
        self.wait_for_data()

    def wait_for_data(self):
        """Monitor data directory for new parquet files/directories"""
        print(f"Monitoring {self.data_dir} for new data... (GPU={self.gpu_mode})", flush=True)
        
        processed_files = set()
        checkpoint_path = self.tmp_dir / "inference_checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    processed_files = set(json.load(f))
                print(f"Resuming from checkpoint: {len(processed_files)} files processed", flush=True)
            except:
                print("Corrupted checkpoint, starting fresh", flush=True)

        batch_fraud_total = 0 # Track cumulative locally for reporting
        latencies = []

        while not STOP_FLAG:
            if (self.run_root / "STOP").exists():
                print("STOP signal received", flush=True)
                break
                
            # PRO FIX 1: Handle Directory-Style Parquet (e.g. features_batch_X.parquet/)
            # Glob returns both files and directories.
            try:
                all_paths = sorted(self.data_dir.glob("*_batch_*.parquet*")) 
                new_paths = [p for p in all_paths if p.name not in processed_files]
                
                if not new_paths:
                    time.sleep(1)
                    continue
                
                for target_path in new_paths:
                    if STOP_FLAG or (self.run_root / "STOP").exists(): break

                    try:
                        start_time = time.time()
                        
                        # PRO FIX 1 cont: Handle directory vs file
                        read_path = str(target_path)
                        
                        # Load features
                        # Use read_parquet which handles both single file and directory datasets if engine supports it
                        # For extremely large directory datasets, scan_parquet is better, but here we process batch by batch
                        try:
                            df = pl.read_parquet(read_path)
                        except Exception as e:
                            self.log(f"Failed to read parquet {read_path}: {e}")
                            processed_files.add(target_path.name) # Skip bad file
                            continue
                        
                        if len(df) == 0:
                            processed_files.add(target_path.name)
                            continue

                        # Prepare for Inference
                        cols_to_drop = ['transaction_id', 'is_fraud', 'trans_date_trans_time', 'cc_num'] 
                        X = df.drop([c for c in cols_to_drop if c in df.columns])
                        
                        # PRO FIX 4: Memory Explosion
                        dmatrix = None
                        if self.gpu_mode:
                           # Best effort zero-copy if possible, or direct arrow import
                           dmatrix = xgb.DMatrix(X.to_arrow()) 
                        else:
                           # CPU: Polars -> Arrow -> DMatrix is efficient
                           dmatrix = xgb.DMatrix(X.to_arrow())

                        # Inference
                        preds = self.bst.predict(dmatrix)
                        
                        # Metrics
                        fraud_indices = (preds > 0.85).astype(int)
                        fraud_count = int(fraud_indices.sum())
                        
                        row_count = len(df)
                        duration = time.time() - start_time
                        tps = row_count / duration if duration > 0 else 0
                        
                        # PRO FIX 2: Correct Counter Increment
                        FRAUD_COUNTER.inc(fraud_count)
                        batch_fraud_total += fraud_count

                        # Track Latency
                        latencies.append(duration * 1000)
                        if len(latencies) > 10000: latencies = latencies[-10000:]
                        p95 = np.percentile(latencies, 95) if latencies else 0
                        p99 = np.percentile(latencies, 99) if latencies else 0

                        # Telemetry
                        TPS_GAUGE.set(tps)
                        P95_GAUGE.set(p95)
                        P99_GAUGE.set(p99)

                        print(f"[TELEMETRY] stage=Inference | status=Running | rows={row_count} | "
                              f"fraud_blocked={batch_fraud_total} | throughput={int(tps)} | "
                              f"elapsed={duration:.3f}", flush=True)
                              
                        # PRO FIX 3: Atomic Checkpoint
                        processed_files.add(target_path.name)
                        self.atomic_write(checkpoint_path, json.dumps(list(processed_files)))

                        # Push Metrics to Gateway
                        try:
                            push_to_gateway(self.push_gateway, job='inference', 
                                            grouping_key={'run_id': self.run_id}, registry=REGISTRY)
                        except: pass

                        # Atomic Status for Dashboard
                        status_msg = {
                            "stage": "inference",
                            "state": "running",
                            "run_id": self.run_id,
                            "metrics": {
                                "tps": tps,
                                "p95_latency_ms": p95,
                                "p99_latency_ms": p99,
                                "fraud_detected": fraud_count
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        self.atomic_write(self.model_dir / "../inference_status.json", json.dumps(status_msg))
                        log_telemetry(row_count, tps, duration, psutil.cpu_count(), psutil.virtual_memory().used/(1024**3), psutil.virtual_memory().percent)

                    except Exception as e:
                        print(f"Inference failed for {target_path.name}: {e}", flush=True)
                        # Don't mark as processed, will retry
                        time.sleep(1)

            except Exception as e:
                self.log(f"Error in main inference loop: {e}")
                time.sleep(1)

def main():
    start_http_server(8000)
    client = InferenceClient()
    
    if not client.wait_for_model():
        print("Model wait timed out or stopped.")
        sys.exit(0)
        
    try:
        client.run_continuous()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
