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

STOP_FLAG = False

def signal_handler(signum, frame):
    global STOP_FLAG
    print("Shutdown signal received")
    STOP_FLAG = True

import signal
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
        
        log_telemetry(0, 0, 0, psutil.cpu_count(), 0, 0, status="Initializing")

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
        return True

    def log(self, msg):
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

    def run_continuous(self):
        """Continuous inference loop with checkpointing."""
        checkpoint_path = self.model_dir / ".processed_features.json"
        processed_files = set()
        total_rows = 0
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    processed_files = set(json.load(f))
                self.log(f"Resuming from checkpoint: {len(processed_files)} files already processed.")
            except Exception as e:
                self.log(f"Failed to load checkpoint: {e}")

        # Load model
        model = xgb.Booster()
        model.load_model(str(self.model_dir / "1" / "xgboost.json"))
        
        latencies = []
        start_bench = time.time()
        total_txns = 0
        fraud_count = 0
        
        while not STOP_FLAG:
            # Check for STOP signal on disk
            if (self.run_root / "STOP").exists():
                self.log("STOP signal detected on disk. Exiting...")
                break
                
            try:
                all_files = sorted(self.data_dir.glob("*.parquet"))
                new_files = [f for f in all_files if f.name not in processed_files]
                
                if not new_files:
                    time.sleep(1)
                    continue
                
                # BATCH LIMIT: Avoid getting stuck in a 48k-file loop
                batch_limit = int(os.getenv('BATCH_SIZE_LIMIT', '100'))
                if len(new_files) > batch_limit:
                    self.log(f"Found {len(new_files)} new files. Scanning first {batch_limit} for this loop.")
                    new_files = new_files[:batch_limit]
                
                for target_file in new_files:
                    if STOP_FLAG: break
                    if (self.run_root / "STOP").exists(): break
                    
                    if total_txns % 1000 == 0:
                        self.log(f"Inference Progress: {total_txns} transactions processed...")
                    try:
                        start_time = time.time()
                        
                        if self.gpu_mode:
                            import cudf
                            df = cudf.read_parquet(str(target_file))
                            X = df.drop(columns=["is_fraud"])
                            if 'category' in X.columns: X = X.drop(columns=['category'])
                            dmatrix = xgb.DMatrix(X)
                        else:
                            df = pl.read_parquet(target_file)
                            X = df.drop(["is_fraud"])
                            if 'category' in X.columns: X = X.drop(['category']) 
                            dmatrix = xgb.DMatrix(X.to_pandas())
                        
                        row_count = df.shape[0]
                        preds = model.predict(dmatrix)
                        
                        # Statistics
                        txn_latency = (time.time() - start_time) * 1000
                        latencies.append(txn_latency)
                        total_txns += row_count
                        
                        fraud_indices = preds > 0.5
                        fraud_count += int(fraud_indices.sum())
                        
                        duration = time.time() - start_bench
                        avg_tps = total_txns / duration if duration > 0 else 0
                        
                        total_rows += row_count
                        processed_files.add(target_file.name)
                        
                        # Update checkpoint
                        with open(checkpoint_path, 'w') as f:
                            json.dump(list(processed_files), f)
                        
                        # Calculate Percentiles
                        p95 = np.percentile(latencies, 95) if latencies else 0
                        p99 = np.percentile(latencies, 99) if latencies else 0
                        
                        # Push Metrics
                        TPS_GAUGE.set(avg_tps)
                        P95_GAUGE.set(p95)
                        P99_GAUGE.set(p99)
                        FRAUD_COUNTER.inc(fraud_count)
                        
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
                                "tps": avg_tps,
                                "p95_latency_ms": p95,
                                "p99_latency_ms": p99,
                                "fraud_detected": fraud_count
                            },
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        atomic_write(self.model_dir / "../inference_status.json", json.dumps(status_msg))
                        log_telemetry(total_rows, avg_tps, duration, 0, 0, 0)
                        
                    except Exception as e:
                        self.log(f"Error processing file {target_file}: {e}")
                        processed_files.add(target_file.name) # Skip bad file
            
            except Exception as e:
                self.log(f"Error in inference loop: {e}")
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
