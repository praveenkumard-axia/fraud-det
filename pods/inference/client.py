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

def log_telemetry(rows, throughput, elapsed, cpu_cores, mem_gb, mem_percent, status="Running"):
    print(f"[TELEMETRY] stage=Inference | status={status} | rows={int(rows)} | throughput={int(throughput)} | elapsed={round(elapsed,1)} | cpu_cores={round(cpu_cores,1)} | ram_gb={round(mem_gb,2)} | ram_percent={round(mem_percent,1)}")

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
        
        # In benchmark, we use the CPU models path as source of truth for metadata
        self.model_dir = Path(os.getenv('MODEL_DIR', f"{self.run_root}/cpu/models/fraud_xgboost"))
        self.data_dir = Path(os.getenv('DATA_DIR', f"{self.run_root}/cpu/data/features"))
        self.push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')
        self.max_wait = int(os.getenv('MAX_WAIT_SECONDS', '3600'))
        
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
        self.log(f"Starting Continuous Inference Benchmark: {self.run_id}")
        
        # Load model
        model = xgb.Booster()
        model.load_model(str(self.model_dir / "1" / "xgboost.json"))
        
        processed_files = set()
        
        while not STOP_FLAG:
            # Check STOP
            if (self.run_root / "STOP").exists():
                self.log("STOP signal detected. Exiting...")
                break
                
            # Scan for new files
            try:
                all_files = sorted(self.data_dir.glob("*.parquet"))
                new_files = [f for f in all_files if f.name not in processed_files]
            except Exception:
                time.sleep(1)
                continue
                
            if not new_files:
                time.sleep(1)
                continue
            
            # Use only one file per iteration to keep TPS stable and "streaming"
            target_file = new_files[0]
            
            try:
                df = pl.read_parquet(target_file)
                X = df.drop(["is_fraud"])
                if 'category' in X.columns: X = X.drop(['category']) 
                dmatrix = xgb.DMatrix(X.to_pandas())
                
                # Start Timing
                latencies = []
                fraud_count = 0
                total_txns = len(X)
                
                start_bench = time.time()
                # Run prediction once for the batch
                batch_start = time.time()
                preds = model.predict(dmatrix)
                batch_elapsed = (time.time() - batch_start) * 1000 # ms
                latencies.append(batch_elapsed / total_txns)
                fraud_indices = preds > 0.5
                fraud_count += int(fraud_indices.sum())
                
                duration = time.time() - start_bench
                avg_tps = total_txns / duration if duration > 0 else 0
                
                # Mark processed (only if successful)
                processed_files.add(target_file.name)
                
                # Calculate Percentiles
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)
                
                # Push Metrics
                TPS_GAUGE.set(avg_tps)
                P95_GAUGE.set(p95)
                P99_GAUGE.set(p99)
                FRAUD_COUNTER.inc(fraud_count)
                
                # Metrics Push
                try:
                    push_to_gateway(self.push_gateway, job='inference', 
                                    grouping_key={'run_id': self.run_id}, registry=REGISTRY)
                except Exception as e:
                    pass

                # Atomic Status for Dashboard
                status = {
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
                atomic_write(self.model_dir / "../inference_status.json", json.dumps(status))
                
                print(f"BATCH: {avg_tps:.1f} TPS | p95={p95:.4f}ms | Fraud={fraud_count}")
                
            except Exception as e:
                self.log(f"Error processing file {target_file}: {e}")
                processed_files.add(target_file.name) # Skip bad file

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
