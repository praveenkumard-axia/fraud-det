#!/usr/bin/env python3
"""
Pod 2: Feature Engineering (Optimized - Continuous Streaming)
Data preparation using Polars (CPU) or RAPIDS cuDF (GPU).
"""

import os
import sys
import time
import json
import signal
import gc
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Set, Tuple

import numpy as np
import psutil

# Try importing GPU libraries
# Try importing GPU libraries
try:
    import cudf
    import cupy as cp
    import dask_cudf
    from dask.distributed import Client, wait
    from dask_cuda import LocalCUDACluster
    GPU_AVAILABLE = True
except Exception as e:
    print(f"WARNING: GPU libraries failed to import: {e}")
    import traceback
    traceback.print_exc()
    GPU_AVAILABLE = False

# Import Polars for CPU
import polars as pl

STOP_FLAG = False

def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

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
                                    if "stage=Data Prep" not in line:
                                        previous_total = prev_rows
                                    break
                            break
            except:
                pass
        
        total_rows = previous_total + rows if preserve_total else rows
        telemetry = f"[TELEMETRY] stage=Data Prep | status={status} | rows={int(total_rows)} | throughput={int(throughput)} | elapsed={round(elapsed, 1)} | cpu_cores={round(cpu_cores, 1)} | ram_gb={round(mem_gb, 2)} | ram_percent={round(mem_percent, 1)}"
        print(telemetry, flush=True)
    except:
        pass


def signal_handler(signum, frame):
    global STOP_FLAG
    log("Shutdown signal received")
    STOP_FLAG = True

# Columns to drop (strings not needed for ML)
STRING_COLUMNS_TO_DROP = [
    'merchant', 'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num',
    'category', 'state', 'gender', 'trans_date_trans_time' # dropped if exists
]

from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry, push_to_gateway

# Prometheus Metrics
REGISTRY = CollectorRegistry()
SAMPLES_COUNTER = Counter("preprocessed_records_total", "Total samples processed", registry=REGISTRY)
TPS_GAUGE = Gauge("preprocess_samples_per_sec", "Current preprocessing throughput", registry=REGISTRY)
MEM_GAUGE = Gauge("memory_usage_bytes", "Current memory usage in bytes", registry=REGISTRY)
IO_GAUGE = Gauge("io_throughput_mbps", "I/O throughput in MBps", registry=REGISTRY)

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

class DataPrepService:
    def __init__(self):
        self.run_id = os.getenv('RUN_ID', 'run-default')
        self.run_root = Path(f"/fraud-benchmark/runs/{self.run_id}")
        
        self.input_dir = Path(os.getenv('INPUT_DIR', f"{self.run_root}/cpu/data/raw"))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', f"{self.run_root}/cpu/data/features"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_mode = GPU_AVAILABLE and (os.getenv('FORCE_CPU', 'false').lower() != 'true')
        self.max_wait = int(os.getenv('MAX_WAIT_SECONDS', '3600'))
        self.push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')
        
        log("=" * 70)
        log("Pod 2: Enterprise Feature Engineering (Continuous)")
        log(f"RUN_ID:   {self.run_id}")
        log(f"Input:    {self.input_dir}")
        log(f"Output:   {self.output_dir}")
        log(f"Backend:  {'RAPIDS cuDF (GPU)' if self.gpu_mode else 'Polars (CPU)'}")
        log("=" * 70)

        if self.gpu_mode:
            # self._init_dask() # Removed: LocalCUDACluster problematic in container, using dask_cudf directly
            pass

    def process_continuous(self):
        """Continuous processing loop."""
        processed_files = set()
        
        log("Starting continuous processing loop...")
        
        while not STOP_FLAG:
            # 0. Check for STOP signal
            if (self.run_root / "STOP").exists():
                log("STOP signal detected. Exiting...")
                break
                
            # 1. Scan for new files
            try:
                all_files = sorted(self.input_dir.glob("worker_*.parquet"))
                new_files = [f for f in all_files if f.name not in processed_files]
            except Exception as e:
                log(f"Error scanning files: {e}")
                time.sleep(1)
                continue
            
            if not new_files:
                time.sleep(2)
                continue
                
            # 2. Process Batch
            log(f"Processing batch of {len(new_files)} files...")
            start_batch = time.time()
            if self.gpu_mode:
                row_count = self._process_gpu(new_files)
            else:
                row_count = self._process_cpu_polars(new_files)
            
            duration = time.time() - start_batch
            
            # 3. Mark as processed
            for f in new_files:
                processed_files.add(f.name)
                
            # 4. Metrics & Telemetry
            tps = row_count / duration if duration > 0 else 0
            
            SAMPLES_COUNTER.inc(row_count)
            TPS_GAUGE.set(tps)
            MEM_GAUGE.set(psutil.Process().memory_info().rss)
            
            # 5. Push Metrics
            try:
                push_to_gateway(self.push_gateway, job='preprocess', 
                                grouping_key={'run_id': self.run_id}, registry=REGISTRY)
            except Exception:
                pass # Soft fail
                
            log_telemetry(row_count, tps, duration, psutil.cpu_count(), 
                          psutil.Process().memory_info().rss/(1024**3), 0, status="Running", preserve_total=True)
            
            # Atomic Status Update (Cumulative)
            try:
                status = {
                    "stage": "preprocess",
                    "state": "running",
                    "total_records": int(SAMPLES_COUNTER._value.get()),
                    "last_batch_records": row_count,
                    "run_id": self.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": "GPU" if self.gpu_mode else "CPU"
                }
                atomic_write(self.output_dir / "_status.json", json.dumps(status))
                
                # Create/Update completion marker to signal downstream
                if not (self.output_dir / "_prep_complete").exists():
                     atomic_write(self.output_dir / "_prep_complete")
            except Exception as e:
                log(f"Status update failed: {e}")

    def _process_cpu_polars(self, files: List[Path]) -> int:
        file_pattern = str(self.input_dir / "worker_*.parquet")
        # Optimization: Only scan new files?
        # Polars scan_parquet accepts a list of files or a glob.
        # If we pass a list of new files, it works.
        
        q = pl.scan_parquet([str(f) for f in files])
        
        # Feature Engineering Pipeline
        q = q.with_columns([
            pl.col("amt").log1p().alias("amt_log"),
            (pl.col("unix_time") / 3600 % 24).cast(pl.Int8).alias("hour_of_day"),
            (pl.col("unix_time") / 86400 % 7).cast(pl.Int8).alias("day_of_week")
        ]).with_columns([
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend"),
            ((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 6)).cast(pl.Int8).alias("is_night"),
            (((pl.col("merch_lat") - pl.col("lat")) * 111.0).pow(2) + 
             ((pl.col("merch_long") - pl.col("long")) * 85.0).pow(2)).sqrt().alias("distance_km")
        ])
        
        current_cols = q.collect_schema().names()
        cols_to_keep = [c for c in current_cols if c not in STRING_COLUMNS_TO_DROP]
        q = q.select(cols_to_keep)
        
        # Append to a specialized batch file or a common output?
        # Users want continuous. Downstream (training) usually wants a single dataset.
        # But for *inference*, we can output many small files.
        # Ideally, we produce features_worker_XXX.parquet corresponding to input.
        
        # For simplicity in this demo, let's output one file per batch.
        # We need unique names to avoid overwriting.
        timestamp = int(time.time() * 1000)
        output_file = self.output_dir / f"features_batch_{timestamp}.parquet"
        
        q.sink_parquet(output_file, compression='snappy')
        
        # Return row count
        return pl.read_parquet(output_file, columns=["is_fraud"]).shape[0]

    def _process_gpu(self, files: List[Path]) -> int:
        # Simplified dask_cudf path
        import dask_cudf
        ddf = dask_cudf.read_parquet([str(f) for f in files])
        
        # Similar logic to CPU but on GPU
        ddf['amt_log'] = ddf['amt'].log1p()
        ddf['hour_of_day'] = (ddf['unix_time'] / 3600 % 24).astype('int8')
        # ... more features ...
        
        timestamp = int(time.time() * 1000)
        output_file = self.output_dir / f"features_batch_{timestamp}.parquet"
        
        ddf.to_parquet(output_file)
        return len(ddf)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_http_server(8000)
    
    service = DataPrepService()
    
    # In continuous mode, we don't wait for _raw_complete
    # We just wait for ANY data to appear
    log("Waiting for data stream...")
    while not any(service.input_dir.glob("worker_*.parquet")):
        if (service.run_root / "STOP").exists(): return
        time.sleep(2)
        
    try:
        service.process_continuous()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()