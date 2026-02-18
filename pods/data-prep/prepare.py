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
import shutil

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
    """Standardized Telemetry Format for Dashboard Parsing"""
    try:
        # Read previous total if preserving
        previous_total = 0
        if preserve_total:
            # We used to parse logs, but now we should ideally use a state file.
            # However, for backward compatibility with the user's setup:
            try:
                status_path = Path("_status.json")
                if status_path.exists():
                    with open(status_path, "r") as f:
                        prev_data = json.load(f)
                        if prev_data.get("stage") == "preprocess":
                            previous_total = prev_data.get("total_records", 0)
            except:
                pass
        
        total_rows = int(SAMPLES_COUNTER._value.get())
        if preserve_total:
             total_rows += previous_total
        telemetry = (
            f"[TELEMETRY] stage=Data Prep | status={status} | rows={int(total_rows)} | "
            f"throughput={int(throughput)} | elapsed={round(elapsed, 1)} | "
            f"cpu_cores={round(cpu_cores, 1)} | ram_gb={round(mem_gb, 2)} | ram_percent={round(mem_percent, 1)}"
        )
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
        
        self.gpu_mode = GPU_AVAILABLE and (os.getenv('FORCE_CPU', 'false').lower() != 'true')
        
        exec_type = "gpu" if self.gpu_mode else "cpu"
        self.input_dir = Path(os.getenv('INPUT_DIR', f"{self.run_root}/{exec_type}/data/raw"))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', f"{self.run_root}/{exec_type}/data/features"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_wait = int(os.getenv('MAX_WAIT_SECONDS', '3600'))
        self.push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')

        # Preflight GPU check — fall back to CPU if CUDA device is unavailable
        if self.gpu_mode:
            try:
                import cupy as cp
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count == 0:
                    raise RuntimeError("No CUDA devices found")
                cp.cuda.Device(0).use()
                log(f"GPU Context initialized. {device_count} device(s) available")
            except Exception as e:
                log(f"GPU preflight FAILED ({e}). Falling back to CPU (Polars).")
                self.gpu_mode = False
        
        log("=" * 70)
        log("Pod 2: Enterprise Feature Engineering (Continuous)")
        log(f"RUN_ID:   {self.run_id}")
        log(f"Input:    {self.input_dir}")
        log(f"Output:   {self.output_dir}")
        log(f"Backend:  {'RAPIDS cuDF (GPU)' if self.gpu_mode else 'Polars (CPU)'}")
        log("=" * 70)

    def process_continuous(self):
        """Continuous processing loop with checkpointing."""
        checkpoint_path = self.output_dir / ".processed_files.json"
        processed_files = set()
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    processed_files = set(json.load(f))
                log(f"Resuming from checkpoint: {len(processed_files)} files already processed.")
            except Exception as e:
                log(f"Failed to load checkpoint: {e}")
        
        log("Starting continuous processing loop...")
        batch_idx = 0
        
        while not STOP_FLAG:
            # 0. Check for STOP signal
            if (self.run_root / "STOP").exists():
                log("STOP signal detected. Exiting...")
                break
                
            # 1. Scan for new files
            try:
                all_files = sorted(self.input_dir.glob("worker_*.parquet"))
                # Filter out temp files from atomic writes just in case
                all_files = [f for f in all_files if not f.name.endswith('.tmp')]
                new_files = [f for f in all_files if f.name not in processed_files]
                
                # Check for upstream completion
                gather_complete = (self.input_dir / "_raw_complete").exists()
                if not new_files:
                    if gather_complete:
                        log("Upstream (Gather) is complete and all files processed. Signaling Prep completion.")
                        break 
                    time.sleep(1)
                    continue

                batch_limit = int(os.getenv('BATCH_SIZE_LIMIT', '50'))
                if len(new_files) > batch_limit:
                    log(f"Found {len(new_files)} new files. Capping current batch to {batch_limit} for stability.")
                    new_files = new_files[:batch_limit]
                    
            except Exception as e:
                log(f"Error scanning files: {e}")
                time.sleep(1)
                continue
            
            # 2. Process Batch
            batch_idx += 1
            if batch_idx % 10 == 0 or len(new_files) >= 100:
                log(f"Processing batch {batch_idx} (files: {len(new_files)})")
            start_batch = time.time()
            try:
                if self.gpu_mode:
                    row_count = self._process_gpu(new_files)
                else:
                    row_count = self._process_cpu_polars(new_files)
            except Exception as batch_exc:
                log(f"Batch processing error (skipping batch): {batch_exc}")
                # Mark files as processed so we don't retry the same corrupt set
                for f in new_files:
                    processed_files.add(f.name)
                continue
            
            duration = time.time() - start_batch
            
            # 3. Mark as processed & update checkpoint
            for f in new_files:
                processed_files.add(f.name)
            atomic_write(checkpoint_path, json.dumps(list(processed_files)))
                
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
                total_records = int(SAMPLES_COUNTER._value.get())
                status = {
                    "stage": "preprocess",
                    "state": "running",
                    "total_records": total_records,
                    "last_batch_records": row_count,
                    "run_id": self.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": "GPU" if self.gpu_mode else "CPU"
                }
                atomic_write(self.output_dir / "_status.json", json.dumps(status))
                # Also write to local directory for log_telemetry to pick up next time
                atomic_write(Path("_status.json"), json.dumps(status))
                # Create/Update completion marker to signal downstream
                if not (self.output_dir / "_prep_ready_for_train").exists() and total_records >= 1000:
                     log("Threshold reached. Signaling ready-for-train.")
                     atomic_write(self.output_dir / "_prep_ready_for_train")
            except Exception as e:
                log(f"Status update failed: {e}")

        # Signaling Prep completion OUTSIDE the loop
        try:
            log("Creating final _prep_complete marker.")
            atomic_write(self.output_dir / "_prep_complete")
        except Exception as e:
            log(f"Marker creation failed: {e}")

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
        
        q.sink_parquet(str(output_file) + ".tmp", compression='snappy')
        os.rename(str(output_file) + ".tmp", output_file)
        
        # Return row count
        return pl.read_parquet(output_file, columns=["is_fraud"]).shape[0]

    def _process_gpu(self, files: List[Path]) -> int:
        import dask_cudf
        # Proactively validate each file before building the batch.
        # This avoids the "Parquet magic bytes not found" crash on partially-written files.
        valid_files = []
        for f in files:
            try:
                # Lightweight schema-only read to confirm the file is complete
                dask_cudf.read_parquet(str(f), columns=[]).head(0)
                valid_files.append(str(f))
            except Exception as e:
                log(f"Skipping corrupted/incomplete file {f.name}: {e}")

        if not valid_files:
            log("No valid files in batch. Skipping...")
            return 0

        log(f"Reading {len(valid_files)}/{len(files)} valid files...")
        ddf = dask_cudf.read_parquet(valid_files)
        
        # Feature Engineering Pipeline (GPU)
        ddf['amt_log'] = ddf['amt'].map_partitions(lambda s: cp.log1p(s))
        
        # Time features
        ddf['hour_of_day'] = (ddf['unix_time'] / 3600 % 24).astype('int8')
        ddf['day_of_week'] = (ddf['unix_time'] / 86400 % 7).astype('int8')
        
        # Binary features
        ddf['is_weekend'] = (ddf['day_of_week'] >= 5).astype('int8')
        ddf['is_night'] = ((ddf['hour_of_day'] >= 22) | (ddf['hour_of_day'] <= 6)).astype('int8')
        
        # Distance calculation (Haversine approx for speed)
        lat_diff = (ddf['merch_lat'] - ddf['lat']) * 111.0
        lon_diff = (ddf['merch_long'] - ddf['long']) * 85.0
        # dask_cudf supports element-wise ops, but sqrt usually safer via map_partitions with cupy
        dist_sq = (lat_diff * lat_diff) + (lon_diff * lon_diff)
        ddf['distance_km'] = dist_sq.map_partitions(lambda s: cp.sqrt(s))

        # Select columns to keep (consistent with CPU)
        # Drop string columns not needed for ML
        cols_to_drop = [c for c in STRING_COLUMNS_TO_DROP if c in ddf.columns]
        ddf = ddf.drop(columns=cols_to_drop)
        
        timestamp = int(time.time() * 1000)
        # dask_cudf writes a DIRECTORY of part files, not a single file.
        # Atomic write: Write to .tmp directory first, then rename.
        final_output_dir = self.output_dir / f"features_batch_{timestamp}.parquet"
        temp_output_dir = self.output_dir / f"features_batch_{timestamp}.parquet.tmp"
        
        # Clean up temp dir if it exists from a previous failed run
        if temp_output_dir.exists():
            shutil.rmtree(str(temp_output_dir), ignore_errors=True)
            
        ddf.to_parquet(str(temp_output_dir), write_metadata_file=True)
        
        # Atomic rename
        os.rename(str(temp_output_dir), str(final_output_dir))
        # Compute row count from partition lengths (avoids re-reading the data)
        try:
            row_count = int(ddf.map_partitions(len).compute().sum())
        except Exception:
            row_count = 0
        return row_count

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