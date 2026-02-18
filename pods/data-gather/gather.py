#!/usr/bin/env python3
"""
Pod 1: Data Generator (Optimized)
Generates synthetic credit card transactions for fraud detection demo.
Optimizations:
- Vectorized generation (kept from original)
- Efficient timestamp handling (no string conversion)
- Optimized data types (Float32, Int32)
"""

import os
import sys
import time
import signal
import subprocess
import pickle
import psutil
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import json

STOP_FLAG = False

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

# Transaction categories with realistic distribution
CATEGORIES = [
    'gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
    'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
    'kids_pets', 'travel', 'health_fitness', 'personal_care'
]
CATEGORY_WEIGHTS = np.array([
    0.243, 0.226, 0.106, 0.092, 0.083, 0.080, 0.079, 0.037,
    0.019, 0.012, 0.008, 0.005, 0.005, 0.004
])

# US states weighted by population
US_STATES = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
    'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
    'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY'
]
STATE_WEIGHTS = np.array([
    0.118, 0.087, 0.065, 0.059, 0.039, 0.038, 0.035, 0.032, 0.031, 0.030,
    0.027, 0.026, 0.023, 0.022, 0.021, 0.021, 0.020, 0.018, 0.018, 0.018,
    0.017, 0.017, 0.015, 0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.010,
    0.010, 0.009, 0.009, 0.009, 0.009, 0.006, 0.006, 0.006, 0.005, 0.004,
    0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002
])

# String pool sizes for Faker-generated data
POOL_SIZES = {
    'first': 10_000,
    'last': 15_000,
    'street': 50_000,
    'city': 10_000,
    'merchant': 20_000,
    'job': 5_000,
    'trans_num': 100_000,
    'dob': 25_000,
}


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} | {msg}", flush=True)

def log_telemetry(rows, throughput, elapsed, cpu_cores, mem_gb, mem_percent, status="Running"):
    """Standardized Telemetry Format for Dashboard Parsing"""
    try:
        telemetry = (
            f"[TELEMETRY] stage=Data Gather | status={status} | rows={int(rows)} | "
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


def generate_pools(output_path: Path) -> Path:
    """Generate string pools using Faker (one-time startup cost)."""
    from faker import Faker
    
    log("Generating string pools...")
    fake = Faker()
    Faker.seed(42)
    
    pools = {
        'first': np.array([fake.first_name() for _ in range(POOL_SIZES['first'])]),
        'last': np.array([fake.last_name() for _ in range(POOL_SIZES['last'])]),
        'street': np.array([fake.street_address() for _ in range(POOL_SIZES['street'])]),
        'city': np.array([fake.city() for _ in range(POOL_SIZES['city'])]),
        'merchant': np.array([f"{fake.company().replace(',', '')} {fake.company_suffix()}" 
                             for _ in range(POOL_SIZES['merchant'])]),
        'job': np.array([fake.job().replace(',', ' ') for _ in range(POOL_SIZES['job'])]),
        'trans_num': np.array([fake.uuid4().replace('-', '') for _ in range(POOL_SIZES['trans_num'])]),
        'dob': np.array([fake.date_of_birth(minimum_age=18, maximum_age=85).strftime('%Y-%m-%d') 
                        for _ in range(POOL_SIZES['dob'])]),
        'categories': np.array(CATEGORIES),
        'category_weights': CATEGORY_WEIGHTS / CATEGORY_WEIGHTS.sum(),
        'states': np.array(US_STATES),
        'state_weights': STATE_WEIGHTS / STATE_WEIGHTS.sum(),
    }
    
    pools_file = output_path / "_pools.pkl"
    with open(pools_file, 'wb') as f:
        pickle.dump(pools, f)
    
    total = sum(POOL_SIZES.values())
    log(f"  Created {total:,} pooled values")
    return pools_file


# Worker script (runs in subprocess for true parallelism)
WORKER_SCRIPT = '''
import sys, pickle, time, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def generate_chunk(pools, n, rng, fraud_rate, base_time):
    """Generate n transactions using vectorized operations."""
    
    # Timestamps: Direct INT64 generation
    # OPTIMIZATION: Removed string conversion for timestamp
    unix_times = rng.integers(base_time, base_time + 31536000, size=n, dtype=np.int64)
    
    # Geographic data (US bounds) - Float32
    lats = rng.uniform(25.0, 48.0, n).astype(np.float32)
    longs = rng.uniform(-125.0, -70.0, n).astype(np.float32)
    
    # Merchant location perturbation
    merch_lats = lats + rng.normal(0, 0.5, n).astype(np.float32)
    merch_longs = longs + rng.normal(0, 0.5, n).astype(np.float32)
    
    # Transaction amounts (lognormal distribution) - Float32
    amts = np.clip(np.abs(rng.lognormal(3.5, 1.5, n)), 1.0, 25000.0).astype(np.float32)
    
    # Pool sampling indices (Int32 is sufficient for indices)
    idx = lambda pool: rng.integers(0, len(pools[pool]), n, dtype=np.int32)
    
    # Data Construction
    # OPTIMIZATION: Direct pyarrow construction would be even faster, 
    # but dictionary -> Table -> Parquet is robust and readable.
    # Using specific PyArrow types for efficiency.
    
    data = {
        # 'trans_date_trans_time': Use unix_time directly instead
        'unix_time': unix_times, # Keep raw int64 for efficiency
        
        'cc_num': rng.integers(4000000000000000, 5000000000000000, size=n, dtype=np.int64),
        'merchant': pools['merchant'][idx('merchant')],
        'category': pools['categories'][rng.choice(len(pools['categories']), size=n, p=pools['category_weights'])],
        'amt': amts,
        'first': pools['first'][idx('first')],
        'last': pools['last'][idx('last')],
        'gender': np.where(rng.random(n) < 0.5, 'M', 'F'),
        'street': pools['street'][idx('street')],
        'city': pools['city'][idx('city')],
        'state': pools['states'][rng.choice(len(pools['states']), size=n, p=pools['state_weights'])],
        'zip': rng.integers(10000, 99999, size=n, dtype=np.int32),
        'lat': lats,
        'long': longs,
        'city_pop': rng.integers(1000, 2000000, size=n, dtype=np.int32),
        'job': pools['job'][idx('job')],
        'dob': pools['dob'][idx('dob')],
        'trans_num': pools['trans_num'][idx('trans_num')],
        
        # Redundant but kept for compatibility with original schema if needed
        'merch_lat': merch_lats,
        'merch_long': merch_longs,
        'is_fraud': (rng.random(n) < fraud_rate).astype(np.int8),
        'merch_zipcode': rng.integers(10000, 99999, size=n, dtype=np.int32),
    }
    
    return data

# Parse args
worker_id = int(sys.argv[1])
output_dir = Path(sys.argv[2])
chunk_size = int(sys.argv[3])
duration = int(sys.argv[4])
pools_file = sys.argv[5]
fraud_rate = float(sys.argv[6])

# Load pools and init RNG
with open(pools_file, 'rb') as f:
    pools = pickle.load(f)
rng = np.random.default_rng(seed=worker_id * 54321 + int(time.time() * 1000) % 100000)
base_time = 1704067200  # 2024-01-01

# Generate until duration expires
start = time.time()
# RESTART ROBUSTNESS: Find latest file sequence to avoid overwrites
existing_files = list(output_dir.glob(f"worker_{worker_id:03d}_*.parquet"))
file_count = len(existing_files)

# Define Schema for better performance
schema = pa.schema([
    ('unix_time', pa.int64()),
    ('cc_num', pa.int64()),
    ('merchant', pa.string()),
    ('category', pa.string()),
    ('amt', pa.float32()),
    ('first', pa.string()),
    ('last', pa.string()),
    ('gender', pa.string()),
    ('street', pa.string()),
    ('city', pa.string()),
    ('state', pa.string()),
    ('zip', pa.int32()),
    ('lat', pa.float32()),
    ('long', pa.float32()),
    ('city_pop', pa.int32()),
    ('job', pa.string()),
    ('dob', pa.string()),
    ('trans_num', pa.string()),
    ('merch_lat', pa.float32()),
    ('merch_long', pa.float32()),
    ('is_fraud', pa.int8()),
    ('merch_zipcode', pa.int32()),
])

try:
    while True:
        data = generate_chunk(pools, chunk_size, rng, fraud_rate, base_time)
        table = pa.Table.from_pydict(data, schema=schema)
        output_file = output_dir / f"worker_{worker_id:03d}_{file_count:05d}.parquet"
        pq.write_table(table, output_file, compression='snappy')
        file_count += 1
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise e
'''




from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry, push_to_gateway

# Prometheus Metrics
REGISTRY = CollectorRegistry()
GEN_COUNTER = Counter("generated_records_total", "Total generated rows", registry=REGISTRY)
TPS_GAUGE = Gauge("records_per_second", "Current generation throughput", registry=REGISTRY)

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

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Prometheus Exporter for real-time scraping
    start_http_server(8000)
    
    # Run ID Isolation
    run_id = os.getenv('RUN_ID', 'run-default')
    run_root = Path(f"/fraud-benchmark/runs/{run_id}")
    
    output_dir_cpu = run_root / "cpu/data/raw"
    output_dir_gpu = run_root / "gpu/data/raw"
    
    num_workers = int(os.getenv('NUM_WORKERS', '16'))
    duration = int(os.getenv('DURATION_SECONDS', '3600')) # Default to 1h for benchmark
    max_rows = int(os.getenv('MAX_ROWS', '1000000'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '100000'))
    fraud_rate = float(os.getenv('FRAUD_RATE', '0.005'))
    push_gateway = os.getenv('PUSHGATEWAY_URL', '10.23.181.153:9091')
    
    output_dir_cpu.mkdir(parents=True, exist_ok=True)
    output_dir_gpu.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: Enterprise Data Generator")
    log(f"RUN_ID:     {run_id}")
    log(f"Output CPU: {output_dir_cpu}")
    log(f"Max Rows:   {max_rows:,}")
    log("-" * 70)
    
    # Generate string pools
    pools_file = generate_pools(output_dir_cpu)
    
    # Modified Worker for Dual Atomic Writes
    # FIX: Ensure indentation matches the surrounding code (8 spaces)
    WORKER_DUAL_SCRIPT = WORKER_SCRIPT.replace(
        "        pq.write_table(table, output_file, compression='snappy')",
        "        tmp_file = str(output_file) + '.tmp'\n" +
        "        pq.write_table(table, tmp_file, compression='snappy')\n" +
        "        os.rename(tmp_file, output_file)\n" +
        "        import shutil\n" +
        "        dst_path = Path(sys.argv[7]) / output_file.name\n" +
        "        tmp_dst = str(dst_path) + '.tmp'\n" +
        "        shutil.copy(output_file, tmp_dst)\n" +
        "        os.rename(tmp_dst, dst_path)"
    )
    
    log(f"Launching {num_workers} workers...")
    processes = []
    for i in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', WORKER_DUAL_SCRIPT, 
             str(i), str(output_dir_cpu), str(chunk_size), str(duration), 
             str(pools_file), str(fraud_rate), str(output_dir_gpu)],
            stdout=subprocess.DEVNULL, stderr=None 
        )
        processes.append(p)
    
    start_time = time.time()
    total_generated_rows = 0
    previous_total_rows = 0
    
    try:
        while not STOP_FLAG:
            elapsed = time.time() - start_time
            running = sum(1 for p in processes if p.poll() is None)
            
            # Accurate row counting via file count
            files = list(output_dir_cpu.glob("worker_*.parquet"))
            total_generated_rows = len(files) * chunk_size
            
            # Update Prometheus
            if total_generated_rows > previous_total_rows:
                delta = total_generated_rows - previous_total_rows
                GEN_COUNTER.inc(delta)
                previous_total_rows = total_generated_rows
            
            tps = total_generated_rows / elapsed if elapsed > 0 else 0
            TPS_GAUGE.set(tps)
            
            # Check for STOP file from Backend
            if (run_root / "STOP").exists():
                log("STOP signal detected in run directory. Exiting...")
                break

            if running == 0:
                 # If workers died, restart them or break? 
                 # For now, if all workers die, we break.
                 break
                 
            # if total_generated_rows >= max_rows or running == 0 or elapsed >= duration:
            #     break
                
            cpu_percent = psutil.cpu_percent()
            mem_info = psutil.virtual_memory()
            log_telemetry(total_generated_rows, tps, elapsed, 
                          (cpu_percent/100)*psutil.cpu_count(), 
                          mem_info.used/(1024**3), mem_info.percent)
            
            time.sleep(2)
            
    finally:
        for p in processes: p.terminate()
        
    actual_rows = len(list(output_dir_cpu.glob("worker_*.parquet"))) * chunk_size
    duration_sec = time.time() - start_time
    
    # 🔔 Atomic Status Tracking
    status = {
        "stage": "data-gather",
        "state": "completed",
        "records": actual_rows,
        "duration_sec": round(duration_sec, 2),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    atomic_write(output_dir_cpu / "_status.json", json.dumps(status))
    atomic_write(output_dir_gpu / "_status.json", json.dumps(status))
    
    # 🚀 Atomic Completion Markers
    atomic_write(output_dir_cpu / "_raw_complete")
    atomic_write(output_dir_gpu / "_raw_complete")
    
    # 📦 Push Final Metrics to PushGateway
    try:
        push_to_gateway(push_gateway, job='data-gather', 
                        grouping_key={'run_id': run_id}, registry=REGISTRY)
        log("Metrics pushed to PushGateway.")
    except Exception as e:
        log(f"PushGateway failed: {e}")

    log(f"COMPLETE: {actual_rows} rows generated and synced in {duration_sec:.2f}s.")

if __name__ == "__main__":
    main()
