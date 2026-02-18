import re
import json
import sqlite3
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration
DB_PATH = "telemetry.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        run_id TEXT,
        cpu_util REAL,
        gpu_util REAL,
        gpu_power REAL,
        fb_read_mbps REAL,
        fb_write_mbps REAL,
        throughput_cpu REAL,
        throughput_gpu REAL,
        nfs_errors INTEGER,
        raw_data TEXT
    )''')
    conn.commit()
    conn.close()

class MatrixParser:
    """Parses raw Prometheus text format into structured dictionary"""
    
    def __init__(self):
        self.metrics = {}

    def parse_text(self, text: str):
        """Parse line-by-line Prometheus format"""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Match standard Prometheus line: metric_name{label="val",...} value
            # Regex to separate name, labels, value
            match = re.match(r'^([a-zA-Z0-9_:]+)(?:\{([^}]+)\})?\s+(.+)$', line)
            if match:
                name, labels_str, value_str = match.groups()
                
                # Parse labels
                labels = {}
                if labels_str:
                    for pair in labels_str.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            labels[k.strip()] = v.strip().strip('"')
                
                try:
                    value = float(value_str)
                except ValueError:
                    continue

                if name not in self.metrics:
                    self.metrics[name] = []
                
                self.metrics[name].append({
                    "labels": labels,
                    "value": value
                })

    def get_scalar(self, name: str, labels: Dict = None) -> float:
        """Retrieve a simple scalar metric, optionally filtering by labels"""
        if name not in self.metrics:
            return 0.0
        
        for sample in self.metrics[name]:
            if labels:
                # Check if sample.labels contains all requested labels
                if all(sample['labels'].get(k) == v for k, v in labels.items()):
                    return sample['value']
            else:
                # If no labels requested, return first found (or sum? usually first for scalar)
                return sample['value']
        return 0.0

    def get_sum(self, name: str) -> float:
        """Sum all values for a metric name"""
        if name not in self.metrics:
            return 0.0
        return sum(s['value'] for s in self.metrics[name])

def extract_business_metrics(parser: MatrixParser) -> Dict[str, Any]:
    """Map low-level metrics to high-level Dashboard JSON"""
    
    # 1. CPU (Node Exporter)
    # 100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)
    # Simplified for snapshot: just look at basic idle? 
    # Hard to do rate() on a static file, so we might store raw counter or mock util if parsing file.
    # For now, let's just grab a dummy or raw value.
    cpu_idle_seconds = parser.get_sum("node_cpu_seconds_total") 
    # Real rate calculation requires two snapshots. 
    # Accessing 'node_load1' is a safe proxy for instant check from file
    cpu_load = parser.get_scalar("node_load1")
    cpu_util = min(100, (cpu_load / 32) * 100) # Assumes 32 cores roughly

    # 2. GPU (DCGM)
    gpu_util = parser.get_scalar("DCGM_FI_DEV_GPU_UTIL")
    gpu_power = parser.get_scalar("DCGM_FI_DEV_POWER_USAGE")
    
    # 3. FlashBlade (Pure Exporter)
    fb_read = parser.get_sum("purefb_array_performance_throughput_bytes") # This sums all dims, careful
    # Filter by dimension if available in labels, but parser dict list handles it
    
    # Refined search
    fb_read_bytes = 0
    fb_write_bytes = 0
    if "purefb_array_performance_throughput_bytes" in parser.metrics:
        for s in parser.metrics["purefb_array_performance_throughput_bytes"]:
            if s['labels'].get('dimension') == 'read':
                fb_read_bytes += s['value']
            elif s['labels'].get('dimension') == 'write':
                fb_write_bytes += s['value']
    
    fb_read_mbps = fb_read_bytes / (1024**2)
    fb_write_mbps = fb_write_bytes / (1024**2)

    # 4. NFS Errors
    nfs_errors = 0
    if "node_filesystem_device_error" in parser.metrics:
        nfs_errors = sum(1 for s in parser.metrics["node_filesystem_device_error"] 
                        if s['value'] > 0 and 'fraud' in s['labels'].get('device', ''))

    return {
        "cpu_util": round(cpu_util, 2),
        "gpu_util": gpu_util,
        "gpu_power": gpu_power,
        "fb_read_mbps": round(fb_read_mbps, 2),
        "fb_write_mbps": round(fb_write_mbps, 2),
        "nfs_errors": int(nfs_errors),
        # Business logic placeholders (would come from PushGateway metrics if in dump)
        "throughput_cpu": 0,
        "throughput_gpu": 0
    }

def save_to_sqlite(data: Dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO metrics (
        timestamp, run_id, cpu_util, gpu_util, gpu_power, 
        fb_read_mbps, fb_write_mbps, throughput_cpu, throughput_gpu, 
        nfs_errors, raw_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        datetime.now().isoformat(),
        "run-manual", # Extract from args if needed
        data['cpu_util'],
        data['gpu_util'],
        data['gpu_power'],
        data['fb_read_mbps'],
        data['fb_write_mbps'],
        data['throughput_cpu'],
        data['throughput_gpu'],
        data['nfs_errors'],
        json.dumps(data)
    ))
    conn.commit()
    conn.close()
    print(f"Stored metrics at {datetime.now().isoformat()}")

def monitor(url: str, interval: int = 1):
    """Continuously poll and store metrics"""
    print(f"Starting telemetry monitor on {url} (interval={interval}s)...")
    init_db()
    parser = MatrixParser()
    
    while True:
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                parser.metrics = {} # Reset
                parser.parse_text(r.text)
                data = extract_business_metrics(parser)
                save_to_sqlite(data)
            else:
                print(f"Error: {r.status_code}", end="\r")
        except Exception as e:
            print(f"Connection Error: {e}", end="\r")
        
        time.sleep(interval)

def main():
    import sys
    
    # continuous mode
    if "--monitor" in sys.argv:
        url = "http://10.23.181.153:9090/metrics" 
        monitor(url)
        return

    init_db()
    parser = MatrixParser()
    
    # Mode 1: Parse from file (if arg provided)
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        print(f"Parsing file: {sys.argv[1]}")
        with open(sys.argv[1], 'r') as f:
            content = f.read()
            parser.parse_text(content)
            
    # Mode 2: Fetch from URL (Default)
    else:
        url = "http://10.23.181.153:9090/metrics" # Node Exporter or Fed Endpoint
        print(f"Fetching from {url}...")
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                parser.parse_text(r.text)
            else:
                print(f"Failed to fetch: {r.status_code}")
                return
        except Exception as e:
            print(f"Connection error: {e}")
            return

    # Extract & Store
    data = extract_business_metrics(parser)
    print("Parsed Data:", json.dumps(data, indent=2))
    save_to_sqlite(data)

if __name__ == "__main__":
    main()
