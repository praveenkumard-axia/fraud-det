#!/usr/bin/env python3
"""
Real-time Metrics Fetcher
Fetches metrics from:
  - Node Exporter (infrastructure)
  - Triton Inference Server (business/ML)
  - Kubernetes Jobs (pipeline status)

Usage:
  python fetch_metrics.py
  python fetch_metrics.py --node http://NODE_IP:9100 --triton http://TRITON_IP:30802 --watch
"""

import argparse
import time
import sys
import json
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import re

# ─── ANSI Colors ────────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ─── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_NODE_EXPORTER = "http://10.23.181.44:9100"
DEFAULT_TRITON        = "http://10.23.181.44:30802"
DEFAULT_INTERVAL      = 10  # seconds


def fetch_metrics(url: str, timeout: int = 10) -> dict[str, list[tuple[dict, float]]]:
    """
    Fetch Prometheus text metrics from a URL.
    Returns dict: metric_name -> list of (labels_dict, value)
    """
    try:
        with urlopen(url + "/metrics", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        print(f"{RED}✗ Cannot reach {url}: {e}{RESET}")
        return {}

    result: dict[str, list[tuple[dict, float]]] = {}
    current_metric = None

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("# HELP"):
            continue
        if line.startswith("# TYPE"):
            parts = line.split()
            current_metric = parts[2] if len(parts) >= 3 else None
            continue

        # Parse metric line:  name{labels} value [timestamp]
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)', line)
        if not m:
            continue

        name   = m.group(1)
        labels = {}
        if m.group(2):
            for pair in re.findall(r'(\w+)="([^"]*)"', m.group(2)):
                labels[pair[0]] = pair[1]
        try:
            value = float(m.group(3))
        except ValueError:
            continue

        result.setdefault(name, []).append((labels, value))

    return result


def get(metrics: dict, name: str, labels: dict = None) -> float | None:
    """Get a single metric value, optionally filtering by labels."""
    entries = metrics.get(name, [])
    if not entries:
        return None
    if labels:
        for lbl, val in entries:
            if all(lbl.get(k) == v for k, v in labels.items()):
                return val
        return None
    return entries[0][1]


def get_all(metrics: dict, name: str) -> list[tuple[dict, float]]:
    return metrics.get(name, [])


def fmt_bytes(b: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} EB"


def fmt_pct(v: float) -> str:
    if v is None:
        return f"{DIM}N/A{RESET}"
    color = GREEN if v < 60 else (YELLOW if v < 85 else RED)
    return f"{color}{v:.1f}%{RESET}"


def status_dot(ok: bool) -> str:
    return f"{GREEN}●{RESET}" if ok else f"{RED}●{RESET}"


def section(title: str):
    width = 70
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def kv(label: str, value: str, width: int = 35):
    print(f"  {DIM}{label:<{width}}{RESET} {value}")


# ─── Infrastructure Metrics ─────────────────────────────────────────────────────

def print_system_overview(m: dict):
    section("🖥  SYSTEM OVERVIEW")

    uname = get_all(m, "node_uname_info")
    if uname:
        lbl = uname[0][0]
        kv("Hostname",    lbl.get("nodename", "?"))
        kv("Kernel",      lbl.get("release", "?"))
        kv("OS",          lbl.get("sysname", "?"))

    boot_ts = get(m, "node_boot_time_seconds")
    if boot_ts:
        uptime_s = time.time() - boot_ts
        days, rem = divmod(int(uptime_s), 86400)
        hrs, rem  = divmod(rem, 3600)
        mins      = rem // 60
        kv("Uptime", f"{days}d {hrs}h {mins}m")

    os_info = get_all(m, "node_os_info")
    if os_info:
        kv("OS Version", os_info[0][0].get("pretty_name", "?"))


def print_cpu(m: dict):
    section("🔲  CPU")

    # Total idle time per CPU → compute utilisation
    cpus = {}
    for lbl, val in get_all(m, "node_cpu_seconds_total"):
        cpu = lbl.get("cpu", "?")
        mode = lbl.get("mode", "?")
        cpus.setdefault(cpu, {})[mode] = val

    total_busy = total_all = 0.0
    for cpu_id, modes in cpus.items():
        idle  = modes.get("idle", 0)
        total = sum(modes.values())
        busy  = total - idle
        total_busy += busy
        total_all  += total

    overall_util = (total_busy / total_all * 100) if total_all else 0
    kv("CPU Count",       str(len(cpus)))
    kv("Overall Util",    fmt_pct(overall_util))
    kv("Load (1m/5m/15m)", (
        f"{get(m, 'node_load1'):.2f} / "
        f"{get(m, 'node_load5'):.2f} / "
        f"{get(m, 'node_load15'):.2f}"
    ))

    # Per-CPU util — show top 5 busiest
    per_cpu = {}
    for cpu_id, modes in cpus.items():
        idle  = modes.get("idle", 0)
        total = sum(modes.values())
        per_cpu[cpu_id] = (total - idle) / total * 100 if total else 0

    top5 = sorted(per_cpu.items(), key=lambda x: x[1], reverse=True)[:5]
    kv("Top Busy CPUs", "  ".join(f"cpu{c}={fmt_pct(u)}" for c, u in top5))


def print_memory(m: dict):
    section("💾  MEMORY")

    total     = get(m, "node_memory_MemTotal_bytes") or 0
    free      = get(m, "node_memory_MemFree_bytes") or 0
    avail     = get(m, "node_memory_MemAvailable_bytes") or 0
    cached    = get(m, "node_memory_Cached_bytes") or 0
    buffers   = get(m, "node_memory_Buffers_bytes") or 0
    used      = total - free
    util_pct  = used / total * 100 if total else 0

    kv("Total RAM",     fmt_bytes(total))
    kv("Used",          f"{fmt_bytes(used)}  ({fmt_pct(util_pct)})")
    kv("Available",     fmt_bytes(avail))
    kv("Cached+Buffers",fmt_bytes(cached + buffers))

    oom = get(m, "node_vmstat_oom_kill")
    color = RED if (oom or 0) > 0 else GREEN
    kv("OOM Kills (total)", f"{color}{int(oom or 0)}{RESET}")

    swap_total = get(m, "node_memory_SwapTotal_bytes") or 0
    kv("Swap", fmt_bytes(swap_total) if swap_total else f"{YELLOW}None configured{RESET}")


def print_disk(m: dict):
    section("💿  DISK / STORAGE")

    # Show dm devices (LVM/multipath = FlashBlade path)
    devices_of_interest = ["dm-0", "dm-3", "dm-4", "sda", "sdb", "sdc", "sdd"]

    print(f"  {'Device':<10} {'Read':<14} {'Written':<16} {'IO Time':<14} {'Writes Merged'}")
    print(f"  {'─'*10} {'─'*14} {'─'*16} {'─'*14} {'─'*15}")

    for dev in devices_of_interest:
        read_b   = get(m, "node_disk_read_bytes_total",    {"device": dev})
        write_b  = get(m, "node_disk_written_bytes_total", {"device": dev})
        io_time  = get(m, "node_disk_io_time_seconds_total", {"device": dev})
        wmerged  = get(m, "node_disk_writes_merged_total", {"device": dev})

        if read_b is None:
            continue

        # Highlight high IO time
        io_str = f"{io_time:.0f}s" if io_time else "?"
        if io_time and io_time > 10000:
            io_str = f"{RED}{io_str}{RESET}"
        elif io_time and io_time > 1000:
            io_str = f"{YELLOW}{io_str}{RESET}"

        print(f"  {dev:<10} {fmt_bytes(read_b):<14} {fmt_bytes(write_b):<16} {io_str:<14} {int(wmerged or 0)}")

    # Filesystem usage
    print()
    seen = set()
    for lbl, avail in get_all(m, "node_filesystem_avail_bytes"):
        mp  = lbl.get("mountpoint", "?")
        dev = lbl.get("device", "?")
        if dev in seen or "kubelet" in mp or "nvidia" in mp or "run/" in mp:
            continue
        seen.add(dev)

        size = get(m, "node_filesystem_size_bytes", lbl)
        if not size or size == 0:
            continue
        used_pct = (1 - avail / size) * 100
        color = GREEN if used_pct < 70 else (YELLOW if used_pct < 85 else RED)
        print(f"  {mp:<35} {fmt_bytes(size):<12} used={color}{used_pct:.1f}%{RESET}")


def print_network(m: dict):
    section("🌐  NETWORK")

    key_devs = ["bond0", "bond1", "enp195s0f0", "enp195s0f1"]
    print(f"  {'Device':<20} {'RX Total':<16} {'TX Total':<16} {'RX Drop':<10} {'Status'}")
    print(f"  {'─'*20} {'─'*16} {'─'*16} {'─'*10} {'─'*8}")

    for dev in key_devs:
        rx  = get(m, "node_network_receive_bytes_total",  {"device": dev})
        tx  = get(m, "node_network_transmit_bytes_total", {"device": dev})
        drp = get(m, "node_network_receive_drop_total",   {"device": dev})
        up  = get(m, "node_network_up", {"device": dev})

        if rx is None:
            continue

        up_str = f"{GREEN}UP{RESET}" if up else f"{RED}DOWN{RESET}"
        drp_str = f"{RED}{int(drp)}{RESET}" if drp and drp > 0 else f"{GREEN}0{RESET}"
        print(f"  {dev:<20} {fmt_bytes(rx):<16} {fmt_bytes(tx):<16} {drp_str:<10} {up_str}")

    # FC port status
    print()
    print(f"  {BOLD}Fibre Channel Ports:{RESET}")
    for lbl, _ in get_all(m, "node_fibrechannel_info"):
        host  = lbl.get("fc_host", "?")
        state = lbl.get("port_state", "?")
        speed = lbl.get("speed", "?")
        color = GREEN if state == "Online" else RED
        print(f"    {host:<12} {color}{state:<12}{RESET} {speed}")


def print_collectors_health(m: dict):
    section("🔧  NODE EXPORTER COLLECTOR HEALTH")
    failed = []
    ok_count = 0
    for lbl, val in get_all(m, "node_scrape_collector_success"):
        name = lbl.get("collector", "?")
        if val == 0:
            failed.append(name)
        else:
            ok_count += 1

    kv("Collectors OK",     f"{GREEN}{ok_count}{RESET}")
    kv("Collectors FAILED", f"{RED}{len(failed)}{RESET}  {', '.join(failed)}")


# ─── Triton / Business Metrics ──────────────────────────────────────────────────

def print_triton(m: dict):
    section("🤖  TRITON INFERENCE SERVER  (Business Metrics)")

    models = set()
    for lbl, _ in get_all(m, "nv_inference_request_success"):
        models.add((lbl.get("model", "?"), lbl.get("version", "?")))

    for model, version in sorted(models):
        lbl = {"model": model, "version": version}
        print(f"\n  {BOLD}Model: {CYAN}{model}{RESET} v{version}")

        success  = get(m, "nv_inference_request_success", lbl) or 0
        failure  = get(m, "nv_inference_request_failure", lbl) or 0
        inf_count= get(m, "nv_inference_count", lbl) or 0
        exec_cnt = get(m, "nv_inference_exec_count", lbl) or 0
        req_dur  = get(m, "nv_inference_request_duration_us", lbl) or 0
        q_dur    = get(m, "nv_inference_queue_duration_us", lbl) or 0
        comp_in  = get(m, "nv_inference_compute_input_duration_us", lbl) or 0
        comp_inf = get(m, "nv_inference_compute_infer_duration_us", lbl) or 0
        comp_out = get(m, "nv_inference_compute_output_duration_us", lbl) or 0
        pending  = get(m, "nv_inference_pending_request_count", lbl) or 0

        total_req = success + failure
        err_rate  = failure / total_req * 100 if total_req else 0
        avg_lat_ms= req_dur / success / 1000 if success else 0
        avg_q_ms  = q_dur  / success / 1000 if success else 0
        avg_infer_ms = comp_inf / exec_cnt / 1000 if exec_cnt else 0
        batch_size = inf_count / exec_cnt if exec_cnt else 0

        kv("  Total Requests",     f"{int(success + failure):,}")
        kv("  Successful",         f"{GREEN}{int(success):,}{RESET}")
        err_color = RED if failure > 0 else GREEN
        kv("  Failed",             f"{err_color}{int(failure):,}{RESET}")
        kv("  Error Rate",         fmt_pct(err_rate))
        kv("  Avg Latency",        f"{avg_lat_ms:.3f} ms")
        kv("  Avg Queue Time",     f"{avg_q_ms:.3f} ms")
        kv("  Avg Infer Time",     f"{avg_infer_ms:.3f} ms")
        kv("  Avg Batch Size",     f"{batch_size:.0f} inferences/exec")
        kv("  Pending Requests",   str(int(pending)))

        # Latency breakdown
        if success > 0:
            total_compute = comp_in + comp_inf + comp_out
            q_pct   = q_dur   / req_dur * 100 if req_dur else 0
            in_pct  = comp_in / req_dur * 100 if req_dur else 0
            inf_pct = comp_inf/ req_dur * 100 if req_dur else 0
            out_pct = comp_out/ req_dur * 100 if req_dur else 0
            print(f"\n  {BOLD}Latency Breakdown:{RESET}")
            print(f"    Queue:          {YELLOW}{q_pct:.1f}%{RESET}")
            print(f"    Input Prep:     {BLUE}{in_pct:.1f}%{RESET}")
            print(f"    Inference:      {CYAN}{inf_pct:.1f}%{RESET}")
            print(f"    Output:         {GREEN}{out_pct:.1f}%{RESET}")


def print_gpu(m: dict):
    section("🎮  GPU METRICS")

    gpus = set(lbl.get("gpu_uuid") for lbl, _ in get_all(m, "nv_gpu_memory_total_bytes"))

    for gpu in sorted(gpus):
        lbl = {"gpu_uuid": gpu}
        total_mem  = get(m, "nv_gpu_memory_total_bytes", lbl) or 0
        used_mem   = get(m, "nv_gpu_memory_used_bytes",  lbl) or 0
        util       = (get(m, "nv_gpu_utilization", lbl) or 0) * 100
        power      = get(m, "nv_gpu_power_usage", lbl) or 0
        power_lim  = get(m, "nv_gpu_power_limit", lbl) or 0
        energy     = get(m, "nv_energy_consumption", lbl) or 0
        mem_pct    = used_mem / total_mem * 100 if total_mem else 0
        power_pct  = power / power_lim * 100 if power_lim else 0

        short_uuid = gpu[-12:] if len(gpu) > 12 else gpu
        print(f"\n  {BOLD}GPU: ...{short_uuid}{RESET}")
        kv("  GPU Utilization",  fmt_pct(util) + (f" {YELLOW}⚠ Low despite active model{RESET}" if util < 5 else ""))
        kv("  Memory Used",      f"{fmt_bytes(used_mem)} / {fmt_bytes(total_mem)}  ({fmt_pct(mem_pct)})")
        kv("  Power Usage",      f"{power:.1f}W / {power_lim:.0f}W  ({fmt_pct(power_pct)})")
        kv("  Energy Total",     f"{energy/1000:.1f} kJ")

    # CPU utilization from Triton's view
    cpu_util = get(m, "nv_cpu_utilization")
    if cpu_util is not None:
        kv("  Triton CPU Util",  fmt_pct(cpu_util * 100))

    pinned_total = get(m, "nv_pinned_memory_pool_total_bytes") or 0
    pinned_used  = get(m, "nv_pinned_memory_pool_used_bytes")  or 0
    kv("  Pinned Memory Pool", f"{fmt_bytes(pinned_used)} / {fmt_bytes(pinned_total)}")


# ─── Pipeline / Job Diagnostics ─────────────────────────────────────────────────

def print_pipeline_diagnosis():
    section("🔍  PIPELINE DIAGNOSIS  (from logs)")
    print(f"""
  {RED}Problem Detected: Jobs launched simultaneously with dependencies{RESET}

  Expected order (sequential):
    1. {GREEN}data-gather{RESET}     → fetch raw data from FlashBlade NFS
    2. {GREEN}data-prep-cpu{RESET}   → CPU preprocessing
    3. {GREEN}data-prep-gpu{RESET}   → GPU preprocessing  (needs data-prep-cpu done)
    4. {GREEN}model-train-gpu{RESET} → training           (needs data-prep done)
    5. {GREEN}inference-cpu{RESET}   → CPU inference      (needs model trained)
    6. {GREEN}inference-gpu{RESET}   → GPU inference      (needs model trained)

  {RED}Observed: All 6 jobs launched within 1 second of each other{RESET}
  {YELLOW}Likely causes:{RESET}
    • NFS mount errors (10.23.181.65:/financial-fraud-detection-demo)
      seen in node_exporter filesystem metrics → jobs can't read data
    • Jobs may be completing instantly (failing silently)
    • Check: kubectl get jobs -n <namespace>
    • Check: kubectl logs job/data-gather-<id>
""")


# ─── NFS / FlashBlade Health ────────────────────────────────────────────────────

def print_flashblade_health(m: dict):
    section("🗄️  FLASHBLADE / NFS HEALTH")

    nfs_errors = []
    for lbl, val in get_all(m, "node_filesystem_device_error"):
        dev = lbl.get("device", "")
        mp  = lbl.get("mountpoint", "")
        if val == 1 and ("nfs" in lbl.get("fstype", "") or "financial" in dev):
            nfs_errors.append(mp)

    if nfs_errors:
        print(f"  {RED}✗ NFS Mount Errors Detected:{RESET}")
        for mp in nfs_errors:
            print(f"    {RED}• {mp}{RESET}")
        print(f"\n  {YELLOW}Fix: Check FlashBlade NFS export and re-mount{RESET}")
        print(f"  {DIM}  showmount -e 10.23.181.65{RESET}")
        print(f"  {DIM}  mount -t nfs 10.23.181.65:/financial-fraud-detection-demo /mnt/fb{RESET}")
    else:
        print(f"  {GREEN}✓ No NFS mount errors detected{RESET}")

    # NFS request stats
    print(f"\n  {BOLD}NFS Request Counts (top operations):{RESET}")
    ops = {}
    for lbl, val in get_all(m, "nfs_requests_total"):
        method = lbl.get("method", "?")
        ops[method] = ops.get(method, 0) + val

    for op, count in sorted(ops.items(), key=lambda x: x[1], reverse=True)[:8]:
        bar_len = min(int(count / max(ops.values()) * 30), 30)
        bar = "█" * bar_len
        print(f"    {op:<20} {CYAN}{bar:<30}{RESET} {count:,.0f}")


# ─── Alerts ──────────────────────────────────────────────────────────────────────

def print_alerts(node_m: dict, triton_m: dict):
    section("🚨  ALERTS")

    alerts = []

    # OOM Kills
    oom = get(node_m, "node_vmstat_oom_kill") or 0
    if oom > 0:
        alerts.append((RED,    f"OOM Kills: {int(oom)} — processes are being killed due to memory pressure"))

    # FC ports down
    for lbl, _ in get_all(node_m, "node_fibrechannel_info"):
        if lbl.get("port_state") != "Online":
            alerts.append((RED, f"FC port {lbl.get('fc_host')} is DOWN — reduced FlashBlade throughput"))

    # NFS errors
    for lbl, val in get_all(node_m, "node_filesystem_device_error"):
        if val == 1 and "nfs" in lbl.get("fstype", ""):
            alerts.append((RED, f"NFS mount error: {lbl.get('mountpoint', '?')}"))

    # High disk IO wait on dm devices
    for lbl, val in get_all(node_m, "node_disk_write_time_seconds_total"):
        dev = lbl.get("device", "")
        if dev in ("dm-3", "dm-4") and val > 50000:
            alerts.append((YELLOW, f"High write time on {dev}: {val:.0f}s accumulated"))

    # Triton GPU util = 0 but model loaded
    gpu_util = get(triton_m, "nv_gpu_utilization")
    if gpu_util is not None and gpu_util == 0:
        success = get(triton_m, "nv_inference_request_success", {"model": "fraud_model", "version": "1"}) or 0
        if success > 0:
            alerts.append((YELLOW, "GPU utilization = 0% despite active model — model may be CPU-only or GPU idle"))

    # Failed collectors
    for lbl, val in get_all(node_m, "node_scrape_collector_success"):
        if val == 0:
            alerts.append((YELLOW, f"Collector failed: {lbl.get('collector', '?')}"))

    # High RX FIFO errors
    fifo = get(node_m, "node_network_receive_fifo_total", {"device": "bond0"}) or 0
    if fifo > 10000:
        alerts.append((YELLOW, f"bond0 RX FIFO errors: {int(fifo):,} — possible NIC overrun"))

    if not alerts:
        print(f"  {GREEN}✓ No alerts — system healthy{RESET}")
    else:
        for color, msg in alerts:
            print(f"  {color}▶ {msg}{RESET}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def run(node_url: str, triton_url: str):
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  METRICS DASHBOARD  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"  Node Exporter : {CYAN}{node_url}{RESET}")
    print(f"  Triton Server : {CYAN}{triton_url}{RESET}")

    print(f"\n{DIM}Fetching metrics...{RESET}")
    node_m   = fetch_metrics(node_url)
    triton_m = fetch_metrics(triton_url)

    if not node_m:
        print(f"{RED}Could not fetch node_exporter metrics from {node_url}{RESET}")
    if not triton_m:
        print(f"{RED}Could not fetch Triton metrics from {triton_url}{RESET}")

    if node_m:
        print_system_overview(node_m)
        print_cpu(node_m)
        print_memory(node_m)
        print_disk(node_m)
        print_network(node_m)
        print_collectors_health(node_m)
        print_flashblade_health(node_m)

    if triton_m:
        print_triton(triton_m)
        print_gpu(triton_m)

    print_pipeline_diagnosis()
    print_alerts(node_m or {}, triton_m or {})

    print(f"\n{BOLD}{'═'*70}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch and display Prometheus metrics")
    parser.add_argument("--node",     default=DEFAULT_NODE_EXPORTER, help="Node exporter base URL")
    parser.add_argument("--triton",   default=DEFAULT_TRITON,        help="Triton server base URL")
    parser.add_argument("--watch",    action="store_true",            help="Refresh continuously")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.watch:
        print(f"{YELLOW}Watch mode — refreshing every {args.interval}s. Ctrl+C to stop.{RESET}")
        try:
            while True:
                print("\033[H\033[J", end="")  # clear screen
                run(args.node, args.triton)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Stopped.{RESET}")
    else:
        run(args.node, args.triton)


if __name__ == "__main__":
    main()