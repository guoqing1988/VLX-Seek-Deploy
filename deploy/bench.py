#!/usr/bin/env python3
"""VLX-Seek 部署压测：显存 / 内存峰值 + 预填充与解码吞吐。

只依赖标准库。每次运行一个独立 vlx-cli 进程：
  * GPU 显存：按 PID 采样 `nvidia-smi --query-compute-apps=pid,used_memory`
    （只统计本进程，排除桌面上其它常驻进程）
  * 进程内存：采样 /proc/<pid>/status 的 VmHWM（峰值 RSS）
  * 系统内存：采样 MemAvailable 最低点（GGUF 走 mmap，命中 page cache）
  * 吞吐：解析 vlx-cli stderr 的 [time] 行（prefill / decode / tok/s / dev-mem）

用法：
  python3 deploy/bench.py                       # 默认用例集
  python3 deploy/bench.py --precision q8_0 f16  # 指定精度
  python3 deploy/bench.py --repeats 3 --warm-runs 2
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import os

GGUF_DIR = os.environ.get("VLX_GGUF_DIR", "/data/www/models/VLX-Seek-1.5-10B-GGUF")
CLI = Path(os.environ.get("VLX_CLI", ROOT / "cpp_ggml" / "build-cuda" / "vlx-cli"))
# 走部署包装（含精度回退、模型目录固定），exec 不换 PID，采样不受影响
WRAPPER = HERE / "vlx.sh"
IMAGES = HERE / "test_images"
OUT = HERE / "out"

RE_TIME = re.compile(r"\[time\] (?P<key>[a-zA-Z ]+?):?\s*(?P<rest>.*)")
RE_DECODE = re.compile(r"decode (\d+) tok ([\d.]+) ms \(([\d.]+) tok/s\)")
RE_PREFILL = re.compile(r"prefill (\d+) tok ([\d.]+) ms")
RE_DEVMEM = re.compile(r"dev-mem ([\d.]+) GiB")
RE_TOTAL = re.compile(r"total ([\d.]+) ms")


def gpu_mem_for_pid(pid: int) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return 0
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            total += int(parts[1])
    return total


def mem_available_kb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def vm_hwm_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def run_case(precision: str, image: str, task: str, text: str, lang: str,
             warm_runs: int, device: str = "cuda", poll_ms: int = 100) -> dict:
    args = ["bash", str(WRAPPER), "--precision", precision,
            "--device", device, "--image-path", str(IMAGES / image),
            "--task", task, "--text", text, "--lang", lang,
            "--no-visualize"]
    if warm_runs > 0:
        args += ["--warm-runs", str(warm_runs)]

    avail0 = mem_available_kb()
    t0 = time.perf_counter()
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    gpu_peak = 0
    hwm_peak = 0
    avail_min = avail0
    while proc.poll() is None:
        gpu_peak = max(gpu_peak, gpu_mem_for_pid(proc.pid))
        hwm_peak = max(hwm_peak, vm_hwm_kb(proc.pid))
        avail_min = min(avail_min, mem_available_kb())
        time.sleep(poll_ms / 1000)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"vlx-cli 失败 ({proc.returncode}):\n{stderr[-2000:]}")

    rec: dict = {
        "precision": precision, "image": image, "task": task, "text": text,
        "wall_s": round(wall, 2),
        "gpu_peak_mib": gpu_peak,
        "vm_hwm_mib": round(hwm_peak / 1024, 1),
        "sys_mem_drop_mib": round((avail0 - avail_min) / 1024, 1),
        "vision_total_ms": [], "prefill_tok": 0, "prefill_ms": 0.0,
        "decode_tok": 0, "decode_ms": 0.0, "decode_tps": 0.0,
        "dev_mem_gib": 0.0, "answer": "", "box_count": 0,
        "stderr": stderr,
    }
    for line in stderr.splitlines():
        m = RE_DECODE.search(line)
        if m:
            rec["decode_tok"] = int(m.group(1))
            rec["decode_ms"] = float(m.group(2))
            rec["decode_tps"] = float(m.group(3))
        m = RE_PREFILL.search(line)
        if m:
            rec["prefill_tok"] = int(m.group(1))
            rec["prefill_ms"] = float(m.group(2))
        m = RE_DEVMEM.search(line)
        if m:
            rec["dev_mem_gib"] = float(m.group(1))
        m = re.search(r"vision run \d+.*total ([\d.]+) ms", line)
        if m:
            rec["vision_total_ms"].append(float(m.group(1)))
    try:
        result = json.loads(stdout)
        rec["answer"] = result.get("answer", "")[:200]
        rec["box_count"] = len(result.get("result_bbox_list", []))
        rec["completion_tokens"] = result.get("completion_tokens")
    except json.JSONDecodeError:
        rec["answer"] = stdout[:200]
    return rec


def summarize(runs: list[dict]) -> dict:
    def stat(key: str) -> dict:
        vals = [r[key] for r in runs]
        return {"min": min(vals), "median": round(statistics.median(vals), 2),
                "max": max(vals)}
    return {k: stat(k) for k in ("wall_s", "gpu_peak_mib", "vm_hwm_mib",
                                 "sys_mem_drop_mib", "decode_tps", "prefill_ms")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", nargs="+", default=["q8_0", "f16"])
    ap.add_argument("--images", nargs="+",
                    default=["demo_image.jpg", "street_traffic.jpg",
                             "water_bottles_table.jpg"])
    ap.add_argument("--task", default="detection")
    ap.add_argument("--text", default="orange; apple; person; bottle")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--poll-ms", type=int, default=100)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--repeats", type=int, default=2,
                    help="每个用例重复次数（取中位数）")
    ap.add_argument("--warm-runs", type=int, default=2,
                    help="视觉链稳态重跑次数（隔离冷启动）")
    ap.add_argument("--image", help="只跑单张图片")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    images = [args.image] if args.image else args.images
    report = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "cli": str(CLI),
              "gguf_dir": GGUF_DIR, "device": args.device, "cases": []}
    print(f"{'精度':<6}{'图片':<26}{'显存MiB':>9}{'RSS MiB':>9}"
          f"{'prefill ms':>11}{'decode tok/s':>13}{'wall s':>8}")
    for prec in args.precision:
        for image in images:
            runs = [run_case(prec, image, args.task, args.text, args.lang,
                             args.warm_runs, args.device, args.poll_ms)
                    for _ in range(args.repeats)]
            s = summarize(runs)
            case = {"precision": prec, "image": image, "summary": s,
                    "runs": [{k: v for k, v in r.items() if k != "stderr"}
                             for r in runs],
                    "stderr_last": runs[-1]["stderr"]}
            report["cases"].append(case)
            print(f"{prec:<6}{image:<26}{s['gpu_peak_mib']['max']:>9}"
                  f"{s['vm_hwm_mib']['max']:>9.0f}"
                  f"{s['prefill_ms']['median']:>11.1f}"
                  f"{s['decode_tps']['median']:>13.2f}"
                  f"{s['wall_s']['median']:>8.1f}")

    path = OUT / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
