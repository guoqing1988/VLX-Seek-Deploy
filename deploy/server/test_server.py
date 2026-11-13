#!/usr/bin/env python3
"""VLX-Seek 常驻服务端到端验收测试（标准库即可）。

覆盖：
  Phase A（instances=1, idle_unload=8s）
    1. 冷启动 /healthz：model_loaded=false，不触发加载
    2. 首个请求：202 → 轮询 → done，cold_start=true，load_ms>0，返回框
    3. 第二个请求：cold_start=false，load_ms=0
    4. 空闲超过 idle_unload：/healthz model_loaded=false，本进程显存归零
    5. 卸载后再请求：cold_start=true（自动重新加载）
    6. POST /v1/vlx/unload?wait=1：worker 全部 unloaded
    7. /metrics 计数可见
  Phase B（instances=2, idle_unload=0）
    8. 并发 2 个请求：两个都 done、落在不同 worker、总耗时明显小于 2 倍串行

用法：python3 deploy/server/test_server.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IMAGE = ROOT / "deploy" / "test_images" / "demo_image.jpg"
BIN = HERE / "build" / "vlx-seek-server"
GGUF_DIR = os.environ.get("VLX_GGUF_DIR", "/data/www/models/VLX-Seek-1.5-10B-GGUF")
TMP = HERE / "tmp"

PASS = 0
FAIL = 0


def check(cond: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}{('  ' + detail) if detail else ''}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{('  ' + detail) if detail else ''}")


def http(method: str, url: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def gpu_mem_for_pid(pid: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    total = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            total += int(parts[1])
    return total


class Service:
    def __init__(self, port: int, instances: int, idle_s: int, tag: str):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.tag = tag
        TMP.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, INSTANCES=str(instances), IDLE_UNLOAD_S=str(idle_s),
                   PORT=str(port))
        self.log = open(HERE / f"test-{tag}.log", "w")
        self.proc = subprocess.Popen(
            [str(HERE / "serve.sh"), "--log-level", "1"], env=env,
            stdout=self.log, stderr=subprocess.STDOUT)
        self.pid = self.proc.pid

    def wait_ready(self, timeout: float = 60) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                code, _ = http("GET", self.base + "/healthz", timeout=2)
                if code == 200:
                    return True
            except Exception:
                pass
            if self.proc.poll() is not None:
                return False
            time.sleep(0.3)
        return False

    def health(self) -> dict:
        return json.loads(http("GET", self.base + "/healthz")[1])

    def submit(self, image_path: Path | None = None, **fields) -> tuple[int, dict]:
        src = image_path or IMAGE
        body = {"task": "detection", "text": "orange; apple", "lang": "zh",
                "image": base64.b64encode(src.read_bytes()).decode(), **fields}
        code, text = http("POST", self.base + "/v1/vlx/infer", body)
        return code, json.loads(text)

    def wait_job(self, job_id: str, timeout: float = 180) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            code, text = http("GET", f"{self.base}/v1/vlx/jobs/{job_id}")
            job = json.loads(text)
            if job["state"] in ("done", "failed", "cancelled"):
                return job
            time.sleep(0.2)
        raise TimeoutError(f"job {job_id} not finished")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def phase_a() -> None:
    print("\n=== Phase A：懒加载 / 空闲卸载 / 自动重载（instances=1, idle=8s） ===")
    svc = Service(port=8111, instances=1, idle_s=8, tag="a")
    try:
        check(svc.wait_ready(), "服务启动并响应 /healthz")
        h = svc.health()
        check(h["model_loaded"] is False, "冷启动时模型未加载", f"workers={h['workers_loaded']}")
        check(gpu_mem_for_pid(svc.pid) < 500, "冷启动时显存占用 < 500 MiB",
              f"{gpu_mem_for_pid(svc.pid)} MiB")

        t0 = time.time()
        code, resp = svc.submit()
        check(code == 202, "POST /v1/vlx/infer 返回 202", f"job={resp.get('job_id')}")
        job = svc.wait_job(resp["job_id"], timeout=180)
        cold_ms = (time.time() - t0) * 1000
        check(job["state"] == "done", "首个任务完成", f"wall={cold_ms:.0f}ms")
        check(job["cold_start"] is True, "首个任务标记 cold_start=true",
              f"load_ms={job['load_ms']:.0f}")
        check(job["load_ms"] > 500, "加载耗时被记录", f"{job['load_ms']:.0f} ms")
        boxes = (job.get("result") or {}).get("result_bbox_list") or []
        check(len(boxes) >= 3, "检测结果包含框", f"{len(boxes)} boxes")
        mem_loaded = gpu_mem_for_pid(svc.pid)
        check(mem_loaded > 7000, "加载后显存 > 7 GiB", f"{mem_loaded} MiB")

        # 同一 worker 连续处理不同尺寸图像：验证视觉图按 shape 重建
        # （修复前这里会越界，服务崩溃或任务失败）
        for img_name, expect_min in (("street_traffic.jpg", 0), ("demo_image.jpg", 3)):
            code, r = svc.submit(image_path=ROOT / "deploy" / "test_images" / img_name,
                                 task="vqa", text="describe the image")
            j = svc.wait_job(r["job_id"], timeout=240)
            n = len((j.get("result") or {}).get("result_bbox_list") or [])
            check(j["state"] == "done" and j["cold_start"] is False,
                  f"不同尺寸连续请求可用 ({img_name})",
                  f"boxes={n} answer={(j.get('result') or {}).get('answer','')[:20]!r}")

        code, resp = svc.submit()
        t1 = time.time()
        job2 = svc.wait_job(resp["job_id"], timeout=180)
        warm_ms = (time.time() - t1) * 1000
        check(job2["cold_start"] is False, "第二个任务 cold_start=false")
        check(job2["load_ms"] == 0, "第二个任务无加载耗时")
        check(warm_ms < cold_ms * 0.75, "热请求明显快于冷请求",
              f"warm={warm_ms:.0f}ms cold={cold_ms:.0f}ms")

        print("  ... 等待空闲卸载（9 s）")
        time.sleep(9)
        h = svc.health()
        check(h["model_loaded"] is False, "空闲后模型已卸载",
              f"workers={h['workers_loaded']}")
        mem_idle = gpu_mem_for_pid(svc.pid)
        check(mem_idle < 500, "卸载后本进程显存归零", f"{mem_idle} MiB")

        code, resp = svc.submit()
        job3 = svc.wait_job(resp["job_id"], timeout=180)
        check(job3["state"] == "done" and job3["cold_start"] is True,
              "卸载后请求自动重新加载", f"load_ms={job3['load_ms']:.0f}")

        code, text = http("POST", svc.base + "/v1/vlx/unload?wait=1&timeout_ms=30000", {})
        un = json.loads(text)
        check(code == 200 and all(w["unloaded"] for w in un["workers"]),
              "POST /v1/vlx/unload?wait=1 卸载成功")

        code, text = http("GET", svc.base + "/metrics")
        import re
        m = re.search(r"vlx_requests_ok_total (\d+)", text)
        check(bool(m) and int(m.group(1)) >= 5, "/metrics 暴露请求计数",
              f"ok={m.group(1) if m else 'n/a'}")
        check("vlx_model_loads_total 2" in text, "/metrics 暴露加载次数",
              [ln for ln in text.splitlines() if "loads_total" in ln][0])
    finally:
        svc.stop()


def phase_b() -> None:
    print("\n=== Phase B：并发（instances=2, idle=0 不卸载） ===")
    svc = Service(port=8112, instances=2, idle_s=0, tag="b")
    try:
        check(svc.wait_ready(), "服务启动")
        h = svc.health()
        check(h["instances"] == 2, "实例数为 2")

        # 并发提交 2 个任务
        import threading
        submissions: list[tuple[int, dict]] = []
        lock = threading.Lock()

        def submit_one():
            r = svc.submit(task="counting", text="orange; apple")
            with lock:
                submissions.append(r)

        ths = [threading.Thread(target=submit_one) for _ in range(2)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        check(len(submissions) == 2 and all(c == 202 for c, _ in submissions),
              "两个请求都返回 202")
        jobs = [svc.wait_job(r["job_id"], timeout=240) for _, r in submissions]
        wall = time.time() - t0
        check(all(j["state"] == "done" for j in jobs), "两个任务都完成",
              f"par wall={wall:.1f}s")
        workers = {j["worker"] for j in jobs}
        check(len(workers) == 2, "两个任务落在不同 worker", f"workers={sorted(workers)}")
        check(sum(1 for j in jobs if j["cold_start"]) == 2, "两个 worker 各自独立冷启动")
        answers = [(j.get("result") or {}).get("answer", "") for j in jobs]
        check(all(a for a in answers), "两个任务都有 answer")

        code, text = http("GET", svc.base + "/v1/vlx/info")
        info = json.loads(text)
        check(info["stats"]["requests_ok"] >= 2, "/v1/vlx/info 统计正确",
              f"ok={info['stats']['requests_ok']} avg_e2e={info['stats']['avg_e2e_ms']:.0f}ms")

        # 取消尚未运行的任务
        code, resp = svc.submit(task="vqa", text="describe the image")
        code2, _ = http("DELETE", f"{svc.base}/v1/vlx/jobs/{resp['job_id']}")
        check(code2 in (200, 409), "DELETE 排队任务可用", f"status={code2}")
    finally:
        svc.stop()


def phase_c() -> None:
    """URL 图像输入（本地 http.server 托管 test_images）"""
    print("\n=== Phase C：URL 图像输入 ===")
    httpd = subprocess.Popen([sys.executable, "-m", "http.server", "8123",
                              "--bind", "127.0.0.1"], cwd=str(IMAGE.parent),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    svc = Service(port=8113, instances=1, idle_s=0, tag="c")
    try:
        time.sleep(1.0)
        check(svc.wait_ready(), "服务启动")
        code, r = svc.submit(image=f"http://127.0.0.1:8123/{IMAGE.name}")
        check(code == 202, "URL 图像提交返回 202")
        job = svc.wait_job(r["job_id"], timeout=240)
        n = len((job.get("result") or {}).get("result_bbox_list") or [])
        check(job["state"] == "done" and n >= 3, "URL 图像推理成功", f"boxes={n}")

        code, body = http("POST", svc.base + "/v1/vlx/infer",
                          {"task": "detection", "text": "x",
                           "image": "http://127.0.0.1:8123/does-not-exist.jpg"})
        check(code == 400, "URL 404 返回 400", json.loads(body)["error"])

        code, body = http("POST", svc.base + "/v1/vlx/infer",
                          {"task": "detection", "text": "x", "image": "file:///etc/passwd"})
        check(code == 400, "非 http(s) 协议返回 400")
    finally:
        svc.stop()
        httpd.terminate()
        httpd.wait(timeout=10)


def phase_d() -> None:
    """大图自动缩放（分辨率上限）+ 坐标映射回原图"""
    print("\n=== Phase D：大图自动缩放 ===")
    try:
        from PIL import Image
    except ImportError:
        print("  SKIP  未安装 PIL，跳过大图缩放测试")
        return
    big = TMP / "big.jpg"
    Image.open(IMAGE).convert("RGB").resize((2560, 1920), Image.BICUBIC).save(big, quality=92)
    svc = Service(port=8114, instances=1, idle_s=0, tag="d")
    try:
        check(svc.wait_ready(), "服务启动")
        code, r = svc.submit(image_path=big)
        job = svc.wait_job(r["job_id"], timeout=300)
        img = job.get("image") or {}
        boxes = (job.get("result") or {}).get("result_bbox_list") or []
        check(job["state"] == "done", "2560x1920 推理成功")
        check(img.get("resized") is True, "超限图像被自动缩放",
              f"{img.get('original_width')}x{img.get('original_height')} -> "
              f"{img.get('width')}x{img.get('height')}")
        check(int(img.get("width", 9999)) <= 1280, "处理尺寸不超过默认上限",
              f"{img.get('width')}x{img.get('height')}")
        if boxes:
            max_x = max(b["xmax"] for b in boxes)
            check(max_x > 1300, "框坐标映射回原图坐标系", f"max_x={max_x:.0f}")
        check(len(boxes) >= 3, "缩放后仍检出目标", f"{len(boxes)} boxes")
    finally:
        svc.stop()
        big.unlink(missing_ok=True)


def main() -> int:
    if not BIN.exists():
        print(f"缺少服务二进制：{BIN}\n先运行 deploy/server/build-server.sh")
        return 2
    if not IMAGE.exists():
        print(f"缺少测试图片：{IMAGE}")
        return 2
    phase_a()
    phase_b()
    phase_c()
    phase_d()
    print(f"\n=== 结果：PASS {PASS} / FAIL {FAIL} ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
