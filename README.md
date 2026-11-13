# VLX-Seek-Deploy

[VLX-Seek 1.5-10B](https://github.com/om-ai-lab/VLX-Seek)（GGUF / CUDA 后端）的**部署脚本与常驻推理服务**。

本仓库只包含部署层，不含上游源码与模型权重：

- 构建脚本（CUDA + llama.cpp 内嵌 ggml）
- CLI 包装（自动定位模型目录、精度回退）
- 冒烟测试、显存/内存/吞吐压测
- **常驻 HTTP 服务**：原生异步接口、多实例并发、懒加载、空闲自动卸载、多卡选择
- systemd 托管（软链方式 + `.env` 环境变量）

> 上游项目：<https://github.com/om-ai-lab/VLX-Seek>
> 上游 C++/GGML 移植位于其 `cpp_ggml/` 目录。

---

## 目录结构

```
deploy/
├── env.sh                  # 路径/CUDA 常量（模型目录、构建目录）
├── build.sh                # 构建 cpp_ggml CUDA 后端 → cpp_ggml/build-cuda/
├── vlx.sh                  # 单次推理 CLI 包装（自动带 --gguf-dir、精度回退）
├── smoke.sh                # 7 个任务模板冒烟测试
├── bench.py                # 显存/内存/预填充/解码压测（按 PID 采样）
├── llama-server.sh         # 纯文本 OpenAI 兼容服务（llama.cpp 原生，无视觉）
├── DEPLOY.md               # 部署文档：构建、用法、实测性能、排障
├── SERVER.md               # 常驻服务文档：接口、生命周期、并发、分辨率、GPU
├── .env.example            # 服务环境变量模板（systemd EnvironmentFile）
├── vlx-seek.service        # systemd unit（软链安装）
├── install-service.sh      # systemd 安装/卸载/状态/重启
├── test_images/            # 自测图片（9 张）
└── server/
    ├── CMakeLists.txt      # 独立工程，链接上游 C API + vendor cpp-httplib
    ├── vlx_seek_server.cpp # 服务主程序（异步队列 + 多实例 + 生命周期管理）
    ├── build-server.sh     # 构建服务 → deploy/server/build/vlx-seek-server
    ├── serve.sh            # 启动包装（读 HOST/PORT/GPU/INSTANCES/…）
    └── test_server.py      # 40 项端到端验收（自动起停服务）
```

## 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，实测 RTX 5880 Ada（sm_89）48 GB；单实例需 9–14 GiB 显存 |
| CUDA | 12.8（`nvcc` 在 PATH；用 `VLX_CUDA_ARCH` 指定架构，默认 `89`） |
| 编译 | gcc 13、cmake ≥ 3.14、32 线程编译约 2 分钟 |
| 上游源码 | `om-ai-lab/VLX-Seek`，且已 `git submodule update --init --recursive`（`cpp_ggml/third_party/llama.cpp`） |
| 模型权重 | 4 个 GGUF，约 12.6 GiB（**不在本仓库**）：`vlx-10b-llm-q8_0`、`vlx-vision-10b-{f16,q8_0}`、`wedetect-base-uni-f32` |
| 服务依赖 | 本机已验证：OpenSSL 开发包（HTTPS 图像 URL 需要）；无新增 Python 依赖 |

模型可从 Hugging Face `omlab/VLX-Seek-1.5-10B-GGUF` 获取，放到任意目录后用 `VLX_GGUF_DIR` 指向。

## 快速开始

```bash
# 1. 准备上游源码
git clone https://github.com/om-ai-lab/VLX-Seek.git VLX-Seek-GGML
cd VLX-Seek-GGML
git submodule update --init --recursive

# 2. 把本仓库的 deploy/ 放到项目根目录
git clone https://github.com/guoqing1988/VLX-Seek-Deploy.git /tmp/VLX-Seek-Deploy
cp -r /tmp/VLX-Seek-Deploy/deploy .

# 3. 指向模型目录并构建
export VLX_GGUF_DIR=/data/models/VLX-Seek-1.5-10B-GGUF
./deploy/build.sh                       # ~2 分钟，产物 cpp_ggml/build-cuda/vlx-cli

# 4. 单次推理
./deploy/vlx.sh --task detection --text "orange; apple" --lang zh \
    --image-path deploy/test_images/demo_image.jpg

# 5. 功能自检（7 个任务模板）
./deploy/smoke.sh q8_0

# 6. 压测（显存 / 内存 / 吞吐）
python3 deploy/bench.py --precision q8_0 f16 --repeats 3 --warm-runs 0
```

> `deploy/` 必须位于 VLX-Seek 项目根目录下：脚本用 `deploy/../cpp_ggml` 定位上游构建。

## 常驻推理服务

```bash
./deploy/server/build-server.sh          # 构建服务（独立 CMake 工程）
./deploy/server/serve.sh --preload       # 默认 0.0.0.0:8110，2 实例，空闲 300s 卸载
python3 deploy/server/test_server.py     # 49 项验收（自动起停）
```

接口是**原生异步**（提交任务 → 轮询结果），不是 OpenAI 兼容：

```bash
curl -s localhost:8110/v1/vlx/infer -H 'Content-Type: application/json' -d '{
  "task":"detection","text":"orange; apple","lang":"zh",
  "image":"https://example.com/a.jpg"}'          # 也支持 data: URL / base64
# → 202 {"job_id":"job-1","state":"queued", ...}

curl -s localhost:8110/v1/vlx/jobs/job-1
# → {"state":"done","result":{"answer":"<ground>orange</ground>...","result_bbox_list":[...]},
#    "timings_ms":{...},"cold_start":false,"image":{"resized":false,...}}
```

运维接口：`GET /healthz`、`GET /v1/vlx/info`、`POST /v1/vlx/unload?wait=1`、`GET /metrics`。

### API key 认证

默认关闭；启用后保护 `/v1/vlx/*` 与 `/metrics`，`/healthz` 保持开放（探活）。

```bash
./deploy/server/serve.sh --api-key "sk-vlx-xxx"     # 或 deploy/.env 里 API_KEY=sk-vlx-xxx
# 多个 key 轮换：API_KEY=key1,key2

curl -H "Authorization: Bearer sk-vlx-xxx" localhost:8110/v1/vlx/info
curl -H "X-API-Key: sk-vlx-xxx"           localhost:8110/v1/vlx/info
```

未带或带错返回 `401` + `WWW-Authenticate: Bearer`；比较是恒定时间的，key 不出现在日志与 `/v1/vlx/info` 里。

### 生命周期

| 场景 | 行为 |
|---|---|
| 启动（默认） | 不加载，显存 0 |
| 第一个请求 | 懒加载（约 1.9 s）+ 推理 |
| 后续请求 | 复用 ctx，热请求 0.5–0.7 s |
| 空闲 `IDLE_UNLOAD_S`（默认 300） | 释放模型，显存回到 ~0.5 GiB |
| 卸载后请求 | 自动重载 |
| `--preload` | 启动即加载并预热（首请求也低延迟） |

### 并发

`--instances N` = N 个独立模型上下文 = **N 路真正并行**（第 N+1 个请求排队）。

- 每实例单线程串行；显存线性叠加（q8_0 下 9–14 GiB/实例，取决于输入分辨率）
- 2 实例实测：两个请求同时执行（766/771 ms），总 803 ms；第 3 个排队
- 队列上限 `--max-queue`（默认 64），满则 429

### 多卡

```bash
./deploy/server/serve.sh --gpu 1        # 只用物理卡 1（写入 CUDA_VISIBLE_DEVICES）
# 或 deploy/.env 里 GPU=1
```

多卡建议每卡起一个实例（不同 `PORT` + `GPU`），不要单进程跨卡。

### 分辨率上限（自动缩放）

超过 `--max-image-tokens`（默认 1200 ≈ 1280×960）的图像会**等比缩小后推理**，结果框坐标自动映射回原图。

| 输入分辨率 | 处理尺寸 | 端到端 | 单实例显存 |
|---|---|---|---|
| 640×480 | 原样 | 0.49 s | 12.0 GiB |
| 1280×960 | 原样 | 1.15 s | 14.4 GiB |
| 1920×1440 | 1280×960 | 1.15 s | 14.4 GiB |
| 3840×2160 | 1478×831 | 1.11 s | 14.6 GiB |

实测：常规目标（几十像素以上）缩放**无召回损失**；只有目标 < 16 px 时才需要
`--max-image-tokens 2000~3800`（代价是 1.9–5.3 s、16.6–19.1 GiB/实例）。详见 `deploy/SERVER.md` §6.1。

## systemd 托管

```bash
cp deploy/.env.example deploy/.env      # 按需修改 HOST/PORT/GPU/INSTANCES/EXTRA_ARGS
./deploy/install-service.sh install     # sudo ln -sfn + daemon-reload + enable + restart
./deploy/install-service.sh status
./deploy/install-service.sh restart     # 改完 .env 后
./deploy/install-service.sh uninstall
```

- 采用**软链**安装：`/etc/systemd/system/vlx-seek.service -> <repo>/deploy/vlx-seek.service`
- 环境变量在 `deploy/.env`（`EnvironmentFile`，`KEY=value`，不入库）
- `vlx-seek.service` 里的 `User=`、`WorkingDirectory=`、`EnvironmentFile=` 是绝对路径，**换机器需要改这三处**
- 日志：`journalctl -u vlx-seek -f`

## 实测性能（RTX 5880 Ada 48 GB / i9-14900K / CUDA 12.8）

| 项 | 数值 |
|---|---|
| 整链峰值显存（CLI，单请求） | 9.2–9.5 GiB（LLM q8_0 主导） |
| 进程峰值内存 | 10.7–11.6 GiB（GGUF mmap） |
| 解码吞吐 | ≈ 88 tok/s（单路） |
| 预填充 | 7.4k–7.9k tok/s |
| CLI 端到端（每次重载） | 3.3–4.1 s |
| 服务热请求 | **0.52–0.74 s** |
| 服务冷启动（懒加载） | 2.9 s；卸载后首请求同 |
| 2 实例显存 | 22.6 GiB（640×480 输入） |
| 精度对比（服务） | q8_0 11.7 GiB / 472 ms；f16 12.4 GiB / 473 ms，输出一致 |

## 精度选择（q8_0 / f16）

视觉链有 `f16` 和 `q8_0` 两套合并 GGUF；《LLM 只有 q8_0》，服务会自动回退。
两种精度在 CLI 层（7/7 冒烟 + 压测）和服务层（同一张图）都实测过：

| 配置 | 视觉 GGUF | 显存（服务，640×480 输入） | 端到端 | 输出 |
|---|---|---|---|---|
| `--precision q8_0`（默认） | 1.35 GiB | 11.7 GiB | 472 ms | 7 框 |
| `--precision f16` | 2.02 GiB | 12.4 GiB（+710 MiB） | 473 ms | 7 框，标签/坐标一致 |

q8_0 更省显存与加载时间，是默认值；需要极致数值保真时用 f16（官方 parity 判据里 q8_0 是独立量化档，
token 一致性由硬约束保证）。

## 上游修复

为了让 C API 支持常驻复用，本部署同时修复了 `cpp_ggml/src/vlx_seek_runtime.cpp` 的三个问题
（CLI 行为不变，详见 `deploy/SERVER.md` §8）：

| 问题 | 现象 | 修复 |
|---|---|---|
| CUDA 快路径漏 HFRE 融合 | `aux_proj projector failed` | 补 `cuda_fuse_dev` |
| 复用 ctx 不清 KV cache | 第二次 prefill 失败 | 每次 run 前 `llama_memory_clear` |
| 视觉图按首次输入 shape 永久复用 | 换分辨率越界/崩溃 | 记录 shape key，变化即重建 |

## 文档

- [`deploy/DEPLOY.md`](deploy/DEPLOY.md) — 构建、CLI 用法、性能实测、排障
- [`deploy/SERVER.md`](deploy/SERVER.md) — 常驻服务：接口、生命周期、并发、分辨率、GPU、systemd

## 许可

部署脚本与本文档随上游 [VLX-Seek](https://github.com/om-ai-lab/VLX-Seek) 的许可使用；
模型权重按其发布方的许可使用。
