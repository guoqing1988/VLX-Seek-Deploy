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
| CUDA | **编译/运行使用 toolkit 13.0.88**（`deploy/env.sh` 固定 `CUDA_HOME=/usr/local/cuda-13.0`，产物链接 `libcudart.so.13`）；用 `VLX_CUDA_ARCH` 指定架构，默认 `89`。`nvidia-smi` 报的 `CUDA Version: 13.0` 是驱动（580.173.02）支持上限，与本机 toolkit 一致；12.8 也验证过（输出逐 token 一致） |
| 编译 | gcc 13、cmake ≥ 3.14、32 线程编译约 2 分钟 |
| 上游源码 | `om-ai-lab/VLX-Seek`，且已 `git submodule update --init --recursive`（`cpp_ggml/third_party/llama.cpp`） |
| 模型权重 | 4 个 GGUF，约 12.6 GiB（**不在本仓库**）：`vlx-10b-llm-q8_0`、`vlx-vision-10b-{f16,q8_0}`、`wedetect-base-uni-f32` |
| 服务依赖 | 本机已验证：OpenSSL 开发包（HTTPS 图像 URL 需要）；无新增 Python 依赖 |

模型可从 Hugging Face `omlab/VLX-Seek-1.5-10B-GGUF` 获取，放到任意目录后用 `VLX_GGUF_DIR` 指向。

### 关于 CUDA 版本

- 当前构建/运行使用 **toolkit 13.0.88**（`nvcc 13.0`），`ldd` 显示 `libcudart.so.13` / `libcublas.so.13`
- `nvidia-smi` 的 `CUDA Version: 13.0` = 驱动（580.173.02）支持的最高 runtime 版本
- 本机另装 12.8.93 并**构建对比过**：同图 q8_0 greedy 推理 `answer` 逐 token 一致、
  token 数一致（411/19）、框坐标最大差 0.0000 px。两套 toolkit 都可复现
- 换 toolkit：`CUDA_HOME=/usr/local/cuda-12.8 ./deploy/build.sh` + `./deploy/server/build-server.sh`

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
sudo systemctl stop vlx-seek && python3 deploy/server/test_server.py   # 65 项验收（先让出显存）
```

服务同时提供两套接口：**原生异步**（推荐，支持并发与长任务）和 **OpenAI 兼容**（同步，方便接现成 SDK）。

原生异步（提交任务 → 轮询结果）：

```bash
curl -s localhost:8110/v1/vlx/infer -H 'Content-Type: application/json' -d '{
  "task":"detection","text":"orange; apple","lang":"zh",
  "image":"https://example.com/a.jpg"}'          # 也支持 data: URL / base64
# → 202 {"job_id":"job-1","state":"queued", ...}

curl -s localhost:8110/v1/vlx/jobs/job-1
# → {"state":"done","result":{"answer":"<ground>orange</ground>...","result_bbox_list":[...]},
#    "timings_ms":{...},"cold_start":false,"image":{"resized":false,...}}
```

运维接口：`GET /healthz`、`GET /v1/vlx/info`、`POST /v1/vlx/unload?wait=1`、`GET /metrics`、
`GET /v1/vlx/jobs/{id}/png`（下载带框图）。

### 任务（task）

7 个任务共用同一套输入/输出，区别在 `text` 的语义和是否需要区域索引：

| task | 用途 | `text` 写法 | bbox_list | target_region_indexes |
|---|---|---|---|---|
| `vqa` | 通用视觉问答（纯文本） | 自然语言问题 | 不需要 | 不需要 |
| `detection` | 开放词汇检测（多类多实例） | `orange; apple` | 自动/自传 | 不需要 |
| `grounding_single` | 指代表达定位（只给一个） | `the orange on the left` | 自动/自传 | 不需要 |
| `counting` | 计数 + 返回实例索引 | `person; car` | 自动/自传 | 不需要 |
| `reasoning_detection` | 先推理再定位（含 `<think>`） | 需要推理的问题 | 自动/自传 | 不需要 |
| `brief_region_caption` | 描述指定区域 | 占位文本 | 自动/自传 | **必填** |
| `region_ocr` | 读取指定区域文字 | 占位文本 | 自动/自传 | **必填** |

```json
POST /v1/vlx/infer
{"task":"detection","text":"orange; apple","lang":"zh","image":"data:image/jpeg;base64,..."}

→ {"answer":"<ground>orange</ground><objects><obj6><obj3>…</objects> <ground>apple</ground><objects><obj1><obj0></objects>",
   "result_bbox_list":[{"object_index":"<obj6>","label":"orange","xmin":13,"ymin":100,"xmax":141,"ymax":229}],
   "prompt_tokens":411,"completion_tokens":19}
```

- `answer` 是模型原始输出（检测/区域类含 `<ground>/<objects>/<objN>`；VQA 纯文本；推理类含 `<think>`）
- `result_bbox_list` 已把 `<objN>` 解析成**原图像素坐标**（缩放已映射回来）
- 不传 `bbox_list` 时由 WeDetect 自动生成 100 个候选框；区域任务的索引默认指向这些自动候选框

每个任务的真实输入输出、返回字段和常见坑见 [`deploy/SERVER.md` §3.5](deploy/SERVER.md)。

### 多图 / 渐进式 / 下载框图

- **多图**：VLX-Seek 是单图模型（区域 token 只对应本图候选框，不支持跨图推理）。
  多图 = 多次请求，异步接口天然并行（默认 2 实例）：
  `jobs = [submit(image=p) for p in paths]` → 各自轮询。
- **渐进式（先粗后细）**：`detection` 拿框 → 用 `bbox_list` + `target_region_indexes`
  对该框做 `region_ocr` / `brief_region_caption`。实测第二步 1.24 s、`load_ms=0`（模型常驻）。
- **下载带框图**：提交时 `render_png: true`，然后
  `curl -s -o result.png localhost:8110/v1/vlx/jobs/<job_id>/png`；
  或 CLI 默认写 `<image>_result.png`。PNG **始终按原图尺寸渲染**，框位置与 `result_bbox_list` 一致，
  内部缩放对调用者完全透明（无需换算）。

详见 [`deploy/SERVER.md` §3.6](deploy/SERVER.md)。

### OpenAI 兼容接口

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8110/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="vlx-seek-1.5-10b",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "orange; apple"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}],
    temperature=0,
    extra_body={"vlx_task": "detection", "vlx_lang": "zh"})
print(resp.choices[0].message.content)              # <ground>orange</ground><objects>...
print(resp.model_extra["vlx"]["result_bbox_list"])  # 框 + timings
```

- `GET /v1/models`、`POST /v1/chat/completions`（标准 `chat.completion` + 额外 `vlx` 块）
- 同步等待（`--openai-timeout-s`，默认 300 s）；`stream=true` 与 `temperature>0` 返回 400；
  `max_tokens` 被忽略（服务级 `--max-new-tokens`）
- 两套接口共用同一个任务队列与模型实例，并发/生命周期行为完全一致

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

## 实测性能（RTX 5880 Ada 48 GB / i9-14900K / CUDA toolkit 13.0）

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
| OpenAI 兼容接口 | Python openai SDK 实测通过（`/v1/models` + `chat.completions`） |
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
