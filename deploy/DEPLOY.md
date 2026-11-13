# VLX-Seek 1.5-10B（GGML/CUDA）部署文档

> 部署日期：2026-11-13 ｜ 主机：RTX 5880 Ada 48GB / i9-14900K 32 线程 / 93GB 内存
> 部署方式：仓库自带 C++/GGML 后端（`cpp_ggml`），环境全部在项目内（`deploy/`、`cpp_ggml/build-cuda/`）

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| 是否 OpenAI 兼容接口 | **仓库交付物没有**（只构建 `vlx-cli` / `radio-cli` / C API，无任何 HTTP 服务）。但**子模块 llama.cpp 自带 `llama-server`（OpenAI 兼容）**：已额外编译到 `cpp_ggml/build-server/` 并用 `deploy/llama-server.sh` 提供纯文本接口；它无法承担 VLX 的视觉/定位推理（实测 `--mmproj` 直接失败，见 §6） |
| 能否跑起来 | 能。CUDA 后端全链路验证通过，7 个任务模板全部通过（q8_0 / f16 各 7/7） |
| 显存需求 | 单请求整链峰值 **≈ 9.2–9.5 GiB**（q8_0 LLM 主导）；视觉链阶段峰值 f16 1.5 GiB / q8_0 1.2 GiB；Wedetect 阶段 0.9 GiB |
| 内存需求 | 进程峰值 RSS **10.7–11.6 GiB**（GGUF 走 mmap，计入 page cache）；系统可用内存下降同量级 |
| 吞吐（RTX 5880 Ada，单路） | 解码 **≈ 88 tok/s**（稳定，q8_0/f16 视觉链相同，因为 LLM 都是 q8_0）；预填充 **7.4k–7.9k tok/s** |
| 常驻服务（`deploy/server/`） | 已用 systemd 托管（软链 + `deploy/.env`，默认 `0.0.0.0:8110`，多卡可用 `--gpu N` / `.env` 的 `GPU=` 指定；支持 `--api-key` 认证；同时提供 OpenAI 兼容的 `/v1/chat/completions`）；热请求端到端 **0.52–0.74 s**（CLI 3.3–4.1 s）；懒加载首请求 2.9 s；空闲 300 s 自动卸载、显存归零；2 实例 22.6 GiB；支持 base64/data URL/http(s) URL；超分辨率预算（默认 1280×960）自动等比缩小并映射回原图坐标（见 `SERVER.md` §6.1） |
| 并发吞吐 | 2 路并发合计 **137 tok/s**（单路 88 → 每路下降，显存线性叠加，2 路需 ≈ 19 GiB） |
| 单请求端到端（含冷启动） | 3.3–4.1 s（q8_0）/ 3.6–4.5 s（f16 视觉）；其中 LLM 权重加载 ≈ 1.2 s + GGUF 解析 ≈ 0.5 s |
| CPU 模式 | 不可用：视觉链 15–18 s、WeDetect 23.6 s、解码 **0.25 tok/s** |

## 2. 目录与环境

```
/data/www/VLX-Seek-GGML/
├── deploy/                  # 本部署目录
│   ├── env.sh               # 路径/工具链常量（模型目录、构建目录、CUDA）
│   ├── build.sh             # 构建 CUDA 后端（日志写 deploy/logs/）
│   ├── vlx.sh               # 启动包装（固定模型目录 + 精度回退）
│   ├── llama-server.sh      # 纯文本 OpenAI 兼容服务（llama.cpp llama-server，无视觉）
│   ├── smoke.sh             # 7 任务冒烟测试
│   ├── bench.py             # 显存/内存/tok-s 压测（按 PID 采样）
│   ├── server/              # 常驻推理服务（原生异步接口 + 并发，见 SERVER.md）
│   ├── web/index.html       # 单文件 Web 控制台（上传图片 / 跑 7 个任务 / 看框图）
│   ├── nginx/               # nginx 配置（:8188 反代 + 静态页面）与软链安装脚本
│   ├── .env                 # systemd EnvironmentFile（HOST=0.0.0.0 / PORT / INSTANCES…，git 忽略）
│   ├── .env.example         # 环境变量模板
│   ├── vlx-seek.service     # systemd unit（软链到 /etc/systemd/system）
│   ├── install-service.sh   # systemd 安装/卸载/状态/重启
│   ├── test_images/         # 模型目录里的测试图（9 张，从 zip 解出）
│   ├── logs/                # 构建 / 冒烟 / 压测日志
│   └── out/                 # 压测结果 JSON
├── cpp_ggml/build-cuda/     # 编译产物 vlx-cli / radio-cli（170 MB）
├── cpp_ggml/build-server/   # 额外编译的 llama.cpp 原生 llama-server（纯文本 HTTP）
└── cpp_ggml/third_party/llama.cpp   # 子模块（含唯一 ggml 源，180 MB）

模型（只读，不复制进项目）：
/data/www/models/VLX-Seek-1.5-10B-GGUF/
├── vlx-10b-llm-q8_0.gguf        8.87 GiB  LLM（Qwen3.5 10B 混合注意力）
├── vlx-vision-10b-f16.gguf      2.02 GiB  视觉合并链（mmproj+fpn+radio）
├── vlx-vision-10b-q8_0.gguf     1.35 GiB  视觉合并链量化版
└── wedetect-base-uni-f32.gguf   0.40 GiB  候选区域检测器（必须 f32）
```

工具链版本：gcc 13.1、cmake 3.22、**CUDA toolkit 13.0.88**（`nvcc` 13.0，产物链接 `libcudart.so.13`）、目标架构 `sm_89`。

### CUDA 版本口径

| 来源 | 显示 | 含义 |
|---|---|---|
| `nvidia-smi` 右上角 | `CUDA Version: 13.0` | **驱动支持的最高 runtime 版本**（驱动 580.173.02），不是已安装的 toolkit |
| `nvcc --version` | `release 13.0, V13.0.88` | 实际使用的编译器 |
| 构建日志 | `Found CUDAToolkit: /usr/local/cuda-13.0/include (13.0.88)` | 实际链接的 toolkit |
| `ldd vlx-seek-server` | `libcudart.so.13`、`libcublas.so.13` | 运行时加载的是 13.x 库 |

`deploy/env.sh` 把 `CUDA_HOME` 固定为 `/usr/local/cuda-13.0`（可用环境变量覆盖）。

另外 **CUDA 12.8.93 也构建并对比过**：同一张图、q8_0、greedy，
`answer` 逐 token 一致、token 数一致（411/19）、框坐标最大差 **0.0000 px**。
因此 12.8 与 13.0 两套 toolkit 都可用；当前默认是 13.0。

## 3. 构建（已执行，约 1 分 50 秒）

```bash
cd /data/www/VLX-Seek-GGML
./deploy/build.sh          # = cmake -DVLX_GGML_CUDA=ON + build，产物在 cpp_ggml/build-cuda
```

要点：
- 构建目录固定在项目内；换 GPU 架构只需 `VLX_CUDA_ARCH=xx ./deploy/build.sh`。
- `cmake` 配置阶段会自动重放 `cpp_ggml/third_party/llama-patches/*.patch`（强制 `GGML_PREC_F32` 用 IEEE fp32，避免 cuBLAS TF32 破坏 f32 精度契约），子模块处于"已打补丁的 dirty 状态"是预期行为。
- 唯一的 ggml 源是 `third_party/llama.cpp` 内嵌副本；子模块缺失时 CMake 直接报错，不会静默降级。
- 日志：`deploy/logs/build-<时间戳>.log`。

## 4. 运行

统一入口 `deploy/vlx.sh`（自动带 `--gguf-dir`，无需记模型路径）：

```bash
# 开放词汇检测（自动跑 WeDetect 出 100 个候选框）
./deploy/vlx.sh --task detection --text "orange; apple" --lang zh \
    --image-path deploy/test_images/demo_image.jpg

# 计数
./deploy/vlx.sh --task counting --text "person; car" --lang en \
    --image-path deploy/test_images/street_traffic.jpg

# 通用 VQA / 区域描述 / 区域 OCR
./deploy/vlx.sh --task vqa --text "图片里有什么？" \
    --image-path deploy/test_images/kitchen_modern.jpg
./deploy/vlx.sh --task brief_region_caption --text "describe" \
    --target-region-indexes '[0]' --image-path deploy/test_images/demo_image.jpg

# 只在视觉链稳态上做基准（权重常驻，剥离冷启动）
./deploy/vlx.sh --warm-runs 3 --task detection --text "bottle" --lang en \
    --image-path deploy/test_images/water_bottles_table.jpg
```

输出：stdout 为标准 JSON（`answer`、`result_bbox_list`、`prompt_tokens`、`completion_tokens`、`visualization_path`），stderr 为分阶段计时（`[time] ...`）。可视化默认写 `<image>_result.png`，用 `--no-visualize` 关闭。

### 4.1 精度组合（重要）

CLI 的 `--precision` 同时决定视觉链和 LLM 的 GGUF 文件名。本模型目录**只有 q8_0 的 LLM**，所以 `--precision f16` 会去找不存在的 `vlx-10b-llm-f16.gguf`。`deploy/vlx.sh` 已做自动回退：

```
[deploy] 未找到 vlx-10b-llm-f16.gguf，回退 --llm vlx-10b-llm-q8_0.gguf
```

也就是说：
- `--precision q8_0` → 视觉 q8_0 + LLM q8_0（推荐，省 0.3 GiB 显存）
- `--precision f16` → 视觉 f16 + LLM q8_0（精度优先；解码速度与 q8_0 完全一致，因为瓶颈在 LLM）
- 显式传 `--llm <path>` 时以调用方为准。

### 4.2 冒烟测试

```bash
./deploy/smoke.sh q8_0   # 7 个任务模板各跑一次 + JSON 校验
./deploy/smoke.sh f16
```

实测结果（`deploy/logs/smoke-*.log`）：

```
PASS  vqa                    boxes=0
PASS  detection              boxes=6
PASS  grounding_single       boxes=1
PASS  counting               boxes=6
PASS  reasoning_detection    boxes=6
PASS  brief_region_caption   boxes=1
PASS  region_ocr             boxes=1
--- 通过 7 / 失败 0
```

## 5. 性能实测

### 5.1 显存与内存

| 阶段/口径 | f16 视觉链 | q8_0 视觉链 | 说明 |
|---|---|---|---|
| Wedetect 阶段峰值 | 0.9 GiB | 0.9 GiB | 检测器固定 f32，阶段结束后释放 |
| 视觉链阶段峰值 | 1.5 GiB | 1.2 GiB | mmproj+fpn+radio+hfre 同时驻留 |
| **整链峰值显存** | **9.23 GiB**（9448 MiB） | **9.23 GiB** | LLM q8_0 权重 8.87 GiB + KV/compute buffer |
| 进程峰值 RSS | 11.2–11.6 GiB | 10.7–11.0 GiB | GGUF mmap 计入 page cache |
| 系统可用内存下降 | 同量级（≈11 GiB） | 同量级 | 换页缓存，非匿名内存 |

GPU 显存时间线（f16，单个请求）：

```
136ms   12 MiB    进程拉起
329ms  748 MiB    Wedetect conv 图
518ms  354 MiB    检测器释放
1273ms 1230 MiB   视觉链（mmproj/fpn/radio）
1414ms 1588 MiB   视觉链峰值（含 HFRE RoIAlign）
2943ms 8740 MiB   LLM q8_0 权重上传
3743ms 9440 MiB   KV + compute buffer → 峰值
4127ms  702 MiB   推理结束释放 LLM
```

建议：**留 12 GiB 显存**给单实例（9.5 GiB 峰值 + 余量）；系统内存按 **12 GiB** 预留。服务 `--max-image-tokens 0` 可关闭缩放（受 n_batch 4096 限制，≤2400×1600）；服务模式单实例约 12 GiB（`n_ctx=8192`），并发 N 路约 `12×N` GiB；CLI 单进程峰值 9.5 GiB。

### 5.2 吞吐（单路，3 张测试图 × 3 次重复，取中位数，`--warm-runs 0` 即真实单请求）

| 精度 | 图片 | prefill | decode | 端到端 wall |
|---|---|---|---|---|
| q8_0 | demo_image.jpg（小图） | 415 tok / 56 ms ≈ **7.4k tok/s** | **88.0 tok/s** | 3.3 s |
| q8_0 | street_traffic.jpg | 1383 tok / 175 ms ≈ 7.9k tok/s | **87.7 tok/s** | 4.0 s |
| q8_0 | water_bottles_table.jpg | 1303 tok / 169 ms ≈ 7.7k tok/s | **88.1 tok/s** | 4.1 s |
| f16 | demo_image.jpg | 415 tok / 58 ms ≈ 7.2k tok/s | 87.9 tok/s | 3.6 s |
| f16 | street_traffic.jpg | 1383 tok / 181 ms ≈ 7.6k tok/s | 87.5 tok/s | 4.4 s |
| f16 | water_bottles_table.jpg | 1303 tok / 167 ms ≈ 7.8k tok/s | 88.1 tok/s | 4.5 s |

结论：**解码 ≈88 tok/s** 与视觉精度无关（LLM 都是 q8_0）；f16 视觉链端到端慢 0.3 s，因为 GGUF 从 1.35 GiB 涨到 2.02 GiB（读盘+上传）。

并发（2 进程同时跑 counting，q8_0）：77.6 + 59.5 = **137.1 tok/s** 合计（单路 88 → 单路吞吐下降 12%/32%，总吞吐 +56%）。

### 5.3 延迟分解（q8_0，water_bottles_table.jpg，1303 prompt token，单次请求无 `--warm-runs`）

| 阶段 | 冷启动（每次进程） | 稳态（`--warm-runs`） |
|---|---|---|
| Wedetect 100 候选框 | 191 ms | —（每请求一次） |
| 视觉 GGUF 读盘+解析 | 453 ms | 0 |
| 视觉链 total | 965 ms（首轮含权重上传） | **550 ms** |
| ├ mmproj | 255 ms | 186 ms |
| ├ fpn | 31 ms | 21 ms |
| ├ radio | 375 ms | 188 ms |
| └ hfre（融合 CUDA RoIAlign） | 249 ms | 129 ms |
| LLM 权重加载 | 1210 ms | 0（CLI 每请求重载） |
| Prefill 1302 tok | 168 ms | 168 ms |
| Decode（21 tok，随任务长度变化） | 229 ms @ 87.5 tok/s | 同左 |
| 图片解码 / tokenize / 输出 | ≈ 300 ms | ≈ 300 ms |
| **端到端** | **3.5 s** | — |

> 小图（demo_image.jpg，415 prompt token）的视觉稳态是 **86 ms**，冷启首轮 332 ms（≈3.9x）；大图（1303 token）冷/热 ≈1.8x。`--warm-runs N` 会把视觉链额外重跑 N 次，所以带该参数的 wall 时间会比真实单请求长。

> CLI 是"一次进程一次请求"，所以每次请求都要付 1.2 s LLM 加载 + 0.5 s GGUF 解析。要压到亚秒级首字延迟必须走常驻服务（`vlx_seek_runtime` / C API 已支持 context 复用），当前仓库没有现成的服务进程。

### 5.4 CPU 对照（--device cpu）

| 指标 | 数值 |
|---|---|
| Wedetect | 23.6 s |
| 视觉链（稳态） | 15.2 s |
| Prefill 1303 tok | 2.6 s |
| Decode | **0.25 tok/s**（41 tok 用了 160 s） |
| 进程峰值 RSS | 10.7 GiB |

结论：CPU 只能做功能验证，不具备可用性。

## 6. HTTP / OpenAI 兼容接口：现状与实测

**先纠正一处容易误解的地方**：llama.cpp 子模块里确实自带完整的 `tools/server`（`llama-server`），
`/v1/models`、`/v1/completions`、`/v1/chat/completions` 路由都在（`tools/server/server.cpp:269-274`）。
但要分两层看：

### 6.1 本项目默认不构建它

`cpp_ggml/CMakeLists.txt` 对 llama.cpp 用的是 `add_subdirectory(... EXCLUDE_FROM_ALL)`，作为子项目时
`LLAMA_STANDALONE=OFF`，因此 `LLAMA_BUILD_SERVER / LLAMA_BUILD_EXAMPLES / LLAMA_BUILD_TOOLS` 默认全 OFF。
主构建产物只有 `vlx-cli`、`radio-cli`、`vlx-seek-capi-smoke`，没有 `llama-server`。

需要时单独构建（已在项目内完成，产物 `cpp_ggml/build-server/bin/llama-server`）：

```bash
cmake -S cpp_ggml/third_party/llama.cpp -B cpp_ggml/build-server \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build cpp_ggml/build-server --target llama-server --parallel $(nproc)
```

### 6.2 它能提供什么（已实测）

`deploy/llama-server.sh` 直接用 VLX 的 LLM 权重起服务，**是标准 OpenAI 兼容接口**：

```bash
./deploy/llama-server.sh &            # 127.0.0.1:8099
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"vlx","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":60}'
```

实测返回正常 chat.completion（`"我是小欧机器人，您好！"`），解码 **83.5 tok/s**（200 tok 生成，
与 `vlx-cli` 的 88 tok/s 同级）。但它是**纯文本**模型：`capabilities: ["completion"]`，没有图像输入。

### 6.3 它为什么跑不了 VLX 视觉链

实测证据：

```
$ llama-server -m vlx-10b-llm-q8_0.gguf --mmproj vlx-vision-10b-f16.gguf
E mtmd_init_from_file: error: Failed to load CLIP model from .../vlx-vision-10b-f16.gguf
E srv   load_model: failed to load multimodal model
```

原因有三条，缺一不可：

1. **vision GGUF 不是 clip 架构**：`vlx-vision-10b-*.gguf` 是本项目自定义的合并权重（`mmproj.*` + `radio.*` + `aux_proj.*`，708 个 tensor），llama.cpp 的 mtmd 只认它自己的 clip 模型格式。
2. **推理协议不兼容**：VLX 的 prompt 是 `<objN><objfeat>` 占位符，prefill 时把 mmproj 图像 embeds 和 HFRE 区域特征通过 `llama_batch.embd` 注入到指定位置（`vlx_cli.cpp:409 decode_entries`）。`llama-server` 没有"外部注入 embedding 序列"的公开接口，只有文本 token 输入。
3. **候选框和区域编码在 server 之外**：100 个 proposal 由 WeDetect 卷积图生成、区域 token 由 RADIO+HFRE 编码，这些都是 `vlx-cli` 进程内的 ggml 图，server 侧不存在。

### 6.4 VLX 能力的 HTTP 接口：已实现（`deploy/server/`）

| 方案 | 能力 | 状态 / 代价 |
|---|---|---|
| A. 薄 HTTP 包装：FastAPI 把 image+text 转成 `deploy/vlx.sh` 调用 | 真正的 VLX 检测/计数/VQA | 不需要了（B 已实现） |
| B. 常驻服务：C API 复用 context，原生异步接口 + 并发 + 空闲卸载 | VLX 全能力，热请求 **0.52 s**（CLI 3.3–4.1 s） | **已实现**：`deploy/server/`，文档 [`SERVER.md`](SERVER.md) |
| C. `llama-server`（纯文本 OpenAI 兼容） | 标准 OpenAI 兼容，但只有文本 LLM | 已就绪：`deploy/llama-server.sh` |

常驻服务的最小用法：

```bash
./deploy/server/build-server.sh          # 构建到 deploy/server/build/
./deploy/server/serve.sh --preload      # 默认 8100（本机 .env 设为 8110），2 实例，空闲 300s 卸载
curl -s localhost:8110/v1/vlx/infer -H 'Content-Type: application/json' \
  -d '{"task":"detection","text":"orange; apple","lang":"zh","image":"data:image/jpeg;base64,..."}'
# → 202 {"job_id":"job-1",...}   然后 GET /v1/vlx/jobs/job-1 轮询结果
```


## 7. 排障

| 现象 | 原因 / 处理 |
|---|---|
| `vlx-cli not built` | 先跑 `./deploy/build.sh` |
| `--precision f16` 报 `failed to open vlx-10b-llm-f16.gguf` | 模型目录只有 q8_0 LLM；用 `deploy/vlx.sh`（自动回退）或显式 `--llm .../vlx-10b-llm-q8_0.gguf` |
| `backend: CPU (...)` | 没传 `--device cuda`（`vlx-cli` 默认 cpu） |
| 显存不足 / OOM | 单实例预留 12 GiB；可用 `-ngl N` 少放几层到 GPU |
| `llama-server` 报 `Failed to load CLIP model` | 正常：vision GGUF 是项目自定义格式，不是 llama.cpp 的 clip；视觉请走 `deploy/vlx.sh` |
| `llama.cpp patch replay failed` | 子模块被手工改过，`git -C cpp_ggml/third_party/llama.cpp checkout -- .` 后重新构建 |
| GGUF 读解析慢 / 首次请求慢 | 正常：`vision setup` ≈0.5 s + `llm load` ≈1.2 s，属每次进程启动成本 |
| 输出坐标与 python 后端有 ±0.01 px 差异 | 已知 fp32 归约噪声（官方 parity 结论），不影响使用 |

## 8. 复现实验清单

```bash
cd /data/www/VLX-Seek-GGML
./deploy/build.sh                                  # 构建（约 2 分钟）
./deploy/smoke.sh q8_0 && ./deploy/smoke.sh f16    # 功能
python3 deploy/bench.py --precision q8_0 f16 --repeats 2 --warm-runs 3   # 显存/内存/tok-s
python3 deploy/bench.py --precision q8_0 --image water_bottles_table.jpg \
        --repeats 1 --device cpu                    # CPU 对照（约 4 分钟）

# 常驻推理服务（原生异步接口 + 并发 + 空闲卸载 + 自动重载）
./deploy/server/build-server.sh
sudo systemctl stop vlx-seek && python3 deploy/server/test_server.py   # 65 项验收（需先让出显存）
./deploy/server/serve.sh --preload                 # 0.0.0.0:8110, 2 实例

# 可选：纯文本 OpenAI 兼容服务（llama.cpp 原生 llama-server）
cmake -S cpp_ggml/third_party/llama.cpp -B cpp_ggml/build-server -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build cpp_ggml/build-server --target llama-server --parallel "$(nproc)"
./deploy/llama-server.sh &   # http://127.0.0.1:8099/v1/chat/completions
```

压测结果落在 `deploy/out/bench-*.json`，包含每轮的 `gpu_peak_mib`、`vm_hwm_mib`、
`sys_mem_drop_mib`、`prefill_*`、`decode_*` 以及完整 stderr 分阶段计时，可直接用于回归对比。
