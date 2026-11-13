# VLX-Seek 常驻推理服务（原生异步接口 + 并发）

> 代码位置：`deploy/server/` ｜ 产物：`deploy/server/build/vlx-seek-server`
> 定位：生产用常驻服务，模型只加载一次；空闲自动卸载；下一次请求自动重载。
> 接口形态：**原生异步**（提交任务 → 轮询结果），支持多实例并发。

## 1. 设计

```
                 ┌────────────── httplib（16 线程）──────────────┐
 HTTP 请求 ─────►│ POST /v1/vlx/infer   → 入队，返回 202+job_id │
                 │ GET  /v1/vlx/jobs/id → 查状态/结果            │
                 └───────────────────┬───────────────────────────┘
                                     │ FIFO 队列（--max-queue）
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        worker 0                worker 1               worker N-1     ← --instances
        vlx_seek_ctx            vlx_seek_ctx           vlx_seek_ctx   ← 每个 9–11 GiB 显存
        （独占，单线程）        （独占，单线程）        （独占，单线程）
              │                      │
              └── 空闲 ≥ --idle-unload-s → vlx_seek_free() 显存归零
                  下一次有任务 → 自动 load（约 1.9 s）
```

- **ctx 归属**：每个 worker 线程独占自己的 `vlx_seek_ctx`，加载/推理/释放全在该线程内，
  没有跨线程的模型访问（C API 本身单 ctx 单线程）。
- **共享状态**：任务表、队列、worker 状态快照由一把 mutex 保护；HTTP 线程只做校验+入队。
- **懒加载**：不 `--preload` 时启动不占显存；第一个请求触发加载（约 1.9 s），
  响应里 `cold_start=true`、`load_ms` 给出耗时。
- **空闲卸载**：worker 空转时按 `--idle-unload-s` 释放 ctx（默认 300 s，0=永不）。
- **预加载**：`--preload` 启动即加载并跑两次 dummy 推理，把 lazy 的视觉图也建好。

## 2. 构建与启动

```bash
cd /data/www/VLX-Seek-GGML
./deploy/server/build-server.sh        # 独立 CMake 工程，链接 cpp_ggml 的 C API
./deploy/server/serve.sh               # 127.0.0.1:8100, 2 实例, 空闲 300s 卸载
```

`serve.sh` 可用的环境变量：`PORT` `HOST` `INSTANCES` `IDLE_UNLOAD_S` `PRECISION` `DEVICE`，
其余参数透传：

```bash
# 单实例 + 5 分钟空闲卸载 + 预加载（低延迟优先）
INSTANCES=1 IDLE_UNLOAD_S=300 ./deploy/server/serve.sh --preload

# 对外监听 + 4 实例 + 详细日志
HOST=0.0.0.0 PORT=8200 INSTANCES=4 ./deploy/server/serve.sh --log-level 0

# 常用参数
#   --gguf-dir D         模型目录（serve.sh 已指向 /data/www/models/VLX-Seek-1.5-10B-GGUF）
#   --instances N        并发实例数，每实例约 9–11 GiB 显存 [2]
#   --idle-unload-s F    空闲卸载秒数，0=不卸载 [300]
#   --preload            启动即加载 + 预热
#   --max-new-tokens N   生成长度上限（服务级，请求不能覆盖）[2048]
#   --max-queue N        队列上限，满则 429 [64]
#   --max-jobs N         内存保留任务数 [512]
#   --job-ttl-s F        已完成任务保留秒数 [3600]
#   --max-body-mb N      请求体上限 [32]
#   --max-image-mb N     URL 下载图像上限 [16]
#   --image-url-timeout-s N  URL 下载超时秒 [10]
#   --image-proxy HOST:PORT  URL 下载代理（默认读 https_proxy/http_proxy，本机地址不走代理）
#   --api-key KEY            启用 API key 认证（多个用逗号分隔；默认读环境变量 API_KEY）
#   --openai-timeout-s N     /v1/chat/completions 同步等待上限 [300]
#   --max-image-tokens N 图像 token 预算，超出等比缩小 [1200 ≈ 1280x960]
#   --log-level N        0=DEBUG 1=INFO 2=WARN 3=ERROR [1]
```

## 3. 接口

### 3.1 提交任务（异步）

`image` 三种写法等价：`https://host/a.jpg`、`data:image/jpeg;base64,...`、裸 base64。


```http
POST /v1/vlx/infer
Content-Type: application/json
```

```json
{
  "task": "detection",
  "text": "橘子; 苹果",
  "lang": "zh",
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "bbox_list": [[12,20,180,210]],
  "target_region_indexes": [0],
  "render_png": false
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` | 是 | `http(s)://` URL、`data:image/...;base64,...` 或裸 base64（URL 下载受 `--max-image-mb` 16 MB / `--image-url-timeout-s` 10 s 限制）；超分辨率预算会自动缩小 |
| `task` | 否 | `detection`(默认) / `grounding_single` / `counting` / `reasoning_detection` / `brief_region_caption` / `region_ocr` / `vqa` |
| `text` | 否 | 目标词/指代/问题（VQA 必填才有意义） |
| `lang` | 否 | `en`(默认) / `zh` |
| `bbox_list` | 否 | 像素框 `[[x1,y1,x2,y2],...]`，最多 100 个；不传则由 WeDetect 生成 100 个候选框 |
| `target_region_indexes` | 区域任务必填 | `brief_region_caption` / `region_ocr` 的目标框索引 |
| `render_png` | 否 | `true` 时返回可视化 PNG（base64） |

**202 响应**（立刻返回）：

```json
{
  "job_id": "job-1", "state": "queued", "queue_position": 0,
  "worker": -1, "created_at_ms": 2186470040,
  "cold_start": false, "load_ms": 0.0,
  "request": {"task": "detection", "text": "橘子; 苹果", "lang": "zh",
              "n_boxes": 0, "target_region_indexes": [], "render_png": false,
              "image_bytes": 125520},
  "result": null, "timings_ms": null, "error": null, "png_base64": null
}
```

### 3.2 查询任务

```http
GET /v1/vlx/jobs/job-1
```

```json
{
  "job_id": "job-1",
  "state": "done",
  "worker": 0,
  "cold_start": true,
  "load_ms": 1976.0,
  "result": {
    "answer": "<ground>orange</ground><objects><obj4><obj1><obj0></objects>",
    "result_bbox_list": [
      {"object_index": "<obj4>", "xmin": 328.22, "ymin": 202.27,
       "xmax": 479.13, "ymax": 355.94, "label": "orange"}
    ],
    "prompt_tokens": 411, "completion_tokens": 19
  },
  "timings_ms": {"preprocess_ms": 210.1, "inference_ms": 712.4,
                 "postprocess_ms": 0.07, "serialization_ms": 0.0, "e2e_ms": 925.1},
  "error": null,
  "png_base64": null
}
```

- `state`：`queued` → `running` → `done` / `failed` / `cancelled`
- `result` 与官方 `inference.py` 的 JSON 完全一致（`vlx_seek_result_json`）
- `cold_start=true` 表示**这个请求**触发了模型加载，`load_ms` 是加载耗时
- 轮询建议：200–500 ms 一次，或任务 `done/failed/cancelled` 即停止

```http
GET  /v1/vlx/jobs?limit=50&state=done      # 任务列表（新的在前）
DELETE /v1/vlx/jobs/job-1                  # 取消排队任务；running 返回 409
```

### 3.2.1 GPU 选择（多卡）

```bash
# 方法一：命令行（进程级 CUDA_VISIBLE_DEVICES，视觉链和 llama.cpp 都只看到这一张卡）
./deploy/server/serve.sh --gpu 1

# 方法二：deploy/.env（systemd 用它，改完 restart 生效）
GPU=1

# 多卡并行：每张卡起一个服务实例，各自端口/实例数
GPU=0 PORT=8110 INSTANCES=2 ./deploy/server/serve.sh --preload
GPU=1 PORT=8111 INSTANCES=2 ./deploy/server/serve.sh --preload
```

- 实现是进程启动时 `setenv("CUDA_VISIBLE_DEVICES", N)`，必须在任何 CUDA 初始化之前，
  所以视觉链（ggml CUDA backend）和 llama.cpp 天然绑定同一张卡。
- 生效情况可以在接口里自证：`GET /v1/vlx/info` →
  `config.gpu` / `config.cuda_visible_devices` / `workers[].device`（如 `CUDA0`）。
- 卡号不存在时不会静默退化：加载失败并打印 `no CUDA-capable device is detected`
  （实测 `--gpu 1` 在单卡机器上就是这条）。
- 不指定时用默认卡（第 0 张 / 全部可见）。`--device cpu` 下该参数无意义。

### 3.2.2 API key 认证

默认关闭；启用后保护 `/v1/vlx/*` 与 `/metrics`，**`/healthz` 保持开放**（探活用）。

```bash
# 启用（二选一）
./deploy/server/serve.sh --api-key "sk-vlx-xxx"
# 或 deploy/.env（systemd 读这个）
API_KEY=sk-vlx-xxx
# 多个 key 轮换（逗号分隔）
API_KEY=key1,key2
```

```bash
# 客户端两种头都支持
curl -H "Authorization: Bearer sk-vlx-xxx" localhost:8110/v1/vlx/info
curl -H "X-API-Key: sk-vlx-xxx"           localhost:8110/v1/vlx/info
```

- 未带/带错：`401` + `WWW-Authenticate: Bearer`，响应体 `{"error":"missing or invalid API key"}`
- 比较是恒定时间的；key 不写日志，`/v1/vlx/info` 只暴露 `auth_enabled` 与头格式，不回显 key
- 修改 `.env` 后 `./deploy/install-service.sh restart`

### 3.3 OpenAI 兼容接口（同步）

同一进程内两套接口并存：原生异步（`/v1/vlx/*`，推荐，支持并发与长任务）与 OpenAI 兼容
（`/v1/chat/completions`，同步等结果，方便接入现成 SDK）。

| 端点 | 说明 |
|---|---|
| `GET /v1/models` | `{"object":"list","data":[{"id":"vlx-seek-1.5-10b",...}]}` |
| `POST /v1/chat/completions` | 同步推理；内部复用同一个任务队列 |

请求：取**最后一条 `role=user`** 的 `content`，数组形式里 `type=text` 是文本、`type=image_url`
是图像（base64 data URL 或 http(s) URL）。VLX 参数放顶层扩展字段。

```bash
curl -s localhost:8110/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "vlx-seek-1.5-10b",
  "messages": [{"role":"user","content":[
      {"type":"text","text":"orange; apple"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}],
  "temperature": 0,
  "vlx_task": "detection", "vlx_lang": "zh"
}'
```

Python OpenAI SDK（已实测）：

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
print(resp.choices[0].message.content)        # <ground>orange</ground><objects>...
print(resp.model_extra["vlx"]["result_bbox_list"])
```

响应是标准 `chat.completion`（`id/object/created/model/choices/usage`），额外带 `vlx` 块：
`job_id`、`task`、`result_bbox_list`、`timings_ms`、`cold_start`、`load_ms`、`image`、`png_base64`。

行为与限制：

- 同步等待，上限 `--openai-timeout-s`（默认 300 s），超时返回 504
- `temperature > 0` → 400（服务固定贪心解码）；`stream=true` → 400（不支持流式）
- `max_tokens` 被忽略，长度由服务级 `--max-new-tokens` 控制
- 必须有图像（VLX 是视觉模型），否则 400
- 错误体是 OpenAI 格式：`{"error":{"message":...,"type":...,"code":...}}`
- 启用 API key 时同样受保护（`/v1/models` 也要凭证）

### 3.4 运维接口

| 接口 | 说明 |
|---|---|
| `GET /healthz` | `model_loaded`、每 worker 的 `state/busy/load_ms/runs`、`queue_depth`。**不触发加载** |
| `GET /v1/vlx/info` | 配置 + 每 worker 状态 + 累计统计（ok/failed/cancelled、加载次数、平均 inference/e2e），含 `endpoints` 列表 |
| `POST /v1/vlx/unload?wait=1&timeout_ms=30000` | 强制卸载；`wait=1` 等到 worker 真正释放 |
| `GET /metrics` | Prometheus 文本：队列深度、请求数、加载/卸载次数、token 累计、每 worker 状态（启用 API key 时需带凭证） |

### 3.4 错误码

| HTTP | 场景 |
|---|---|
| 400 | JSON 非法 / task 未知 / image 缺失或 base64 非法 / bbox 格式错 / 区域任务缺参数 / 传了 `max_new_tokens` 或 `temperature` |
| 401 | 启用了 API key 但未带/带错凭证（`/healthz` 除外） |
| 404 | job 不存在 |
| 409 | 取消一个正在运行的任务 |
| 429 | 队列已满（`--max-queue`） |
| 500 | 内部错误（OpenAI 接口：任务失败也会转成 500 + OpenAI 错误体） |
| 504 | OpenAI 接口同步等待超时（`--openai-timeout-s`） |
| 任务内错误 | 任务 `state=failed`，`error` 字段给出原因（如 `model load failed: ...`、`image decode failed`） |

## 4. 模型生命周期（实测）

| 场景 | 行为 | 实测 |
|---|---|---|
| 启动（默认） | 不加载，显存 0 | 0 MiB |
| 第一个请求 | 懒加载 + 推理 | e2e **2.9 s**（load 1.98 s） |
| 后续同尺寸请求 | 复用 ctx | e2e **0.52 s** |
| 空闲 ≥ `idle-unload-s` | 释放 ctx | 显存回到 **~470 MiB**（进程基础） |
| 卸载后请求 | 自动重载 | e2e 2.9 s，`cold_start=true` |
| `POST /v1/vlx/unload` | 立即（或等当前任务结束后）卸载 | `workers: [{unloaded: true}]` |
| `--preload` | 启动即加载 + 预热 | 4.2 s（2 实例）/ 显存 22.2 GiB |

## 5. 并发与资源

| 项 | 数值 |
|---|---|
| 单实例显存 | **11.7 GiB**（LLM q8_0 8.87 GiB + KV/compute buffer；C API 固定 `n_ctx=8192`） |
| 2 实例 | **22.2 GiB**（进程内共享 CUDA context，增量约 10.5 GiB/实例） |
| 峰值（2 实例推理中） | 22.96 GiB |
| 48 GB 卡建议 | `--instances 2`（留 ~25 GB 余量）；4 实例约 33 GiB 也可，但注意与主机其它 CUDA 进程共存 |
| 并发行为 | 每实例串行；多实例并行。队列 FIFO，超出 `--max-queue` 返回 429 |

实测（detection，demo_image.jpg，411 prompt tokens / 19 输出 tokens，q8_0，2 实例预热）：

| 场景 | 结果 |
|---|---|
| preload 后首请求 | 925 ms（首次为真实尺寸重建视觉图，约 +0.2 s） |
| 第二个请求（同 worker） | **518 ms** |
| 2 并发 | 518 ms / 978 ms（总 wall 979 ms） |
| 4 并发 | 820 / 819 / 1578 / 1581 ms（两实例并行，后两个排队；总 wall 1.58 s） |
| 对比 CLI（每次重载） | 3.3–4.1 s → 服务热请求 **0.52 s，约 6–8×** |

## 6. 已知限制

1. **`max_new_tokens` / `temperature` 是服务级**：请求里传这两个会返回 400。服务始终贪心解码
   （与官方建议一致；`temperature>0` 时 C++ 与 Python 的采样流不一致）。
2. **单轮**：VLX 本身无多轮对话；接口就是一个请求一条任务。
3. **图像 URL 由服务端主动下载**：支持 http/https（含重定向与系统代理环境变量），
   有 16 MB / 10 s 上限；生产环境请配合网络隔离或白名单（服务端可被诱导访问内网地址）。
   本机地址（localhost/127.x）自动跳过代理。
4. **取消只对排队任务生效**：C API 无中断接口，运行中的任务会跑完（返回 409）。
5. **换图像尺寸会重建视觉图**（约 +0.2–0.4 s/次），这是正确性要求：
   上游 runtime 的图缓存按首次输入 shape 分配，复用不同尺寸会越界崩溃（见 §8）。
   生产建议固定分辨率，或按尺寸分桶部署。
6. 每个任务在内存里保留结果（默认 1 小时 / 512 条），`render_png=true` 时结果含 PNG base64。
7. `render_png=true` 且触发了缩放时，PNG 是**处理后的尺寸**（`image.width × image.height`），
   而 `result_bbox_list` 是**原图坐标**；要在原图上标注请按 `image.scale` 换算（或直接用 `result_bbox_list`）。

## 6.1 分辨率上限：超限自动缩放

**服务端会先把图缩到预算内再推理**（`--max-image-tokens`，默认 1200），不会因为图太大而拒绝请求。

内部规则（`cpp_ggml/src/image_prep.cpp` + 服务侧）：

- 主视觉塔 `smart_resize`：对齐 32 的倍数，**最小 65536 像素**（小图会放大）、最大 16.7M 像素；
  图像 token = `(h/32)×(w/32)`，1 token ≈ 32×32 像素。
- 服务在**提交阶段**估算 token 数，超过 `--max-image-tokens` 就用双线性等比缩小到预算内
  （`nw=floor(w·k)`，`k=sqrt(budget·1024/(w·h))`），缩小后的图才送进模型。
- 结果框坐标按 `原图/处理图` 比例**映射回原图坐标系**（响应里 `image.original_*` / `image.width/height` / `image.scale`）。
- RADIO 辅助塔在原分辨率（对齐 16，不降采样）上跑，所以缩放同时也把大图的显存/耗时压下来。
- WeDetect 固定输出 100 个候选框，区域 token 恒为 100（与分辨率无关）。

实测（q8_0、单实例、`detection`、7 个结果框、稳态请求）：

| 输入分辨率 | 自动处理尺寸 | prompt tokens | 端到端 | 单实例显存 |
|---|---|---|---|---|
| 320×240 | 放大到 256×256（最小 65536 px） | 297 | **0.43 s** | 11.4 GiB |
| 640×480 | 原样 | 491 | **0.49 s** | 12.0 GiB |
| 1280×960 | 原样（≈默认上限） | 1419 | **1.15 s** | 14.4 GiB |
| 1600×1200 | 缩到 1280×960 | 1419 | **1.15 s** | 14.4 GiB |
| 1920×1440 | 缩到 1280×960 | 1419 | **1.15 s** | 14.4 GiB |
| 2560×1920 | 缩到 1280×960 | 1419 | **1.14 s** | 14.4 GiB |
| 3840×2160（4K） | 缩到 1478×831 | 1353 | **1.11 s** | 14.6 GiB |

坐标一致性：把 640×480 的图放大 4 倍成 2560×1920 再送服务（自动缩回 1280×960），
框坐标乘回原图后与 4× 内插结果相差 **1–7 px**（缩放重采样误差）。

### 上限怎么选

| 预算 | 约等于 | 端到端 | 单实例显存 | 2 实例显存 | 适用 |
|---|---|---|---|---|---|
| `--max-image-tokens 800` | 1024×768 | ~0.8 s | ~13 GiB | ~26 GiB | 低延迟优先、目标较大 |
| **`1200`（默认）** | **1280×960** | **~1.1 s** | **14.4 GiB** | **~29 GiB** | 通用；48 GB 卡 2 实例很宽裕 |
| `2000` | 1600×1200 | ~1.9 s | 16.6 GiB | ~33 GiB | 需要更小的目标 |
| `3800` | 1920×1440 | ~3.0 s | 19.1 GiB | ~38 GiB | 保小目标，48 GB 卡勉强 2 实例 |
| 4K 原分辨率 | — | 不可用 | ~32 GiB | 不可行 | RADIO 在原分辨率跑，不建议 |

选 1200 的理由：demo 图从 320×240 到 4K 都稳定检出同样 7 个框，1280×960 已覆盖绝大多数
安防/桌面场景的目标尺寸；再往上只增加时间与显存，不换来明显召回提升。目标特别小（如远景车牌）
再调到 2000–3800，并相应减少 `--instances`。

> 文件大小（KB）几乎不影响推理，只影响上传/下载与 JPEG 解码（同分辨率 65 KB vs 716 KB 差 <10 ms）。

### 缩放对识别效果的影响（实测）

三个场景，同一张 demo 图构造，`detection "orange; apple"`，以 `--max-image-tokens 3800`
（能容纳的最大保真度）为参照，比较框的 IoU≥0.5 匹配：

| 场景 | 原图中目标大小 | 预算 1200（默认） | 预算 3800（参照） | 结论 |
|---|---|---|---|---|
| 4K 图、大目标（3840×2160） | 橘子约 300 px | 7 框，处理 1478×831 | 7 框，处理 2630×1479 | **完全一致 7/7**（坐标 IoU 匹配全中） |
| 1920×1440、4 个缩略图 | 橘子约 40 px | 28 框，处理 1280×960 | 28 框，处理 1920×1440 | **基本一致 28/28**（2000 预算下 27/28） |
| 4K 图、16 个极小缩略图 | 橘子约 25 px | 16 框（目标约 10 px） | 26 框（目标约 17 px） | **少召回约 38%**；16 个全是真目标、无假阳性 |

机制说明：

- 缩放后**模型和 WeDetect 都在缩小图上工作**，小目标先丢掉的是候选框（WeDetect 固定 100 个 proposal），
  这是主要的损失来源；RADIO 辅助塔同样按缩小后的分辨率取细节。
- 经验阈值（以送进模型的像素计）：目标 **≥ 32 px（≈1 个 token）基本无损**；
  **16–32 px 视对比度而定**；**< 16 px 明显掉召回**。
- 注意 3840×2160 想全保真需要约 8100 图像 token，超过服务 `n_batch=4096` 的硬上限，
  所以 3800 已经是"能开到的最高保真度"。

结论：**默认 1200 对常规目标（几十像素以上）没有可测量的损失**；只有业务里存在
小于约 16 px 的目标（远景车牌、货架小件）时才需要提高预算，而且代价是延迟（1.9–5.3 s）
和显存（16.6–19.1 GiB/实例，建议 `--instances 1`）。这类场景更推荐先 ROI 裁剪再送模型
（局部放大等于提高有效分辨率），比整体提高预算更省。

## 7. 排障

| 现象 | 处理 |
|---|---|
| `vlx-seek-server: 未构建` | 先 `./deploy/server/build-server.sh` |
| 任务 `failed: model load failed` | 检查 `--gguf-dir`、显存是否足够（日志会打印 `device admission failed`） |
| `/healthz` 一直 `model_loaded=false` | 没有请求进来，或被空闲卸载（正常）；看 `idle_unload_s` |
| 429 | 队列满：调大 `--max-queue` 或加 `--instances` |
| 请求 400 `image is not valid base64` | 只接受 `data:` 或裸 base64 |
| 卸载后显存仍占 ~470 MiB | 进程基础/CUDA context 预留，不是模型（模型 9–11 GiB 已释放） |
| 想常驻不卸载 | `--idle-unload-s 0` |
| 想开机自启 | 见下方 systemd 示例（未安装，按需使用） |

### systemd 管理（已安装，软链方式）

```bash
./deploy/install-service.sh install     # sudo ln -sfn + daemon-reload + enable + restart
./deploy/install-service.sh status      # 软链 + 服务状态 + /healthz
./deploy/install-service.sh restart     # 改完 deploy/.env 后重启
./deploy/install-service.sh uninstall   # 停服务并删除软链
```

- **软链不做复制**：`/etc/systemd/system/vlx-seek.service -> <repo>/deploy/vlx-seek.service`，
  改 unit 文件后 `daemon-reload` 即生效，避免两份文件不同步。
- **环境变量**：`deploy/.env`（systemd `EnvironmentFile`，`KEY=value` 格式），模板见 `deploy/.env.example`；
  `.env` 被 git 忽略，模板入库。支持
  `HOST/PORT/GPU/INSTANCES/IDLE_UNLOAD_S/PRECISION/DEVICE/IMAGE_PROXY/API_KEY/EXTRA_ARGS`。
- **默认监听 `0.0.0.0`**（不再只绑本机）：上线前确认防火墙/安全组，或把 `deploy/.env` 的 `HOST` 改回 `127.0.0.1`。
- 日志：`journalctl -u vlx-seek -f`；异常退出 `Restart=on-failure`；`systemctl stop` 会走 SIGTERM 释放模型。

当前机器实测：软链已安装，`0.0.0.0:8110`，2 实例 preload 3.6 / 3.7 s，显存 22.6 GiB，
热请求（640×480 图，no scale）e2e 737 ms。端口被占用时改 `.env` 的 `PORT` 再 `restart`。

## 8. 为跑通常驻服务修复的上游 bug

`cpp_ggml/src/vlx_seek_runtime.cpp`（C API 实现）此前只在"一次进程一次请求"的 CLI 场景用过，
连续/异构请求会暴露三个问题，均已修复（CLI 行为不变）：

| # | 问题 | 现象 | 修复 |
|---|---|---|---|
| 1 | CUDA 快路径缺少 HFRE 融合 | 所有请求 `aux_proj projector failed` | 补 `cuda_fuse_dev`（与 `vlx_cli.cpp` 一致） |
| 2 | 复用一个 ctx 时不清 KV cache | 第二次 prefill `decode: failed to initialize batch` | 每次 run 前 `llama_memory_clear(mem, true)` |
| 3 | 视觉/检测图按首次输入 shape 缓存后永久复用 | 换图片尺寸后越界（进程崩溃或任务挂起/生成失控） | 记录 shape key（image/grid/aux），变化即重建对应图 |

## 9. 验收测试

```bash
python3 deploy/server/test_server.py    # 29 项断言，自动起停服务
```

覆盖：冷启动不加载 → 首请求冷启动标记/加载耗时/结果框 → 不同尺寸连续请求（bug #3 回归）→
热请求延迟 → 空闲卸载（显示 + 显存归零）→ 卸载后自动重载 → 手动卸载 → /metrics →
2 实例并发（不同 worker）→ 取消排队任务 → /v1/vlx/info 统计 → URL 图像输入/404/协议校验 →
2560×1920 自动缩放 + 处理尺寸 ≤1280 + 坐标映射回原图 →
API key：healthz 豁免、无 key/错 key 401、Bearer 与 X-API-Key、多 key 轮换 →
OpenAI 兼容：/v1/models、chat.completion + usage + vlx 框、无图/temperature>0/stream 400、vqa 自由文本。

最近一次结果：`PASS 59 / FAIL 0`（含大图自动缩放、API key 认证、OpenAI 兼容接口；无 PIL 时大图组跳过）（日志在 `deploy/server/test-a.log`、`test-b.log`）。
