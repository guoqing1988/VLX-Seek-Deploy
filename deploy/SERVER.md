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
./deploy/server/serve.sh               # 0.0.0.0:8110（本机 .env），2 实例, 空闲 300s 卸载
```

`serve.sh` 可用的环境变量：`PORT` `HOST` `INSTANCES` `IDLE_UNLOAD_S` `PRECISION` `DEVICE`，
其余参数透传：

```bash
# 单实例 + 5 分钟空闲卸载 + 预加载（低延迟优先）
INSTANCES=1 IDLE_UNLOAD_S=300 ./deploy/server/serve.sh --preload

# 对外监听 + 4 实例 + 详细日志
HOST=0.0.0.0 PORT=8200 INSTANCES=4 ./deploy/server/serve.sh --log-level 0

# 常用参数（完整列表：./deploy/server/build/vlx-seek-server --help）
#   --host H                 监听地址 [0.0.0.0]
#   --port N                 监听端口 [8100；本机 .env 设为 8110]
#   --device D               cuda|cpu|vulkan [cuda]
#   --gpu N                  只用第 N 张 GPU（写入 CUDA_VISIBLE_DEVICES）
#   --precision P            q8_0|f16 [q8_0]
#   --instances N            并发实例数，每实例约 12 GiB 显存 [2]
#   --idle-unload-s F        空闲卸载秒数，0=不卸载 [300]
#   --preload                启动即加载 + 预热
#   --tmp-dir D              临时文件目录 [<exe_dir>/tmp]
#   --threads N              CPU 线程，0=自动 [0]
#   --max-new-tokens N       生成长度上限（服务级，请求不能覆盖）[2048]
#   --max-image-tokens N     图像 token 预算，超出等比缩小；0=不缩放（受 n_batch 4096 限制）[1200]
#   --max-image-mb N         URL 下载图像上限 [16]
#   --image-url-timeout-s N  URL 下载超时秒 [10]
#   --image-proxy HOST:PORT  URL 下载代理（默认读 https_proxy/http_proxy）
#   --api-key KEY            API key 认证，多个用逗号分隔（默认读 API_KEY）
#   --openai-timeout-s N     /v1/chat/completions 同步等待上限 [300]
#   --max-queue N            队列上限，满则 429 [64]
#   --max-jobs N             内存保留任务数 [512]
#   --job-ttl-s F            已完成任务保留秒数 [3600]
#   --max-body-mb N          请求体上限 [32]
#   --log-level N            0=DEBUG 1=INFO 2=WARN 3=ERROR [1]
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
  "result": null, "timings_ms": null, "error": null,
  "has_png": false, "png_url": null
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
  "has_png": true,
  "png_url": "/v1/vlx/jobs/job-1/png"
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

默认关闭；启用后**所有接口都需要 key**（`/v1/vlx/*`、`/v1/models`、`/v1/chat/completions`、`/healthz`、`/metrics`），只有登录页 HTML 是静态的。

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
`job_id`、`task`、`result_bbox_list`、`timings_ms`、`cold_start`、`load_ms`、`image`、`png_url`（响应绝不内联图片本体）。

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
| `POST /v1/vlx/reload?wait=1` | **软重启**：卸载并重新加载模型（约 3–5 s/实例，含预热），返回每个 worker 的 `load_ms` |
| `GET /healthz` 的 `tps` | 三个口径：`last_decode_tps`（最近一次请求的**纯解码** tok/s，= completion_tokens / decode_ms，**推荐**）、`active_tps`（10s 内忙时平均，同样基于 decode 时间）、`tps`（10s/60s 窗口内完成 tokens / 窗口时长，**含空闲**，空闲后自然下降） |
| `GET /v1/vlx/jobs?limit=50&state=done` | 任务列表（新的在前） |
| `DELETE /v1/vlx/jobs/{id}` | 取消排队中的任务（运行中返回 409） |
| `GET /v1/vlx/jobs/{id}/png` | 下载带框的可视化 PNG（需提交时 `render_png=true`；未完成 409） |
| `GET /metrics` | Prometheus 文本：队列深度、请求数、加载/卸载次数、token 累计、每 worker 状态（启用 API key 时需带凭证） |

### 3.5 任务（task）详解

7 个任务共用同一条链路：**图像 + task + text（+ 可选候选框 / 区域索引）→ `answer` + `result_bbox_list`**。
区别只在 `text` 的语义、是否需要引用区域，以及 `answer` 的形态。

#### 通用输入字段

| 字段 | 说明 |
|---|---|
| `image` | 必填。`data:`/裸 base64 或 http(s) URL；超过 `--max-image-tokens` 会自动等比缩小 |
| `task` | 默认 `detection`，见下表 |
| `text` | 语义随任务变化（类别 / 指代表达 / 问题 / 区域索引占位），见各任务示例 |
| `lang` | `en`（默认）/ `zh`：只影响**任务模板**的语言，不限制回答语言 |
| `bbox_list` | 可选。`[[x1,y1,x2,y2],...]`（原图像素，最多 100）。省略时由 WeDetect 自动生成 100 个候选框 |
| `target_region_indexes` | 仅 `brief_region_caption` / `region_ocr` 需要。索引指向候选框（你传的 bbox_list 或自动候选框，从 0 开始） |

#### 7 个任务

| task | 用途 | `text` 怎么写 | 候选框 | 区域索引 | 实测耗时（640×480~1280×930） |
|---|---|---|---|---|---|
| `vqa` | 通用视觉问答，自由文本 | 自然语言问题 | 不需要 | 不需要 | 0.4–2.8 s（视图像大小） |
| `detection` | 开放词汇检测（多类别、多实例） | 类别/描述，`;` 分隔多个 | 自动或自传 | 不需要 | ~0.9 s |
| `grounding_single` | 指代表达定位，只给一个实例 | 一句指代（`the orange on the left`） | 自动或自传 | 不需要 | ~0.4 s |
| `counting` | 计数并返回对象索引 | 目标描述（可多个，`;` 分隔） | 自动或自传 | 不需要 | 1–4.3 s |
| `reasoning_detection` | 先推理再定位（输出 `<think>`） | 需要推理的问题/描述 | 自动或自传 | 不需要 | 4–6.7 s（输出长） |
| `brief_region_caption` | 描述指定区域 | 占位文本（如 `describe`，会被索引替换） | 自动或自传 | **必填** | ~1.8 s |
| `region_ocr` | 读取指定区域的文字 | 占位文本（如 `read`） | 自动或自传 | **必填** | ~1.8 s |

#### vqa — 通用问答

```json
{"task":"vqa","text":"图里有什么水果？","lang":"zh","image":"data:image/jpeg;base64,..."}
```

```json
{"answer":"图片中有苹果和橙子。","result_bbox_list":[]}
```

不引用任何候选框，`result_bbox_list` 恒为空；回答语言由问题本身决定。

#### detection — 开放词汇检测

```json
{"task":"detection","text":"orange; apple","lang":"zh","image":"data:image/jpeg;base64,..."}
```

```json
{"answer":"<ground>orange</ground><objects><obj6><obj3><obj2><obj5><obj4></objects> <ground>apple</ground><objects><obj1><obj0></objects>",
 "result_bbox_list":[
   {"object_index":"<obj6>","label":"orange","xmin":13,"ymin":100,"xmax":141,"ymax":229},
   {"object_index":"<obj3>","label":"orange","xmin":137,"ymin":107,"xmax":270,"ymax":232}]}
```

一个 `<ground>` 对应一个类别，`<objects>` 里是选中的候选框索引；`;` 分隔可一次查多类。
实测 7 个框、411 prompt tokens / 19 completion tokens。

#### grounding_single — 指代表达定位

```json
{"task":"grounding_single","text":"the orange on the left","lang":"en","image":"data:image/jpeg;base64,..."}
```

```json
{"answer":"<ground>the orange on the left</ground><objects><obj6></objects>",
 "result_bbox_list":[{"object_index":"<obj6>","label":"the orange on the left",
                      "xmin":13,"ymin":100,"xmax":141,"ymax":229}]}
```

只返回一个实例；`label` 是整句指代表达（不是类别词），下游要展示类别请自己映射。

#### counting — 计数

```json
{"task":"counting","text":"person; car","lang":"en","image":"data:image/jpeg;base64,..."}
```

```json
{"answer":"The total number of <ground>person</ground><objects><obj4><obj9><obj3>…</objects> is 11. "
          "The total number of <ground>car</ground><objects><obj21><obj49><obj33>…</objects> is 26.",
 "result_bbox_list":[{"object_index":"<obj4>","label":"person","xmin":788,"ymin":202,"xmax":848,"ymax":285}]}
```

- 数量看 `answer` 里的自然语言（`is 11`），框列表是它逐个数出来的实例（本例 37 个框）
- 输出比检测长（69 tokens），耗时也更高（4.3 s）

#### reasoning_detection — 先推理再定位

```json
{"task":"reasoning_detection","text":"which object is used for drinking?","lang":"en","image":"data:image/jpeg;base64,..."}
```

```json
{"answer":"<think> OK, the user needs us to detect the object used for drinking. … Step 2: …</think>…<ground>…</ground><objects><obj21><obj31></objects>",
 "result_bbox_list":[{"object_index":"<obj21>","label":"which object is used for drinking?",
                      "xmin":977,"ymin":671,"xmax":1007,"ymax":711}]}
```

- `answer` 先有 `<think>` 段，再给结论；需要展示推理过程时直接用它
- 输出可达 450 tokens、耗时 6.7 s；服务级 `--max-new-tokens` 要留足（默认 2048 够）

#### brief_region_caption — 区域描述

```json
{"task":"brief_region_caption","text":"describe","lang":"en",
 "image":"data:image/jpeg;base64,...","target_region_indexes":[0]}
```

```json
{"answer":"<ground>Clear plastic bottle with blue cap and label</ground><objects><obj0></objects>",
 "result_bbox_list":[{"object_index":"<obj0>","label":"Clear plastic bottle with blue cap and label",
                      "xmin":680,"ymin":189,"xmax":747,"ymax":396}]}
```

- 省略 `bbox_list` 时，`target_region_indexes` 指向 WeDetect 自动生成的候选框（与官方 CLI 一致）
- 传了 `bbox_list` 时，索引指向你传入的框（顺序即你给的顺序）

#### region_ocr — 区域文字识别

```json
{"task":"region_ocr","text":"read","lang":"en",
 "image":"data:image/jpeg;base64,...","target_region_indexes":[0]}
```

```json
{"answer":"<ground>3H86</ground><objects><obj0></objects>",
 "result_bbox_list":[{"object_index":"<obj0>","label":"3H86",
                      "xmin":94,"ymin":269,"xmax":620,"ymax":545}]}
```

识别结果在 `<ground>` 里；一块区域对应一个框。

#### 返回字段总表

| 字段 | 说明 |
|---|---|
| `answer` | 模型原始输出。检测/计数/区域类含 `<ground>标签</ground><objects><objN>…</objects>`；VQA 是纯文本；推理类含 `<think>` |
| `result_bbox_list` | 从 `answer` 解析出的框：`object_index`（对应的 `<objN>`）、`label`、`xmin/ymin/xmax/ymax`（**原图像素**，已按缩放比例映射回来） |
| `prompt_tokens` / `completion_tokens` | 输入（图像 token + 100 区域 token + 文本）/ 输出 token 数 |
| （服务层）`timings_ms` | `preprocess_ms`（含 WeDetect）/ `inference_ms` / `postprocess_ms` / `e2e_ms` |
| （服务层）`image` | `original_width/height`、`width/height`（实际送模型）、`resized`、`scale` |

#### 常见坑

1. **`label` 来自 `<ground>`**：`grounding_single` / `reasoning_detection` 里它是整句问题，不是类别词。
2. **候选框上限 100**：不传 `bbox_list` 时由 WeDetect 生成，密集/极小目标可能不在候选集里 → 自传 bbox_list 或提高分辨率预算。
3. **区域任务的索引语义**随输入变化：自传 `bbox_list` → 指向你的框；省略 → 指向自动候选框。同一索引在两种模式下不是同一个框。
4. **`vqa` 不需要候选框**，传 `bbox_list` 也会被忽略（但服务端仍会校验格式）。
5. **输出必须解析**：`<objN>` 是模型协议的一部分；`result_bbox_list` 已帮你解析并映射回原图。
6. **`reasoning_detection` 慢**：输出可达 450 tokens，端到端 6.7 s，不适合高并发低延迟场景。

### 3.6 常见用法模式

#### 多张图片

VLX-Seek 是**单图模型**：一次请求只处理一张图，区域 token 只对应当前图的 100 个候选框，
**不支持跨图推理**（比如"比较图 A 和图 B"）。多图就是多次请求，服务默认 2 实例并行执行：

```python
# 异步接口：N 张图立刻拿到 N 个 job_id，由 2 个实例并行处理
jobs = [submit(task="detection", text="person", image=b64(p)) for p in paths]
results = [wait(j["job_id"]) for j in jobs]     # 每个结果里带 worker 编号
```

- 异步接口天然支持：提交不阻塞，队列 FIFO，超 `--max-queue` 才 429
- OpenAI 接口是同步的，多图用线程池/进程池并发（服务端同样是 2 实例）
- 没有做"批量接口"因为模型不共享计算：批量只会排队，不会更快

#### 渐进式（先粗后细）

利用常驻模型把一次大请求拆成两步，第二步只看关心的区域——更快、更准，也省 token：

```
步骤 1  detection / grounding_single     → 拿到候选框（原图坐标）
步骤 2  用这些框 + target_region_indexes  → region_ocr / brief_region_caption
```

实测（`water_bottles_table.jpg`）：

| 步骤 | 请求 | 结果 |
|---|---|---|
| 1 | `detection`，`text="bottle"` | 6 个框，`cold_start=false` |
| 2 | `region_ocr`，`bbox_list=步骤1的框`，`target_region_indexes=[0]` | **1.24 s**，`load_ms=0`（模型常驻，无重载） |
| 2b | `brief_region_caption`，同一框 | `<ground>a bottle of water</ground>` |

```python
d = wait(submit(task="detection", text="bottle", lang="en", image=img)["job_id"])
boxes = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in d["result"]["result_bbox_list"]]

o = wait(submit(task="region_ocr", text="read", lang="en", image=img,
                bbox_list=boxes, target_region_indexes=[0])["job_id"])
print(o["result"]["answer"])          # <ground>...</ground><objects><obj0></objects>
```

要点：

- 第二步的 `target_region_indexes` 指向**你传入的 `bbox_list`**（顺序即你给的顺序），所以框的顺序要保留
- 模型常驻时第二步没有加载开销；空闲卸载后再调会先重载（2.9 s）
- 想对多个框逐个识别就循环提交（每个约 1.2 s），或并发（2 实例并行）

#### 下载带框的可视化图片

三种方式：

```bash
# 1) 异步接口：先提交时要求渲染，再用独立端点直接下载 PNG
curl -s localhost:8110/v1/vlx/infer -H 'Content-Type: application/json' \
  -d '{"task":"detection","text":"bottle","image":"data:image/jpeg;base64,...","render_png":true}'
# → 202 {"job_id":"job-7", ...}
curl -s -o result.png localhost:8110/v1/vlx/jobs/job-7/png    # image/png
```

```json
// 2) 响应只给路径：has_png / png_url（异步接口与 OpenAI 接口一致，不内联 base64）
{"task":"detection","render_png":true, ...}
// → "has_png": true, "png_url": "/v1/vlx/jobs/job-7/png"
```

```bash
# 3) CLI：默认就会写 <image>_result.png（--no-visualize 关闭）
./deploy/vlx.sh --task detection --text "bottle" --image-path a.jpg
ls a_result.png
```

- `GET /v1/vlx/jobs/{id}/png`：任务未完成 409；没请求 `render_png` 时 400 并提示；PNG 在 job 保留期内（`--job-ttl-s`，默认 1 h）可重复下载
- **`?format=jpeg&q=88`**：返回 JPEG（照片类图像体积约为 PNG 的 1/10，实测 363 KB → 46 KB）。
  响应里只给 `has_png` + `png_url`，从不内联图片本体（避免轮询/同步响应被几 MB 拖慢）
- PNG **始终按原图尺寸渲染**（`image.original_width × original_height`）：缩放只影响送进模型的分辨率，
  框坐标在服务内部已映射回原图，PNG 与 `result_bbox_list` 完全一致，调用者不需要换算
- 渲染样式与官方 `inference.py` 一致：红色 3 px 矩形 + 标签（非 ASCII 标签会退化成 `<objN>`）

### 3.7 Web 控制台（nginx :8188）

`deploy/web/index.html` 是一个单文件控制台，通过 nginx 反向代理到服务：

```
浏览器 :8188 ──► nginx ──┬─ /            → deploy/web/index.html（静态页面）
                         └─ /v1/、/healthz、/metrics → 127.0.0.1:8110
```

安装（软链，不复制配置）：

```bash
./deploy/nginx/install-nginx.sh install     # sudo ln -sfn + nginx -t + reload
./deploy/nginx/install-nginx.sh status
./deploy/nginx/install-nginx.sh uninstall
```

- 访问：`http://<本机IP>:8188/`
- **密码 = `deploy/.env` 里的 `API_KEY`**：页面把输入的 key 存进 `sessionStorage`，
  用 `GET /v1/vlx/info`（带 `Authorization: Bearer`）校验，实际把关的是服务端 API key 认证
- key 不写进 nginx 配置；轮换/停用只改 `.env` 并重启服务，页面重新登录即可
- nginx 配置：`deploy/nginx/vlx-seek.conf`（软链到 `/etc/nginx/conf.d/vlx-seek.conf`）

页面功能：

| 功能 | 说明 |
|---|---|
| 状态条 | `/healthz`：模型是否加载、实例数、队列深度、**最近解码 TPS**（登录后自动带 key） |
| 服务信息 | `/v1/vlx/info`：精度、实例数、认证状态、图像预算、成功/失败计数 |
| 多图上传 | 拖拽或选择，多张会逐个提交（服务端 2 实例并行） |
| 图片 URL | 每行一个 http(s) 地址 |
| 7 个任务 | 下拉选择，text 提示随任务变化；区域任务要求填 `target_region_indexes` |
| 高级参数 | `bbox_list`、`target_region_indexes`、是否返回带框图 |
| 结果展示 | `answer`（`<ground>/<objects>/<think>` 高亮）、框表格、耗时/tokens/冷启动/处理尺寸、**带框图** |

限制与注意：

- `client_max_body_size 32m`（与服务 `--max-body-mb` 一致），base64 大图不要超过
- 反代超时 300 s，匹配 `--openai-timeout-s`（同步 OpenAI 接口）
- 多图时不支持共用一组 `bbox_list`（页面会拦），单图提交即可
- nginx 监听 `8188`（所有网卡）：对外暴露前请配防火墙/安全组

### 3.8 错误码

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
| `--preload` | 启动即加载 + 预热 | 4.2 s（2 实例）/ 显存 22.6 GiB |

## 5. 并发与资源

| 项 | 数值 |
|---|---|
| 单实例显存 | **11.7 GiB**（LLM q8_0 8.87 GiB + KV/compute buffer；C API 固定 `n_ctx=8192`） |
| 2 实例 | **22.6 GiB**（进程内共享 CUDA context，增量约 11 GiB/实例） |
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
7. `render_png=true` 的可视化 PNG **始终按原图尺寸渲染**（缩放只影响推理，坐标已在服务内部映射回原图），
   所以 PNG 尺寸 = `image.original_width × original_height`，框位置与 `result_bbox_list` 一致，调用者不需要任何换算。

## 6.1 分辨率处理：最长边 1920（可配置）

规则（在 `deploy/.env` 配置，改完 `systemctl restart vlx-seek` 生效）：

1. **最长边 > `MAX_IMAGE_SIDE`（默认 1920）** → 等比缩放到该边长（结果对齐到 32 的倍数）
2. **否则完全不处理**（小图原样送模型）
3. `MAX_IMAGE_TOKENS`（默认 3800）是**兜底**：保证一次 prefill 放得下（runtime `n_batch=4096`），
   正常情况下边长规则先生效，兜底不会触发

```ini
# deploy/.env
MAX_IMAGE_SIDE=1920      # 0 = 不限制
MAX_IMAGE_TOKENS=3800    # 0 = 不限制
```

或者启动参数：`--max-image-side N` / `--max-image-tokens N`。

实测（q8_0、单实例、`detection`）：

| 输入分辨率 | 处理尺寸 | 是否缩放 | prompt tokens | 端到端 |
|---|---|---|---|---|
| 1000×700 | 1000×700 | 否 | 901 | 1.1 s |
| 1181×1575 | 1181×1575 | **否** | 2030 | 2.4 s |
| 1920×1080 | 1920×1080 | **否**（正好在上限） | 2197 | 2.6 s |
| 2560×1920 | 1920×1440 | 是 | 2919 | 3.7 s |
| 3840×2160 | 1920×1056 | 是 | 2137 | 2.5 s |

显存：单实例随处理尺寸增长，1000×700 ≈ 13 GiB，1920×1440 ≈ 19 GiB（2 实例约 38 GiB）。

### 坐标与可视化

- 缩放只影响"模型看到的像素"，**不影响坐标正确性**：服务内部按 `原图/处理图` 比例把
  `result_bbox_list` 映射回原图，PNG 也画在**原图**上（回归测试断言 PNG 红框位置与 result 一致，7/7）
- 缩放的真实影响是**候选框/召回**（WeDetect 在缩小图上生成 100 个候选框）

### 缩放对识别效果的影响（实测）

同一张 1181×1575 的图（目标：划船队员）：

| 处理方式 | 框数 | 说明 |
|---|---|---|
| 原样（1181×1575，2030 tokens） | 12 | 多检出 2 个窗户里的小目标 |
| 缩到 959×1280（旧默认 1200 token 口径） | 10 | 小目标少 2 个 |

经验阈值（以送进模型的像素计）：目标 **≥ 32 px（约 1 token）基本无损**；16–32 px 视对比度而定；
**< 16 px 明显掉召回**。所以默认把最长边限制在 1920，既控制了时间/显存，又几乎不损失常规目标。

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
python3 deploy/server/test_server.py    # 65 项断言，自动起停服务
```

覆盖：冷启动不加载 → 首请求冷启动标记/加载耗时/结果框 → 不同尺寸连续请求（bug #3 回归）→
热请求延迟 → 空闲卸载（显示 + 显存归零）→ 卸载后自动重载 → 手动卸载 → /metrics →
2 实例并发（不同 worker）→ 取消排队任务 → /v1/vlx/info 统计 → URL 图像输入/404/协议校验 →
2560×1920 自动缩放 + 处理尺寸 ≤1280 + 坐标映射回原图 →
API key：所有接口（含 healthz/metrics）都要 key，无 key/错 key 401、Bearer 与 X-API-Key、多 key 轮换 →
OpenAI 兼容：/v1/models、chat.completion + usage + vlx 框、无图/temperature>0/stream 400、vqa 自由文本 →
可视化：`render_png` 返回 base64、`GET /jobs/{id}/png` 下载（PNG 尺寸=原图）、未请求渲染时 400 →
渐进式：detection 的框喂给 brief_region_caption 且 `load_ms=0`（模型常驻）。

最近一次结果：`PASS 65 / FAIL 0`（含大图自动缩放、API key 认证、OpenAI 兼容接口；无 PIL 时大图组跳过）（日志在 `deploy/server/test-a.log`、`test-b.log`）。
