// VLX-Seek 常驻推理服务（原生异步接口 + 多实例并发）
//
// 设计要点
//   * 每个 worker 线程独占一个 vlx_seek_ctx（C API 明确单 ctx 单线程），
//     因此 ctx 的加载/推理/释放全部发生在所属 worker 线程内，不存在跨线程
//     访问；共享状态（任务表、队列、状态快照）由一把 mutex 保护。
//   * 懒加载：请求到达且该 worker 尚未加载时才加载；加载耗时写进响应
//     （cold_start / load_ms），便于客户端区分冷启动。
//   * 空闲卸载：worker 空转时按 --idle-unload-s 释放 ctx，显存归零；
//     下一次请求自动重新加载。也可以 POST /v1/vlx/unload 强制卸载。
//   * 异步：POST /v1/vlx/infer 立刻返回 202 + job_id，结果用
//     GET /v1/vlx/jobs/<id> 轮询获取。
//
// 构建：deploy/server/build-server.sh（产物 deploy/server/build/vlx-seek-server）

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "vlx/vlx_seek_capi.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kVersion = "1.0.0";
constexpr const char* kTasks[] = {"vqa",          "detection",           "grounding_single",
                                  "counting",     "reasoning_detection", "brief_region_caption",
                                  "region_ocr"};

double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now().time_since_epoch())
        .count();
}

bool known_task(const std::string& t) {
    for (const char* k : kTasks)
        if (t == k) return true;
    return false;
}

bool region_task(const std::string& t) {
    return t == "brief_region_caption" || t == "region_ocr";
}

// ---------------- 日志 ----------------
std::mutex g_log_mu;
int g_log_level = 1;  // 0=DEBUG 1=INFO 2=WARN 3=ERROR

void logmsg(int level, const char* fmt, ...) {
    if (level < g_log_level) return;
    static const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> lk(g_log_mu);
    std::fprintf(stderr, "[%s] %s\n", names[std::min(level, 3)], buf);
    std::fflush(stderr);
}

// ---------------- base64 ----------------
bool b64_decode(const std::string& in, std::string* out) {
    static signed char tbl[256];
    static bool init = false;
    if (!init) {
        std::fill(std::begin(tbl), std::end(tbl), static_cast<signed char>(-1));
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tbl[static_cast<unsigned char>(a[i])] = static_cast<signed char>(i);
        init = true;
    }
    out->clear();
    out->reserve(in.size() / 4 * 3);
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const signed char d = tbl[c];
        if (d < 0) return false;
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0) {
            out->push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return true;
}

// data:image/jpeg;base64,xxxx 或纯 base64
bool decode_image_field(const std::string& s, std::string* bytes, std::string* err) {
    std::string body = s;
    const size_t comma = body.find(',');
    if (body.rfind("data:", 0) == 0 && comma != std::string::npos) body = body.substr(comma + 1);
    if (body.rfind("http://", 0) == 0 || body.rfind("https://", 0) == 0) {
        *err = "remote image URLs are not supported; send base64 or a data: URL";
        return false;
    }
    if (!b64_decode(body, bytes)) {
        *err = "image is not valid base64";
        return false;
    }
    if (bytes->empty()) {
        *err = "image decoded to zero bytes";
        return false;
    }
    return true;
}

// 与 cpp_ggml/src/image_prep.cpp::smart_resize 完全一致：
// factor=32, min_pixels=65536, max_pixels=16777216；图像 token = (hb/32)*(wb/32)。
int estimate_image_tokens(int height, int width) {
    const int factor = 32;
    const double min_pixels = 65536.0, max_pixels = 16777216.0;
    int hb = int(std::nearbyint(double(height) / factor)) * factor;
    int wb = int(std::nearbyint(double(width) / factor)) * factor;
    if (double(hb) * wb > max_pixels) {
        const double beta = std::sqrt(double(height) * width / max_pixels);
        hb = std::max(factor, int(std::floor(height / beta / factor)) * factor);
        wb = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    } else if (double(hb) * wb < min_pixels) {
        const double beta = std::sqrt(min_pixels / (double(height) * width));
        hb = int(std::ceil(height * beta / factor)) * factor;
        wb = int(std::ceil(width * beta / factor)) * factor;
    }
    return (hb / factor) * (wb / factor);
}

// 双线性缩放到目标尺寸（宽高比由调用方保持）
void resize_rgb_bilinear(const uint8_t* src, int sw, int sh, int dw, int dh,
                         std::vector<uint8_t>* dst) {
    dst->resize(size_t(dw) * dh * 3);
    const double sx = double(sw) / dw, sy = double(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        fy = std::clamp(fy, 0.0, double(sh - 1));
        const int y0 = int(fy);
        const int y1 = std::min(y0 + 1, sh - 1);
        const double wy = fy - y0;
        for (int x = 0; x < dw; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            fx = std::clamp(fx, 0.0, double(sw - 1));
            const int x0 = int(fx);
            const int x1 = std::min(x0 + 1, sw - 1);
            const double wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                const double v00 = src[(size_t(y0) * sw + x0) * 3 + c];
                const double v01 = src[(size_t(y0) * sw + x1) * 3 + c];
                const double v10 = src[(size_t(y1) * sw + x0) * 3 + c];
                const double v11 = src[(size_t(y1) * sw + x1) * 3 + c];
                const double v = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                                 wy * ((1 - wx) * v10 + wx * v11);
                (*dst)[(size_t(y) * dw + x) * 3 + c] =
                    uint8_t(std::clamp(std::lround(v), 0L, 255L));
            }
        }
    }
}

// ---------------- 配置 ----------------
struct ServerOptions {
    std::string host = "0.0.0.0";
    int port = 8100;
    int instances = 2;
    double idle_unload_s = 300.0;
    bool preload = false;
    std::string device = "cuda";
    std::string gpu;              // 物理 GPU 索引（写入 CUDA_VISIBLE_DEVICES）
    std::string precision = "q8_0";
    std::string gguf_dir;
    std::string tmp_dir;
    int threads = 0;
    int max_new_tokens = 2048;
    int max_queue = 64;
    size_t max_jobs = 512;
    double job_ttl_s = 3600.0;
    int max_body_mb = 32;
    int max_image_mb = 16;        // URL 下载的图像上限
    int max_image_tokens = 1200;  // 超过即等比缩小（1200 ≈ 1280x960，1.2 MP）
    int image_url_timeout_s = 10; // URL 下载超时
    std::string image_proxy;      // 可选 HTTP(S) 代理 host:port（默认读环境变量）
    int log_level = 1;
};

enum class JobState { Queued, Running, Done, Failed, Cancelled };

const char* state_name(JobState s) {
    switch (s) {
        case JobState::Queued: return "queued";
        case JobState::Running: return "running";
        case JobState::Done: return "done";
        case JobState::Failed: return "failed";
        case JobState::Cancelled: return "cancelled";
    }
    return "unknown";
}

struct Job {
    std::string id;
    JobState state = JobState::Queued;
    std::string image_path;
    std::string task, text, lang;
    std::vector<float> boxes;      // 4*n
    std::vector<int32_t> targets;
    bool render_png = false;
    size_t image_bytes = 0;
    // 超限自动缩放：scaled_rgb 只在 worker 执行期间使用，完成后释放
    std::vector<uint8_t> scaled_rgb;
    int orig_w = 0, orig_h = 0;      // 原图尺寸（结果坐标回到这个坐标系）
    int proc_w = 0, proc_h = 0;      // 实际送进模型的尺寸
    bool resized = false;
    int64_t created_ms = 0, started_ms = 0, finished_ms = 0;
    int worker = -1;
    json result;                   // 官方 inference.py JSON
    json timings;                  // 分阶段耗时
    bool cold_start = false;
    double load_ms = 0.0;
    std::string error;
    std::string png_b64;
};

struct TempFile {
    std::string path;
    ~TempFile() {
        if (!path.empty()) {
            std::error_code ec;
            fs::remove(path, ec);
        }
    }
};


}  // namespace

// ============================================================================
// 服务主体
// ============================================================================
class VlxServer {
  public:
    explicit VlxServer(ServerOptions opts) : opts_(std::move(opts)) {}
    ~VlxServer() { request_stop(); join_workers(); }

    int run();
    void request_stop();

  private:
    struct Worker {
        int idx = 0;
        std::thread th;
        // 仅属于本线程的 ctx：加载/推理/释放都在 worker 线程内完成
        vlx_seek_ctx* ctx = nullptr;       // 只在本线程访问
        double ctx_load_ms = 0.0;
        std::atomic<int64_t> last_used_ms{0};
        uint64_t runs = 0;
        bool busy = false;                 // 仅本线程读写（用于状态快照）
        std::atomic<bool> unload_requested{false};
    };

    struct WorkerSnapshot {
        std::string state = "unloaded";
        bool busy = false;
        double load_ms = 0.0;
        int64_t last_used_ms = 0;
        uint64_t runs = 0;
        std::string last_error;
        std::string device;   // backend 解析到的设备名，如 CUDA0
    };

    // ---- worker 侧 ----
    void worker_loop(int idx);
    void run_job(Worker& w, const std::shared_ptr<Job>& job);
    vlx_seek_ctx* load_ctx(std::string* err);
    bool fetch_image_url(const std::string& url, std::string* out, std::string* err) const;
    void warmup_ctx(vlx_seek_ctx* ctx);
    void unload_ctx(Worker& w);
    void maybe_unload(Worker& w);
    void publish_state(Worker& w, const char* state);

    // ---- HTTP ----
    void install_routes();
    void handle_infer(const httplib::Request& req, httplib::Response& res);
    void handle_job_get(const httplib::Request& req, httplib::Response& res);
    void handle_job_list(const httplib::Request& req, httplib::Response& res);
    void handle_job_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_info(const httplib::Request& req, httplib::Response& res);
    void handle_unload(const httplib::Request& req, httplib::Response& res);
    void handle_metrics(const httplib::Request& req, httplib::Response& res);

    // ---- 内部 ----
    json job_json(const std::shared_ptr<Job>& job, bool with_queue_pos) const;
    std::shared_ptr<Job> find_job(const std::string& id) const;
    void cleanup_jobs();
    void join_workers();
    std::string models_info_json() const;

    ServerOptions opts_;
    httplib::Server http_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> job_order_;  // 清理用：按创建顺序
    std::vector<WorkerSnapshot> snapshots_;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::thread reaper_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> next_job_{1};

    // 统计
    std::atomic<uint64_t> total_ok_{0};
    std::atomic<uint64_t> total_failed_{0};
    std::atomic<uint64_t> total_cancelled_{0};
    std::atomic<uint64_t> load_count_{0};
    std::atomic<uint64_t> unload_count_{0};
    std::atomic<uint64_t> total_prompt_tokens_{0};
    std::atomic<uint64_t> total_completion_tokens_{0};
    std::atomic<uint64_t> finished_runs_{0};
    // 由 mu_ 保护（只在持锁时读写）
    double sum_inference_ms_ = 0.0;
    double sum_e2e_ms_ = 0.0;
};

// ============================================================================
// 生命周期
// ============================================================================
void VlxServer::request_stop() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) return;
    cv_.notify_all();
    http_.stop();
}

void VlxServer::join_workers() {
    for (auto& w : workers_) {
        if (w->th.joinable()) w->th.join();
    }
    if (reaper_.joinable()) reaper_.join();
}

int VlxServer::run() {
    if (opts_.gguf_dir.empty()) {
        logmsg(3, "gguf dir is empty");
        return 2;
    }
    if (!fs::is_directory(opts_.gguf_dir)) {
        logmsg(3, "gguf dir does not exist: %s", opts_.gguf_dir.c_str());
        return 2;
    }
    if (opts_.instances < 1) opts_.instances = 1;
    if (opts_.tmp_dir.empty()) return 2;
    std::error_code ec;
    fs::create_directories(opts_.tmp_dir, ec);
    // 清理上次崩溃残留的临时文件
    for (const auto& e : fs::directory_iterator(opts_.tmp_dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("job-", 0) == 0) fs::remove(e.path(), ec);
    }

    snapshots_.resize(size_t(opts_.instances));
    workers_.reserve(size_t(opts_.instances));
    for (int i = 0; i < opts_.instances; ++i) {
        auto w = std::make_unique<Worker>();
        w->idx = i;
        workers_.push_back(std::move(w));
    }

    install_routes();

    for (auto& w : workers_) {
        Worker* wptr = w.get();
        wptr->th = std::thread([this, wptr] { worker_loop(wptr->idx); });
    }
    reaper_ = std::thread([this] {
        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            if (stop_.load()) break;
            cleanup_jobs();
        }
    });

    logmsg(1, "vlx-seek-server %s listening on http://%s:%d (instances=%d, precision=%s, idle_unload=%.0fs)",
           kVersion, opts_.host.c_str(), opts_.port, opts_.instances, opts_.precision.c_str(),
           opts_.idle_unload_s);

    const bool ok = http_.listen(opts_.host.c_str(), opts_.port);
    if (!ok) logmsg(3, "failed to listen on %s:%d", opts_.host.c_str(), opts_.port);

    request_stop();
    join_workers();
    logmsg(1, "server stopped");
    return ok ? 0 : 1;
}

// ============================================================================
// worker
// ============================================================================
vlx_seek_ctx* VlxServer::load_ctx(std::string* err) {
    vlx_seek_options* o = vlx_seek_options_new();
    vlx_seek_options_set_device(o, opts_.device.c_str());
    vlx_seek_options_set_threads(o, opts_.threads);
    vlx_seek_options_set_precision(o, opts_.precision.c_str());
    vlx_seek_options_set_models_dir(o, opts_.gguf_dir.c_str());
    vlx_seek_options_set_max_new_tokens(o, opts_.max_new_tokens);

    // LLM 精度回退：模型目录通常只有 q8_0 的 LLM（与 deploy/vlx.sh 同语义）
    const std::string lp = opts_.precision == "f32" ? "f16" : opts_.precision;
    const std::string llm = opts_.gguf_dir + "/vlx-10b-llm-" + lp + ".gguf";
    if (!fs::exists(llm)) {
        const std::string q8 = opts_.gguf_dir + "/vlx-10b-llm-q8_0.gguf";
        if (fs::exists(q8)) {
            logmsg(1, "llm %s missing, falling back to %s", llm.c_str(), q8.c_str());
            vlx_seek_options_set_llm_gguf(o, q8.c_str());
        }
    }

    vlx_seek_ctx* ctx = vlx_seek_load_opts(o);
    vlx_seek_options_free(o);
    if (!ctx) {
        const char* vis = std::getenv("CUDA_VISIBLE_DEVICES");
        *err = std::string("vlx_seek_load_opts returned NULL (out of memory / no usable device; "
                          "CUDA_VISIBLE_DEVICES=") +
               (vis && *vis ? vis : "unset") + ")";
        return nullptr;
    }
    if (!vlx_seek_is_ready(ctx)) {
        const char* e = vlx_seek_last_error(ctx);
        *err = e && *e ? e : "context not ready (device admission failed)";
        vlx_seek_free(ctx);
        return nullptr;
    }
    // 预热 backend（编译 kernel），把首请求的 JIT 成本并进加载时间
    vlx_seek_warmup_backend(ctx);
    return ctx;
}

// 用 64x64 灰图跑两条 dummy 请求：一次带框（mmproj+radio+fpn+hfre+LLM），
// 一次不带框（WeDetect 图）。只为了让 lazy 的图/权重在启动时建好，失败只记日志。
void VlxServer::warmup_ctx(vlx_seek_ctx* ctx) {
    const int W = 64, H = 64;
    std::vector<uint8_t> rgb(size_t(W) * H * 3, 128);
    vlx_image_view img{rgb.data(), W, H, size_t(W) * 3, VLX_IMAGE_RGB8};
    const float box[4] = {4.f, 4.f, 60.f, 60.f};
    if (vlx_seek_run_image(ctx, &img, "detection", "object", "en", box, 1, nullptr, 0) != 0) {
        const char* e = vlx_seek_last_error(ctx);
        logmsg(2, "warmup (with boxes) failed: %s", e ? e : "unknown");
    }
    if (vlx_seek_run_image(ctx, &img, "detection", "object", "en", nullptr, 0, nullptr, 0) != 0) {
        const char* e = vlx_seek_last_error(ctx);
        logmsg(2, "warmup (detector) failed: %s", e ? e : "unknown");
    }
}

void VlxServer::publish_state(Worker& w, const char* state) {
    std::lock_guard<std::mutex> lk(mu_);
    if (size_t(w.idx) >= snapshots_.size()) return;
    snapshots_[size_t(w.idx)].state = state;
    snapshots_[size_t(w.idx)].busy = w.busy;
    snapshots_[size_t(w.idx)].load_ms = w.ctx_load_ms;
    snapshots_[size_t(w.idx)].runs = w.runs;
    snapshots_[size_t(w.idx)].last_used_ms = w.last_used_ms.load();
}

void VlxServer::unload_ctx(Worker& w) {
    if (!w.ctx) return;
    vlx_seek_free(w.ctx);
    w.ctx = nullptr;
    w.ctx_load_ms = 0.0;
    unload_count_.fetch_add(1);
    publish_state(w, "unloaded");
}

void VlxServer::maybe_unload(Worker& w) {
    if (!w.ctx) return;
    const bool forced = w.unload_requested.exchange(false);
    const int64_t last = w.last_used_ms.load();
    const double idle = last > 0 ? double(now_ms() - last) / 1000.0 : 0.0;
    if (!forced && (opts_.idle_unload_s <= 0 || idle < opts_.idle_unload_s)) return;
    logmsg(1, "worker %d: unloading model%s (idle %.1fs)", w.idx,
           forced ? " on request" : "", idle);
    unload_ctx(w);
}

void VlxServer::worker_loop(int idx) {
    Worker& w = *workers_[size_t(idx)];
    w.last_used_ms.store(now_ms());

    if (opts_.preload) {
        std::string err;
        const int64_t t0 = now_ms();
        w.ctx = load_ctx(&err);
        if (w.ctx) {
            // 记录 backend 实际解析到的设备（多卡时用于自证用了哪张卡）
            {
                const char* dev = vlx_seek_context_device(w.ctx);
                std::lock_guard<std::mutex> lk(mu_);
                if (size_t(idx) < snapshots_.size()) snapshots_[size_t(idx)].device = dev ? dev : "";
            }
            // 视觉图是懒上传的：预加载时先跑两次 dummy 推理把图建好，
            // 否则第一个真实请求还要付 ~0.5-1 s 的首次上传成本。
            warmup_ctx(w.ctx);
            w.ctx_load_ms = double(now_ms() - t0);
            w.last_used_ms.store(now_ms());
            load_count_.fetch_add(1);
            publish_state(w, "loaded");
            logmsg(1, "worker %d: preloaded + warmed in %.0f ms", idx, w.ctx_load_ms);
        } else {
            logmsg(3, "worker %d: preload failed: %s", idx, err.c_str());
            publish_state(w, "error");
        }
    }

    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(400), [&] {
                return stop_.load() || !queue_.empty() || w.unload_requested.load();
            });
            if (stop_.load()) break;
            if (!queue_.empty()) {
                job = queue_.front();
                queue_.pop_front();
            }
        }
        if (job) {
            if (job->state != JobState::Queued) continue;  // 已取消
            w.busy = true;
            run_job(w, job);
            w.busy = false;
            w.last_used_ms.store(now_ms());
            publish_state(w, w.ctx ? "loaded" : "unloaded");
            cv_.notify_all();
        }
        maybe_unload(w);
    }

    w.busy = false;
    unload_ctx(w);
    publish_state(w, "stopped");
}

void VlxServer::run_job(Worker& w, const std::shared_ptr<Job>& job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        job->state = JobState::Running;
        job->started_ms = now_ms();
        job->worker = w.idx;
        if (size_t(w.idx) < snapshots_.size()) snapshots_[size_t(w.idx)].state = "running";
    }
    logmsg(0, "job %s: start (worker %d)", job->id.c_str(), w.idx);

    // 1) 懒加载：只有本 worker 没 ctx 时才加载
    bool did_load = false;
    if (!w.ctx) {
        const int64_t t0 = now_ms();
        std::string err;
        vlx_seek_ctx* ctx = load_ctx(&err);
        if (!ctx) {
            std::lock_guard<std::mutex> lk(mu_);
            job->state = JobState::Failed;
            job->error = "model load failed: " + err;
            job->finished_ms = now_ms();
            if (size_t(w.idx) < snapshots_.size()) {
                snapshots_[size_t(w.idx)].state = "error";
                snapshots_[size_t(w.idx)].last_error = job->error;
            }
            total_failed_.fetch_add(1);
            return;
        }
        w.ctx = ctx;
        {
            const char* dev = vlx_seek_context_device(ctx);
            std::lock_guard<std::mutex> lk(mu_);
            if (size_t(w.idx) < snapshots_.size()) snapshots_[size_t(w.idx)].device = dev ? dev : "";
        }
        w.ctx_load_ms = double(now_ms() - t0);
        w.last_used_ms.store(now_ms());
        did_load = true;
        load_count_.fetch_add(1);
        publish_state(w, "loaded");
        logmsg(1, "worker %d: model loaded in %.0f ms (cold start for job %s)", w.idx,
               w.ctx_load_ms, job->id.c_str());
    }
    {
        // cold_start 只表示"本请求触发了加载"，客户端可据此区分首请求延迟
        std::lock_guard<std::mutex> lk(mu_);
        job->cold_start = did_load;
        job->load_ms = did_load ? w.ctx_load_ms : 0.0;
    }

    // 2) 图像解码（C API 的 decode_image；失败返回错误）
    TempFile img_cleanup;
    img_cleanup.path = job->image_path;
    uint8_t* rgb = nullptr;
    int32_t iw = 0, ih = 0;
    bool owns_rgb = false;
    if (job->resized) {
        // 提交阶段已经解码并缩放过，worker 直接用缩小后的像素
        iw = job->proc_w;
        ih = job->proc_h;
    } else if (vlx_seek_load_path_rgb(job->image_path.c_str(), &rgb, &iw, &ih) == 0 && rgb) {
        owns_rgb = true;
    } else {
        std::lock_guard<std::mutex> lk(mu_);
        job->state = JobState::Failed;
        job->error = "image decode failed";
        job->finished_ms = now_ms();
        total_failed_.fetch_add(1);
        return;
    }
    if (job->resized) {
        rgb = job->scaled_rgb.data();
        owns_rgb = false;
    }
    vlx_image_view view{rgb, iw, ih, size_t(iw) * 3, VLX_IMAGE_RGB8};

    // 3) 推理
    const float* boxes = job->boxes.empty() ? nullptr : job->boxes.data();
    const int32_t* targets = job->targets.empty() ? nullptr : job->targets.data();
    const int32_t n_boxes = int32_t(job->boxes.size() / 4);
    const int32_t n_targets = int32_t(job->targets.size());
    const int rc = vlx_seek_run_image(w.ctx, &view, job->task.c_str(), job->text.c_str(),
                                      job->lang.c_str(), boxes, n_boxes, targets, n_targets);

    json res_json = json::object();
    json timings = json::object();
    std::string png_b64;
    std::string err;
    if (rc != 0) {
        const char* e = vlx_seek_last_error(w.ctx);
        err = e && *e ? e : "inference failed";
    } else {
        char* js = vlx_seek_result_json(w.ctx);
        if (js) {
            res_json = json::parse(js, nullptr, false);
            vlx_seek_free_buffer(js);
            if (res_json.is_discarded()) {
                res_json = json::object();
                err = "result JSON parse failed";
            }
        } else {
            err = "empty result JSON";
        }
        vlx_pipeline_timings t{};
        if (vlx_seek_last_pipeline_timings(w.ctx, &t) == 0) {
            timings = {{"preprocess_ms", t.preprocess_ms},
                       {"inference_ms", t.inference_ms},
                       {"postprocess_ms", t.postprocess_ms},
                       {"serialization_ms", t.serialization_ms},
                       {"e2e_ms", t.e2e_ms}};
        }
        if (job->render_png && err.empty()) {
            TempFile png_tmp;
            png_tmp.path = opts_.tmp_dir + "/" + job->id + ".png";
            if (vlx_seek_result_render_png(w.ctx, &view, png_tmp.path.c_str()) == 0) {
                std::ifstream in(png_tmp.path, std::ios::binary);
                std::string raw((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
                if (!raw.empty()) {
                    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                             "abcdefghijklmnopqrstuvwxyz0123456789+/";
                    png_b64.reserve((raw.size() + 2) / 3 * 4);
                    size_t i = 0;
                    while (i + 2 < raw.size()) {
                        const uint32_t v = (uint8_t(raw[i]) << 16) | (uint8_t(raw[i + 1]) << 8) |
                                           uint8_t(raw[i + 2]);
                        png_b64 += tbl[(v >> 18) & 63];
                        png_b64 += tbl[(v >> 12) & 63];
                        png_b64 += tbl[(v >> 6) & 63];
                        png_b64 += tbl[v & 63];
                        i += 3;
                    }
                    if (i < raw.size()) {
                        uint32_t v = uint8_t(raw[i]) << 16;
                        if (i + 1 < raw.size()) v |= uint8_t(raw[i + 1]) << 8;
                        png_b64 += tbl[(v >> 18) & 63];
                        png_b64 += tbl[(v >> 12) & 63];
                        png_b64 += (i + 1 < raw.size()) ? tbl[(v >> 6) & 63] : '=';
                        png_b64 += '=';
                    }
                }
            } else {
                err = "png render failed";
            }
        }
    }
    if (owns_rgb) vlx_seek_free_buffer(rgb);

    // 结果框坐标回到原图坐标系（模型看到的是缩小后的图）
    if (job->resized && err.empty() && job->orig_w > 0 && job->proc_w > 0) {
        const double kx = double(job->orig_w) / job->proc_w;
        const double ky = double(job->orig_h) / job->proc_h;
        if (res_json.contains("result_bbox_list")) {
            for (auto& b : res_json["result_bbox_list"]) {
                if (b.contains("xmin")) b["xmin"] = b["xmin"].get<double>() * kx;
                if (b.contains("xmax")) b["xmax"] = b["xmax"].get<double>() * kx;
                if (b.contains("ymin")) b["ymin"] = b["ymin"].get<double>() * ky;
                if (b.contains("ymax")) b["ymax"] = b["ymax"].get<double>() * ky;
            }
        }
    }
    job->scaled_rgb.clear();
    job->scaled_rgb.shrink_to_fit();

    // 4) 写结果
    {
        std::lock_guard<std::mutex> lk(mu_);
        job->finished_ms = now_ms();
        if (!err.empty()) {
            job->state = JobState::Failed;
            job->error = err;
            total_failed_.fetch_add(1);
        } else {
            job->state = JobState::Done;
            job->result = std::move(res_json);
            job->timings = std::move(timings);
            job->png_b64 = std::move(png_b64);
            total_ok_.fetch_add(1);
            w.runs++;
            if (job->result.contains("prompt_tokens"))
                total_prompt_tokens_.fetch_add(job->result["prompt_tokens"].get<uint64_t>());
            if (job->result.contains("completion_tokens"))
                total_completion_tokens_.fetch_add(job->result["completion_tokens"].get<uint64_t>());
            if (job->timings.contains("inference_ms"))
                sum_inference_ms_ += job->timings["inference_ms"].get<double>();
            if (job->timings.contains("e2e_ms"))
                sum_e2e_ms_ += job->timings["e2e_ms"].get<double>();
            finished_runs_.fetch_add(1);
        }
        if (size_t(w.idx) < snapshots_.size()) {
            snapshots_[size_t(w.idx)].state = w.ctx ? "loaded" : "unloaded";
            snapshots_[size_t(w.idx)].runs = w.runs;
        }
    }
    logmsg(0, "job %s: %s on worker %d (%.0f ms)", job->id.c_str(), state_name(job->state), w.idx,
           double(job->finished_ms - job->started_ms));
}

// ============================================================================
// 任务表
// ============================================================================
std::shared_ptr<Job> VlxServer::find_job(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = jobs_.find(id);
    return it == jobs_.end() ? nullptr : it->second;
}

json VlxServer::job_json(const std::shared_ptr<Job>& job, bool with_queue_pos) const {
    json j;
    j["job_id"] = job->id;
    j["state"] = state_name(job->state);
    j["worker"] = job->worker;
    j["created_at_ms"] = job->created_ms;
    j["started_at_ms"] = job->started_ms;
    j["finished_at_ms"] = job->finished_ms;
    j["request"] = {{"task", job->task},           {"text", job->text},
                    {"lang", job->lang},           {"n_boxes", job->boxes.size() / 4},
                    {"target_region_indexes", job->targets},
                    {"render_png", job->render_png}, {"image_bytes", job->image_bytes}};
    j["image"] = {{"original_width", job->orig_w},
                  {"original_height", job->orig_h},
                  {"width", job->proc_w},
                  {"height", job->proc_h},
                  {"resized", job->resized},
                  {"scale", job->proc_w > 0 ? double(job->orig_w) / job->proc_w : 1.0}};
    j["cold_start"] = job->cold_start;
    j["load_ms"] = job->load_ms;
    j["error"] = job->error.empty() ? json(nullptr) : json(job->error);
    j["result"] = job->result.is_null() || job->result.empty() ? json(nullptr) : job->result;
    j["timings_ms"] = job->timings.empty() ? json(nullptr) : job->timings;
    j["png_base64"] = job->png_b64.empty() ? json(nullptr) : json(job->png_b64);
    if (with_queue_pos && job->state == JobState::Queued) {
        size_t pos = 0;
        for (const auto& q : queue_) {
            if (q->id == job->id) break;
            ++pos;
        }
        j["queue_position"] = pos;
    }
    return j;
}

void VlxServer::cleanup_jobs() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t cutoff = now_ms() - int64_t(opts_.job_ttl_s * 1000.0);
    // TTL：只删已完成的
    std::deque<std::string> keep;
    for (const auto& id : job_order_) {
        auto it = jobs_.find(id);
        if (it == jobs_.end()) continue;
        const auto& j = it->second;
        const bool finished = j->state == JobState::Done || j->state == JobState::Failed ||
                              j->state == JobState::Cancelled;
        if (finished && j->finished_ms > 0 && j->finished_ms < cutoff) {
            jobs_.erase(it);
            continue;
        }
        keep.push_back(id);
    }
    job_order_.swap(keep);
    // 数量上限：从最老的已完成任务开始删
    while (jobs_.size() > opts_.max_jobs) {
        bool erased = false;
        for (auto it = job_order_.begin(); it != job_order_.end(); ++it) {
            auto jt = jobs_.find(*it);
            if (jt == jobs_.end()) {
                job_order_.erase(it);
                erased = true;
                break;
            }
            const auto& j = jt->second;
            if (j->state == JobState::Done || j->state == JobState::Failed ||
                j->state == JobState::Cancelled) {
                jobs_.erase(jt);
                job_order_.erase(it);
                erased = true;
                break;
            }
        }
        if (!erased) break;
    }
}

// ============================================================================
// HTTP
// ============================================================================
void VlxServer::install_routes() {
    http_.new_task_queue = [] { return new httplib::ThreadPool(16); };
    http_.set_payload_max_length(size_t(opts_.max_body_mb) * 1024 * 1024);
    http_.set_read_timeout(300, 0);
    http_.set_write_timeout(300, 0);
    http_.set_keep_alive_max_count(100);
    http_.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string msg = "internal error";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        res.status = 500;
        res.set_content(json{{"error", msg}}.dump(), "application/json");
    });

    http_.Get("/healthz", [this](const auto& r, auto& s) { handle_health(r, s); });
    http_.Get("/metrics", [this](const auto& r, auto& s) { handle_metrics(r, s); });
    http_.Get("/v1/vlx/info", [this](const auto& r, auto& s) { handle_info(r, s); });
    http_.Post("/v1/vlx/unload", [this](const auto& r, auto& s) { handle_unload(r, s); });
    http_.Post("/v1/vlx/infer", [this](const auto& r, auto& s) { handle_infer(r, s); });
    http_.Get("/v1/vlx/jobs", [this](const auto& r, auto& s) { handle_job_list(r, s); });
    http_.Get(R"(/v1/vlx/jobs/([A-Za-z0-9_\-]+))",
              [this](const auto& r, auto& s) { handle_job_get(r, s); });
    http_.Delete(R"(/v1/vlx/jobs/([A-Za-z0-9_\-]+))",
                 [this](const auto& r, auto& s) { handle_job_cancel(r, s); });

    http_.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (res.status == 404) {
            res.set_content(json{{"error", "not found"}, {"path", req.path}}.dump(),
                            "application/json");
        }
    });
}

// 下载 http(s) 图像。生产环境请配合网络隔离/白名单（服务端会主动出网）。
bool VlxServer::fetch_image_url(const std::string& url, std::string* out,
                                std::string* err) const {
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        *err = "invalid image URL";
        return false;
    }
    const std::string scheme = url.substr(0, scheme_end);
    if (scheme != "http" && scheme != "https") {
        *err = "only http/https image URLs are supported";
        return false;
    }
    const size_t path_start = url.find('/', scheme_end + 3);
    const std::string base = url.substr(0, path_start);  // scheme://host[:port]
    const std::string path = path_start == std::string::npos ? "/" : url.substr(path_start);

    // 代理只用于非本机地址（与 curl 的 no_proxy 习惯一致）
    std::string host = base.substr(base.find("://") + 3);
    const size_t at = host.rfind('@');
    if (at != std::string::npos) host = host.substr(at + 1);
    const size_t port_sep = host.rfind(':');
    if (host.rfind('[') != 0 && port_sep != std::string::npos) host = host.substr(0, port_sep);
    const bool is_local = host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;

    httplib::Result res;
    try {
        httplib::Client cli(base);
        if (!opts_.image_proxy.empty() && !is_local) {
            // host:port 或 http://host:port
            std::string p = opts_.image_proxy;
            const size_t se = p.find("://");
            if (se != std::string::npos) p = p.substr(se + 3);
            const size_t colon = p.rfind(':');
            if (colon != std::string::npos) {
                cli.set_proxy(p.substr(0, colon), std::atoi(p.c_str() + colon + 1));
            } else {
                cli.set_proxy(p, 8080);
            }
        }
        cli.set_follow_location(true);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(opts_.image_url_timeout_s, 0);
        res = cli.Get(path.c_str());
    } catch (const std::exception& e) {
        *err = std::string("image URL fetch failed: ") + e.what();
        return false;
    }
    if (!res) {
        *err = "image URL fetch failed: " + httplib::to_string(res.error());
        return false;
    }
    if (res->status != 200) {
        *err = "image URL returned HTTP " + std::to_string(res->status);
        return false;
    }
    const size_t max_bytes = size_t(opts_.max_image_mb) * 1024 * 1024;
    if (res->body.size() > max_bytes) {
        *err = "image exceeds " + std::to_string(opts_.max_image_mb) + " MB (--max-image-mb)";
        return false;
    }
    if (res->body.empty()) {
        *err = "image URL returned an empty body";
        return false;
    }
    *out = std::move(res->body);
    logmsg(0, "downloaded image %.1f KB from %s", double(out->size()) / 1024.0, url.c_str());
    return true;
}

void VlxServer::handle_infer(const httplib::Request& req, httplib::Response& res) {
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res.status = 400;
        res.set_content(json{{"error", "request body must be a JSON object"}}.dump(),
                        "application/json");
        return;
    }

    std::string task = body.value("task", "detection");
    std::string text = body.value("text", "");
    std::string lang = body.value("lang", "en");
    std::string image = body.value("image", "");
    const bool render_png = body.value("render_png", false);

    if (!known_task(task)) {
        res.status = 400;
        res.set_content(json{{"error", "unknown task"}, {"task", task},
                             {"allowed", json::array({"vqa", "detection", "grounding_single", "counting", "reasoning_detection", "brief_region_caption", "region_ocr"})}}.dump(), "application/json");
        return;
    }
    if (lang != "en" && lang != "zh") {
        res.status = 400;
        res.set_content(json{{"error", "lang must be \"en\" or \"zh\""}}.dump(),
                        "application/json");
        return;
    }
    if (image.empty()) {
        res.status = 400;
        res.set_content(json{{"error", "image is required (base64 or data: URL)"}}.dump(),
                        "application/json");
        return;
    }
    if (body.contains("max_new_tokens") || body.contains("temperature")) {
        res.status = 400;
        res.set_content(json{{"error",
                              "max_new_tokens/temperature are server-wide: set them at startup "
                              "(--max-new-tokens), the service always decodes greedily"}}.dump(),
                        "application/json");
        return;
    }

    std::vector<float> boxes;
    if (body.contains("bbox_list") && body["bbox_list"].is_array()) {
        for (const auto& b : body["bbox_list"]) {
            if (!b.is_array() || b.size() != 4) {
                res.status = 400;
                res.set_content(json{{"error", "bbox_list entries must be [x1,y1,x2,y2]"}}.dump(),
                                "application/json");
                return;
            }
            for (const auto& v : b) {
                if (!v.is_number()) {
                    res.status = 400;
                    res.set_content(json{{"error", "bbox_list values must be numbers"}}.dump(),
                                    "application/json");
                    return;
                }
                boxes.push_back(v.get<float>());
            }
        }
        if (boxes.size() / 4 > 100) {
            res.status = 400;
            res.set_content(json{{"error", "at most 100 boxes are supported"}}.dump(),
                            "application/json");
            return;
        }
    }
    std::vector<int32_t> targets;
    if (body.contains("target_region_indexes") && body["target_region_indexes"].is_array()) {
        for (const auto& v : body["target_region_indexes"]) {
            if (!v.is_number_integer()) {
                res.status = 400;
                res.set_content(json{{"error", "target_region_indexes must be integers"}}.dump(),
                                "application/json");
                return;
            }
            targets.push_back(v.get<int32_t>());
        }
    }
    if (region_task(task) && targets.empty()) {
        res.status = 400;
        res.set_content(json{{"error", "target_region_indexes is required for this task"}}.dump(),
                        "application/json");
        return;
    }
    if (!region_task(task) && !targets.empty()) {
        res.status = 400;
        res.set_content(json{{"error", "target_region_indexes is only valid for "
                                       "brief_region_caption/region_ocr"}}.dump(),
                        "application/json");
        return;
    }
    if (region_task(task) && boxes.empty()) {
        res.status = 400;
        res.set_content(json{{"error", "region tasks require bbox_list"}}.dump(), "application/json");
        return;
    }

    // 队列容量
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (int(queue_.size()) >= opts_.max_queue) {
            res.status = 429;
            res.set_content(json{{"error", "queue full"},
                                 {"queue_depth", queue_.size()},
                                 {"max_queue", opts_.max_queue}}
                                .dump(),
                            "application/json");
            return;
        }
    }

    // 图像输入：http(s) URL / data URL / 裸 base64，统一落临时文件供 C API 读取
    std::string bytes, err;
    const bool is_url = image.rfind("http://", 0) == 0 || image.rfind("https://", 0) == 0;
    if (is_url) {
        if (!fetch_image_url(image, &bytes, &err)) {
            res.status = 400;
            res.set_content(json{{"error", err}, {"image", image}}.dump(), "application/json");
            return;
        }
    } else if (!decode_image_field(image, &bytes, &err)) {
        res.status = 400;
        res.set_content(json{{"error", err}}.dump(), "application/json");
        return;
    }

    auto job = std::make_shared<Job>();
    job->id = "job-" + std::to_string(next_job_.fetch_add(1));
    job->image_path = opts_.tmp_dir + "/" + job->id + ".img";
    {
        std::ofstream out(job->image_path, std::ios::binary);
        out.write(bytes.data(), std::streamsize(bytes.size()));
        if (!out) {
            res.status = 500;
            res.set_content(json{{"error", "failed to stage image"}}.dump(), "application/json");
            std::error_code ec;
            fs::remove(job->image_path, ec);
            return;
        }
    }
    // 分辨率上限：超过 --max-image-tokens 就等比缩小（不拒绝）。
    // 缩小后的坐标在 worker 里按 orig/proc 比例映射回原图坐标系。
    {
        uint8_t* probe_rgb = nullptr;
        int32_t pw = 0, ph = 0;
        if (vlx_seek_load_path_rgb(job->image_path.c_str(), &probe_rgb, &pw, &ph) == 0 && probe_rgb) {
            job->orig_w = pw;
            job->orig_h = ph;
            job->proc_w = pw;
            job->proc_h = ph;
            const int est = estimate_image_tokens(ph, pw);
            if (est > opts_.max_image_tokens) {
                const double target_px = double(opts_.max_image_tokens) * 1024.0;
                const double k = std::sqrt(target_px / (double(pw) * double(ph)));
                int nw = std::max(32, int(std::floor(double(pw) * k)));
                int nh = std::max(32, int(std::floor(double(ph) * k)));
                // 至少保留 32 的网格步长，直到估算 token 数满足预算
                while (estimate_image_tokens(nh, nw) > opts_.max_image_tokens &&
                       nw > 32 && nh > 32) {
                    nw = std::max(32, nw - 32);
                    nh = std::max(32, nh - 32);
                }
                resize_rgb_bilinear(probe_rgb, pw, ph, nw, nh, &job->scaled_rgb);
                job->proc_w = nw;
                job->proc_h = nh;
                job->resized = true;
                logmsg(1, "job %s: resized %dx%d (%d tok) -> %dx%d (%d tok)", job->id.c_str(),
                       pw, ph, est, nw, nh, estimate_image_tokens(nh, nw));
            }
            vlx_seek_free_buffer(probe_rgb);
        }
    }

    job->task = task;
    job->text = text;
    job->lang = lang;
    job->boxes = std::move(boxes);
    job->targets = std::move(targets);
    job->render_png = render_png;
    job->image_bytes = bytes.size();
    job->created_ms = now_ms();

    json out;
    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_[job->id] = job;
        job_order_.push_back(job->id);
        queue_.push_back(job);
        out = job_json(job, true);
    }
    cv_.notify_all();
    logmsg(0, "job %s: queued (%s, image %.1f KB)", job->id.c_str(), task.c_str(),
           double(job->image_bytes) / 1024.0);

    res.status = 202;
    res.set_content(out.dump(), "application/json");
}

void VlxServer::handle_job_get(const httplib::Request& req, httplib::Response& res) {
    auto job = find_job(req.matches[1].str());
    if (!job) {
        res.status = 404;
        res.set_content(json{{"error", "job not found"}}.dump(), "application/json");
        return;
    }
    json j;
    {
        std::lock_guard<std::mutex> lk(mu_);
        j = job_json(job, true);
    }
    res.set_content(j.dump(), "application/json");
}

void VlxServer::handle_job_list(const httplib::Request& req, httplib::Response& res) {
    size_t limit = 50;
    if (req.has_param("limit")) {
        try {
            limit = std::min<size_t>(500, std::max<size_t>(1, std::stoul(req.get_param_value("limit"))));
        } catch (...) {
            limit = 50;
        }
    }
    std::string filter = req.get_param_value("state");
    json arr = json::array();
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = job_order_.rbegin(); it != job_order_.rend() && arr.size() < limit; ++it) {
            auto jt = jobs_.find(*it);
            if (jt == jobs_.end()) continue;
            if (!filter.empty() && filter != state_name(jt->second->state)) continue;
            arr.push_back(job_json(jt->second, false));
        }
    }
    res.set_content(json{{"jobs", arr}, {"count", arr.size()}}.dump(), "application/json");
}

void VlxServer::handle_job_cancel(const httplib::Request& req, httplib::Response& res) {
    auto job = find_job(req.matches[1].str());
    if (!job) {
        res.status = 404;
        res.set_content(json{{"error", "job not found"}}.dump(), "application/json");
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (job->state == JobState::Queued) {
            job->state = JobState::Cancelled;
            job->finished_ms = now_ms();
            total_cancelled_.fetch_add(1);
            std::error_code ec;
            fs::remove(job->image_path, ec);
            res.set_content(json{{"job_id", job->id}, {"state", "cancelled"}}.dump(),
                            "application/json");
            return;
        }
        if (job->state == JobState::Running) {
            res.status = 409;
            res.set_content(json{{"error", "job is already running and cannot be interrupted"},
                                 {"job_id", job->id}}
                                .dump(),
                            "application/json");
            return;
        }
    }
    res.set_content(json{{"job_id", job->id}, {"state", state_name(job->state)},
                         {"note", "job already finished"}}
                        .dump(),
                    "application/json");
}

void VlxServer::handle_health(const httplib::Request&, httplib::Response& res) {
    json workers = json::array();
    bool any_loaded = false;
    size_t queue_depth = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_depth = queue_.size();
        const double t = now_s();
        for (size_t i = 0; i < snapshots_.size(); ++i) {
            const auto& s = snapshots_[i];
            const bool loaded = s.state == "loaded" || s.state == "running";
            any_loaded = any_loaded || loaded;
            workers.push_back({{"worker", i},
                               {"state", s.state},
                               {"model_loaded", loaded},
                               {"busy", s.busy},
                               {"load_ms", s.load_ms},
                               {"runs", s.runs},
                               {"device", s.device.empty() ? json(nullptr) : json(s.device)},
                               {"last_error", s.last_error.empty() ? json(nullptr) : json(s.last_error)}});
        }
    }
    json j{{"status", "ok"},
           {"model_loaded", any_loaded},
           {"workers_loaded", 0},
           {"queue_depth", queue_depth},
           {"idle_unload_s", opts_.idle_unload_s},
           {"instances", opts_.instances},
           {"workers", workers}};
    size_t loaded_n = 0;
    for (const auto& w : workers)
        if (w["model_loaded"].get<bool>()) ++loaded_n;
    j["workers_loaded"] = loaded_n;
    res.set_content(j.dump(), "application/json");
}

std::string VlxServer::models_info_json() const {
    return json{{"device", opts_.device},
                {"gpu", opts_.gpu.empty() ? json(nullptr) : json(opts_.gpu)},
                {"cuda_visible_devices",
                 std::getenv("CUDA_VISIBLE_DEVICES") ? json(std::getenv("CUDA_VISIBLE_DEVICES"))
                                                      : json(nullptr)},
                {"precision", opts_.precision},
                {"gguf_dir", opts_.gguf_dir},
                {"preload", opts_.preload},
                {"max_new_tokens", opts_.max_new_tokens},
                {"temperature", 0.0},
                {"instances", opts_.instances},
                {"idle_unload_s", opts_.idle_unload_s},
                {"max_image_tokens", opts_.max_image_tokens},
                {"max_image_mb", opts_.max_image_mb},
                {"image_url_timeout_s", opts_.image_url_timeout_s},
                {"image_proxy", opts_.image_proxy},
                {"max_queue", opts_.max_queue},
                {"max_jobs", opts_.max_jobs},
                {"job_ttl_s", opts_.job_ttl_s},
                {"max_body_mb", opts_.max_body_mb}}
        .dump();
}

void VlxServer::handle_info(const httplib::Request&, httplib::Response& res) {
    json workers = json::array();
    double sum_avg_inference_ms = 0.0, sum_avg_e2e_ms = 0.0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        sum_avg_inference_ms = sum_inference_ms_;
        sum_avg_e2e_ms = sum_e2e_ms_;
        for (size_t i = 0; i < snapshots_.size(); ++i) {
            const auto& s = snapshots_[i];
            workers.push_back({{"worker", i}, {"state", s.state}, {"busy", s.busy},
                               {"load_ms", s.load_ms}, {"runs", s.runs},
                               {"device", s.device.empty() ? json(nullptr) : json(s.device)}});
        }
    }
    const double runs = double(finished_runs_.load());
    json j{{"service", "vlx-seek-server"},
           {"version", kVersion},
           {"model", "VLX-Seek-1.5-10B"},
           {"config", json::parse(models_info_json())},
           {"workers", workers},
           {"stats",
            {{"requests_ok", total_ok_.load()},
             {"requests_failed", total_failed_.load()},
             {"requests_cancelled", total_cancelled_.load()},
             {"model_loads", load_count_.load()},
             {"model_unloads", unload_count_.load()},
             {"prompt_tokens", total_prompt_tokens_.load()},
             {"completion_tokens", total_completion_tokens_.load()},
             {"avg_inference_ms", runs > 0 ? sum_avg_inference_ms / runs : 0.0},
             {"avg_e2e_ms", runs > 0 ? sum_avg_e2e_ms / runs : 0.0}}}};
    res.set_content(j.dump(2), "application/json");
}

void VlxServer::handle_unload(const httplib::Request& req, httplib::Response& res) {
    const bool wait = req.has_param("wait") && req.get_param_value("wait") != "0";
    int timeout_ms = 5000;
    if (req.has_param("timeout_ms")) {
        try {
            timeout_ms = std::max(0, std::stoi(req.get_param_value("timeout_ms")));
        } catch (...) {
            timeout_ms = 5000;
        }
    }
    for (auto& w : workers_) w->unload_requested.store(true);
    cv_.notify_all();

    json result = json::array();
    const int64_t deadline = now_ms() + timeout_ms;
    for (auto& w : workers_) {
        bool unloaded = false;
        while (true) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                const auto& s = snapshots_[size_t(w->idx)];
                if (s.state == "unloaded" || s.state == "stopped" || s.state == "error") {
                    unloaded = true;
                    break;
                }
            }
            if (!wait || now_ms() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::lock_guard<std::mutex> lk(mu_);
        const auto& s = snapshots_[size_t(w->idx)];
        result.push_back({{"worker", w->idx},
                          {"unloaded", unloaded},
                          {"state", s.state},
                          {"busy", s.busy}});
    }
    res.set_content(json{{"requested", true}, {"wait", wait}, {"workers", result}}.dump(),
                    "application/json");
}

void VlxServer::handle_metrics(const httplib::Request&, httplib::Response& res) {
    std::string out;
    auto line = [&](const std::string& name, const std::string& value,
                    const std::string& help = "") {
        if (!help.empty()) out += "# HELP " + name + " " + help + "\n";
        out += name + " " + value + "\n";
    };
    std::lock_guard<std::mutex> lk(mu_);
    line("vlx_queue_depth", std::to_string(queue_.size()), "Queued inference jobs");
    line("vlx_jobs_total", std::to_string(jobs_.size()), "Jobs retained in memory");
    line("vlx_requests_total", std::to_string(total_ok_.load() + total_failed_.load()));
    line("vlx_requests_ok_total", std::to_string(total_ok_.load()));
    line("vlx_requests_failed_total", std::to_string(total_failed_.load()));
    line("vlx_requests_cancelled_total", std::to_string(total_cancelled_.load()));
    line("vlx_model_loads_total", std::to_string(load_count_.load()));
    line("vlx_model_unloads_total", std::to_string(unload_count_.load()));
    line("vlx_prompt_tokens_total", std::to_string(total_prompt_tokens_.load()));
    line("vlx_completion_tokens_total", std::to_string(total_completion_tokens_.load()));
    for (size_t i = 0; i < snapshots_.size(); ++i) {
        const auto& s = snapshots_[i];
        const std::string tag = "{worker=\"" + std::to_string(i) + "\"}";
        line("vlx_worker_model_loaded" + tag,
             (s.state == "loaded" || s.state == "running") ? "1" : "0");
        line("vlx_worker_busy" + tag, s.busy ? "1" : "0");
        line("vlx_worker_runs_total" + tag, std::to_string(s.runs));
        line("vlx_worker_load_seconds" + tag, std::to_string(s.load_ms / 1000.0));
    }
    res.set_content(out, "text/plain; version=0.0.4");
}

// ============================================================================
// main
// ============================================================================
namespace {
VlxServer* g_server = nullptr;

void signal_handler(int) {
    if (g_server) g_server->request_stop();
}

void usage() {
    std::fprintf(stderr,
                 "vlx-seek-server %s — VLX-Seek 1.5-10B 常驻推理服务（原生异步接口）\n\n"
                 "用法: vlx-seek-server --gguf-dir DIR [options]\n\n"
                 "  --host H                 监听地址 [0.0.0.0]\n"
                 "  --port N                 监听端口 [8100]\n"
                 "  --instances N            并发实例数（每实例一个 ctx，约 9.5 GiB 显存）[2]\n"
                 "  --idle-unload-s F        空闲多少秒后卸载模型，0=不卸载 [300]\n"
                 "  --preload                启动即加载（默认懒加载）\n"
                 "  --device D               cuda|cpu|vulkan [cuda]\n"
                 "  --gpu N                  只用第 N 张 GPU（nvidia-smi 的物理索引，写入 CUDA_VISIBLE_DEVICES）\n"
                 "  --precision P            q8_0|f16 [q8_0]\n"
                 "  --gguf-dir D             模型目录（必填）\n"
                 "  --tmp-dir D              临时文件目录 [<exe_dir>/tmp]\n"
                 "  --threads N              CPU 线程（0=自动）\n"
                 "  --max-new-tokens N       生成长度上限 [2048]\n"
                 "  --max-queue N            队列上限，满则 429 [64]\n"
                 "  --max-jobs N             内存中保留的任务数上限 [512]\n"
                 "  --job-ttl-s F            已完成任务的保留秒数 [3600]\n"
                 "  --max-body-mb N          请求体上限 MB [32]\n"
                 "  --max-image-mb N         URL 下载图像上限 MB [16]\n"
                 "  --max-image-tokens N     图像 token 预算，超出等比缩小 [1200 ≈ 1280x960]\n"
                 "  --image-url-timeout-s N  URL 下载超时秒 [10]\n"
                 "  --image-proxy HOST:PORT  URL 下载代理（默认读 https_proxy/http_proxy）\n"
                 "  --log-level N            0=DEBUG 1=INFO 2=WARN 3=ERROR [1]\n",
                 kVersion);
}

bool parse_args(int argc, char** argv, ServerOptions* o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage();
            std::exit(0);
        } else if (a == "--host") o->host = next("--host");
        else if (a == "--port") o->port = std::atoi(next("--port").c_str());
        else if (a == "--instances") o->instances = std::atoi(next("--instances").c_str());
        else if (a == "--idle-unload-s") o->idle_unload_s = std::atof(next("--idle-unload-s").c_str());
        else if (a == "--preload") o->preload = true;
        else if (a == "--device") o->device = next("--device");
        else if (a == "--gpu") o->gpu = next("--gpu");
        else if (a == "--precision") o->precision = next("--precision");
        else if (a == "--gguf-dir") o->gguf_dir = next("--gguf-dir");
        else if (a == "--tmp-dir") o->tmp_dir = next("--tmp-dir");
        else if (a == "--threads") o->threads = std::atoi(next("--threads").c_str());
        else if (a == "--max-new-tokens") o->max_new_tokens = std::atoi(next("--max-new-tokens").c_str());
        else if (a == "--max-queue") o->max_queue = std::atoi(next("--max-queue").c_str());
        else if (a == "--max-jobs") o->max_jobs = size_t(std::atoll(next("--max-jobs").c_str()));
        else if (a == "--job-ttl-s") o->job_ttl_s = std::atof(next("--job-ttl-s").c_str());
        else if (a == "--max-body-mb") o->max_body_mb = std::atoi(next("--max-body-mb").c_str());
        else if (a == "--max-image-mb") o->max_image_mb = std::atoi(next("--max-image-mb").c_str());
        else if (a == "--max-image-tokens") o->max_image_tokens = std::atoi(next("--max-image-tokens").c_str());
        else if (a == "--image-url-timeout-s") o->image_url_timeout_s = std::atoi(next("--image-url-timeout-s").c_str());
        else if (a == "--image-proxy") o->image_proxy = next("--image-proxy");
        else if (a == "--log-level") o->log_level = std::atoi(next("--log-level").c_str());
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return false;
        }
    }
    if (o->gguf_dir.empty()) {
        std::fprintf(stderr, "--gguf-dir is required\n");
        return false;
    }
    g_log_level = o->log_level;
    return true;
}

std::string exe_dir(const char* argv0) {
    std::error_code ec;
    const fs::path p = fs::weakly_canonical(fs::path(argv0), ec);
    return p.has_parent_path() ? p.parent_path().string() : std::string(".");
}

}  // namespace

int main(int argc, char** argv) {
    ServerOptions opts;
    if (!parse_args(argc, argv, &opts)) return 2;
    if (opts.tmp_dir.empty()) opts.tmp_dir = exe_dir(argv[0]) + "/tmp";
    // 未显式指定 --image-proxy 时，沿用系统代理环境变量（生产环境常需要）
    if (opts.image_proxy.empty()) {
        const char* env = std::getenv("https_proxy");
        if (!env || !*env) env = std::getenv("HTTPS_PROXY");
        if (!env || !*env) env = std::getenv("http_proxy");
        if (!env || !*env) env = std::getenv("HTTP_PROXY");
        if (env && *env) opts.image_proxy = env;
    }
    if (!opts.image_proxy.empty()) logmsg(1, "image URL proxy: %s", opts.image_proxy.c_str());

    // GPU 选择：必须在任何 CUDA/ggml 初始化之前设置（进程级生效，视觉链与
    // llama.cpp 都只看到这一张卡）。外部已设置 CUDA_VISIBLE_DEVICES 时以命令行优先。
    if (!opts.gpu.empty()) {
        setenv("CUDA_VISIBLE_DEVICES", opts.gpu.c_str(), 1);
        logmsg(1, "GPU selection: CUDA_VISIBLE_DEVICES=%s (--gpu)", opts.gpu.c_str());
    } else if (const char* vis = std::getenv("CUDA_VISIBLE_DEVICES"); vis && *vis) {
        logmsg(1, "GPU selection from environment: CUDA_VISIBLE_DEVICES=%s", vis);
    } else {
        logmsg(1, "GPU selection: all visible devices (CUDA_VISIBLE_DEVICES unset)");
    }

    VlxServer server(opts);
    g_server = &server;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);
    return server.run();
}
