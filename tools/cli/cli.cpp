#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unordered_set>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

// Phase 15C: file SHA256 via popen("sha256sum") — zero linking dependency
static std::string file_sha256_hex(const char * filepath) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "sha256sum %s 2>/dev/null", filepath);
    FILE * fp = popen(cmd, "r");
    if (!fp) return "";
    char buf[128] = {0};
    char sha_hex[128] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        // First token is the 64-char hex hash
        sscanf(buf, "%64s", sha_hex);
    }
    int status = pclose(fp);
    if (status != 0 || strlen(sha_hex) != 64) return "";
    return std::string(sha_hex);
}

// PRT (Perturbation) API - only available when libllama has PRT support
extern "C" void llama_set_prt_debug_mode(int mode);
extern "C" void llama_set_prt_sidecar(int layer, const float * data, int M, int N);
extern "C" void llama_set_prt_sidecar_int8(int layer, const int8_t * int8_data, const float * scales, int M, int N);
extern "C" void llama_set_prt_sidecar_int6(int layer, const int8_t * int8_data, const float * scales, int M, int N);
extern "C" void llama_set_prt_sidecar_int6_predecode_f32(int layer, const int8_t * int8_data, const float * scales, int M, int N);
extern int g_prt_kernel_mode;  // 0=scalar, 1=AVX2
extern int g_prt_predecode_f32_enabled;
extern std::chrono::high_resolution_clock::time_point g_prt_predecode_start;
extern "C" void llama_set_prt_force_native_layers(int n_layers, const int * layer_ids);
extern "C" void llama_set_prt_log_file(const char * path);
extern "C" void llama_set_prt_log_level(int level);
extern "C" void llama_dump_prt_timing_summary(void);
extern "C" void llama_dump_prt_build_info(void);
extern "C" void llama_reset_prt_timing(void);
extern "C" void llama_pretouch_prt_sidecars(void);
extern FILE * g_prt_log_file;

#include "server-context.h"
#include "server-task.h"

#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <signal.h>
#include <vector>
#include <sys/stat.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;
    int reasoning_budget = -1;
    std::string reasoning_budget_message;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress

        verbose_prompt = params.verbose_prompt;
        reasoning_budget = params.reasoning_budget;
        reasoning_budget_message = params.reasoning_budget_message;
    }

    std::string generate_completion(result_timings & out_timings) {
        server_response_reader rd = ctx_server.get_response_reader();
        auto chat_params = format_chat();
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            task.cli_prompt = chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            // chat template settings
            task.params.chat_parser_params = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            // reasoning budget sampler
            if (!chat_params.thinking_end_tag.empty()) {
                const llama_vocab * vocab = llama_model_get_vocab(
                    llama_get_model(ctx_server.get_llama_context()));

                task.params.sampling.reasoning_budget_tokens = reasoning_budget;
                task.params.sampling.generation_prompt = chat_params.generation_prompt;

                if (!chat_params.thinking_start_tag.empty()) {
                    task.params.sampling.reasoning_budget_start =
                        common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
                }
                task.params.sampling.reasoning_budget_end =
                    common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
                task.params.sampling.reasoning_budget_forced =
                    common_tokenize(vocab, reasoning_budget_message + chat_params.thinking_end_tag, false, true);
            }

            rd.post_task({std::move(task)});
        }

        if (verbose_prompt) {
            console::set_display(DISPLAY_TYPE_PROMPT);
            console::log("%s\n\n", chat_params.prompt.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }

    common_chat_params format_chat() {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = {}; // TODO
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.force_pure_content    = chat_params.force_pure_content;
        inputs.enable_thinking       = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }
};

// TODO?: Make this reusable, enums, docs
static const std::array<const std::string, 7> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line[0] == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](const std::string & cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, (*it).length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, "~")) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = std::string(home) + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, "/")) {
#else
        if (std::isalpha(expanded_prefix[0]) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str = "";
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);
    console::set_completion_callback(auto_completion_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    static std::vector<float *> g_prt_sidecar_buffers;
    static std::vector<int8_t *> g_prt_int8_sidecar_buffers;
    static std::vector<float *> g_prt_int8_scale_buffers;
    static std::vector<int8_t *> g_prt_int6_sidecar_buffers;  // packed INT6 data (needs unpacking)
    static std::vector<float *> g_prt_int6_scale_buffers;
    bool use_int8 = (params.prt_sidecar_format == "int8");
    bool use_int6 = (params.prt_sidecar_format == "int6");
    if (params.prt_mode > 0) {
        // Phase 13W: reset timing accumulators at start of each run
        llama_reset_prt_timing();

        // Set PRT log file FIRST so subsequent PRT logs route correctly
        if (!params.prt_log_file.empty()) {
            llama_set_prt_log_file(params.prt_log_file.c_str());
        }
        // Set PRT log level (default=2=debug)
        llama_set_prt_log_level(params.prt_log_level);
        llama_set_prt_debug_mode(params.prt_mode);
        // Phase 19J: set predecode mode if requested
        if (params.prt_predecode_f32) {
            g_prt_predecode_f32_enabled = 1;
            g_prt_predecode_start = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "[PRT-PREDECODE] enabled via --prt-predecode-f32\n");
        }
        // Note: llama_set_prt_debug_mode logs "Debug mode set to N" internally via g_prt_log_file

        // Load sidecars
        // Sidecar files are pure float arrays without headers.
        // Detect M/N from file size: M*N*4 = bytes.
        // Known shapes: Qwen2.5-0.5B=896*4864, Qwen2.5-1.5B=1536*8960, Qwen2.5-3B=2048*11008
        // Phase 14B/15B-G: log format selection
        if (g_prt_log_file) {
            const char* fmt_str = use_int8 ? "int8" : (use_int6 ? "int6" : "float32");
            const char* scale_str = use_int8 ? "per_row" : (use_int6 ? "per_row" : "none");
            fprintf(g_prt_log_file, "[PRT_FORMAT] sidecar_format=%s scale_scheme=%s\n",
                    fmt_str, scale_str);
            fflush(g_prt_log_file);
        }
        auto sidecar_load_start = std::chrono::high_resolution_clock::now();
        std::string sidecar_dir = params.prt_sidecar_dir.empty() ? "/tmp/prt_sidecars/" : params.prt_sidecar_dir;
        const llama_model * model = llama_get_model(ctx_cli.ctx_server.get_llama_context());
        int n_layer = llama_model_n_layer(model);
        int loaded = 0;
        size_t total_sidecar_bytes = 0;
        std::string provenance_fmt_str = use_int8 ? "int8" : (use_int6 ? "int6" : "float32");
        std::unordered_set<std::string> unique_sha_set;
        std::vector<int> force_native_layers;
        if (!params.prt_force_native.empty()) {
            std::string s = params.prt_force_native;
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? s.size() - pos : comma - pos);
                try { force_native_layers.push_back(std::stoi(tok)); } catch (...) {}
                pos = (comma == std::string::npos) ? s.size() : comma + 1;
            }
        }
        // Phase 15E: detailed timing
        auto t_provenance_begin = std::chrono::high_resolution_clock::now();
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "\n[PRT_PROVENANCE_BEGIN]\n");
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] sidecar_dir=%s\n", sidecar_dir.c_str());
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] sidecar_format=%s\n", provenance_fmt_str.c_str());
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] expected_layers=%d\n", n_layer);
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] force_native_count=%zu\n", force_native_layers.size());
            for (size_t fi = 0; fi < force_native_layers.size(); fi++) {
                fprintf(g_prt_log_file, "[PRT_PROVENANCE] force_native_layer=%d\n", force_native_layers[fi]);
            }
            fflush(g_prt_log_file);
        }
        double ms_provenance_begin = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_provenance_begin).count();
        double ms_sidecar_hash = 0.0;
        double ms_sidecar_unpack = 0.0;
        bool hash_mode_manifest = false;
        bool manifest_valid = false;

        // Phase 15F: manifest loading
        std::string manifest_path = sidecar_dir + "/prt_sidecar_manifest.json";
        struct ManifestLayer { int layer; std::string filename; int64_t size; std::string sha256; bool fallback; };
        std::vector<ManifestLayer> manifest_layers;
        std::string manifest_format;
        std::string manifest_gen_version;

        {
            FILE * mf = fopen(manifest_path.c_str(), "r");
            if (mf) {
                fseek(mf, 0, SEEK_END);
                long mfsz = ftell(mf);
                fseek(mf, 0, SEEK_SET);
                std::string mbuf((size_t)mfsz, '\0');
                if (fread(&mbuf[0], 1, (size_t)mfsz, mf) == (size_t)mfsz) {
                    // Minimal JSON parse: look for expected_layers, sidecar_format, and layer entries
                    // Validate by checking file sizes match for all entries
                    bool manifest_ok = true;
                    // Extract expected_layers
                    int exp_layers = n_layer;
                    {
                        const char * el = strstr(mbuf.c_str(), "\"expected_layers\"");
                        if (el) { sscanf(el, "%*[^:]: %d", &exp_layers); }
                    }
                    {
                        char fmtbuf[32] = {0};
                        const char * sf = strstr(mbuf.c_str(), "\"sidecar_format\"");
                        if (sf) {
                            const char * colon = strchr(sf, ':');
                            if (colon) {
                                const char * vp = colon + 1;
                                while (*vp && (*vp == ' ' || *vp == '\t' || *vp == '\n' || *vp == '\r')) vp++;
                                if (*vp == '"') { vp++; }
                                const char * ve = vp;
                                while (*ve && *ve != '"' && *ve != ',' && *ve != '\n' && *ve != '\r' && *ve != '}') ve++;
                                if (vp < ve && (size_t)(ve - vp) < sizeof(fmtbuf)) {
                                    memcpy(fmtbuf, vp, (size_t)(ve - vp)); fmtbuf[(size_t)(ve - vp)] = '\0';
                                    manifest_format = fmtbuf;
                                }
                            }
                        }
                    }
                    // Validate: format must match, layers must match
                    if (exp_layers != n_layer) manifest_ok = false;
                    if (!manifest_format.empty() && manifest_format != provenance_fmt_str) manifest_ok = false;
                    if (manifest_ok) {
                        // Parse layer entries by scanning for "layer": N patterns
                        const char * p = mbuf.c_str();
                        while ((p = strstr(p, "\"layer\":")) != nullptr) {
                            int lay = -1;
                            char fnbuf[128] = {0}; long long fsz = 0; char shabuf[128] = {0}; bool is_fb = false;
                            if (sscanf(p, "%*[^:]: %d", &lay) == 1) {
                                const char * fp = strstr(p, "\"filename\"");
                                const char * sp = strstr(p, "\"sha256\"");
                                const char * szp = strstr(p, "\"size\"");
                                const char * fbp = strstr(p, "\"fallback\"");
                                if (fp) { const char * q = fp; while (*q && *q != '"') q++; if (*q) q++; while (*q && *q != '"') { int nl = strlen(fnbuf); if (nl < 127) fnbuf[nl] = *q++; } }
                                if (sp) { const char * q = sp; while (*q && *q != ':') q++; q++; while (*q && (*q == ' ' || *q == '\t')) q++; if (*q == '"') { q++; } const char * qe = q; while (*qe && *qe != '"') qe++; if (qe > q && (size_t)(qe - q) < 128) { memcpy(shabuf, q, (size_t)(qe - q)); shabuf[(size_t)(qe - q)] = '\0'; } }
                                if (szp) { sscanf(szp, "%*[^:]: %lld", &fsz); }
                                if (fbp) { char fbbuf[8] = {0}; if (sscanf(fbp, "%*[^:]: %7s", fbbuf) == 1) is_fb = (strcmp(fbbuf, "true") == 0); }
                                if (lay >= 0 && strlen(shabuf) == 64) {
                                    ManifestLayer ml; ml.layer = lay; snprintf(fnbuf, sizeof(fnbuf), "ffn_up_layer%d_prt.%s", lay, provenance_fmt_str.c_str()); ml.filename = fnbuf; ml.size = fsz; ml.sha256 = shabuf; ml.fallback = is_fb; manifest_layers.push_back(ml);
                                }
                            }
                            p++;
                        }
                        // Verify all expected files exist and sizes match
                        int chk_ok = 0, chk_fail = 0;
                        for (auto & ml : manifest_layers) {
                            std::string fp = sidecar_dir + "/" + ml.filename;
                            struct stat fst; if (stat(fp.c_str(), &fst) == 0 && fst.st_size == ml.size) chk_ok++; else chk_fail++;
                        }
                        if (chk_fail == 0 && (int)manifest_layers.size() == n_layer) {
                            manifest_valid = true;
                            hash_mode_manifest = true;
                        }
                    }
                }
                fclose(mf);
            }
        }

        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] manifest_status=%s\n", manifest_valid ? "valid" : "missing_or_stale");
            fprintf(g_prt_log_file, "[PRT_PROVENANCE] hash_mode=%s\n", hash_mode_manifest ? "manifest" : "full");
            if (!manifest_format.empty()) fprintf(g_prt_log_file, "[PRT_PROVENANCE] manifest_format=%s\n", manifest_format.c_str());
            fflush(g_prt_log_file);
        }

        auto t_sidecar_load_start = std::chrono::high_resolution_clock::now();
        for (int l = 0; l < n_layer; l++) {
            // Phase 14B: branch based on format
            auto t_layer_start = std::chrono::high_resolution_clock::now();
            auto t_file_open = t_layer_start;
            if (use_int8) {
                // INT8 sidecar: .int8 file with per-row float32 scales appended
                // File layout: [M*K bytes int8 data][M*4 bytes float32 scales]
                std::string path = sidecar_dir + "/ffn_up_layer" + std::to_string(l) + "_prt.int8";
                struct stat st;
                if (stat(path.c_str(), &st) != 0) continue;
                int64_t raw_bytes = st.st_size;
                // Infer M from file size: M*K int8 + M*4 float32
                // For 3B: M=11008, K=2048 → 11008*2048=22,540,544 + 11008*4=44,032 = 22,584,576
                // For 0.5B: M=4864, K=896 → 4864*896=4,358,144 + 4864*4=19,456 = 4,377,600
                // For 7B: M=18944, K=3584 → 18944*3584=67,895,296 + 18944*4=75,776 = 67,971,072
                int M = 0, K = 0;
                int64_t int8_bytes_3b = (int64_t)11008 * 2048;
                int64_t scale_bytes_3b = (int64_t)11008 * 4;
                int64_t int8_bytes_05b = (int64_t)4864 * 896;
                int64_t scale_bytes_05b = (int64_t)4864 * 4;
                int64_t int8_bytes_7b = (int64_t)18944 * 3584;
                int64_t scale_bytes_7b = (int64_t)18944 * 4;
                int64_t total_3b = int8_bytes_3b + scale_bytes_3b;
                int64_t total_05b = int8_bytes_05b + scale_bytes_05b;
                int64_t total_7b = int8_bytes_7b + scale_bytes_7b;
                if (raw_bytes == total_3b) { M = 11008; K = 2048; }
                else if (raw_bytes == total_05b) { M = 4864; K = 896; }
                else if (raw_bytes == total_7b) { M = 18944; K = 3584; }
                else {
                    fprintf(stderr, "[PRT] Unknown INT8 sidecar size %ld for layer %d, skipping\n", (long)raw_bytes, l);
                    continue;
                }
                FILE * f = fopen(path.c_str(), "rb");
                if (!f) continue;
                int64_t int8_n = (int64_t)M * K;
                int8_t * int8_data = (int8_t *)malloc((size_t)int8_n);
                if (!int8_data) { fclose(f); continue; }
                float * scales = (float *)malloc((size_t)M * sizeof(float));
                if (!scales) { free(int8_data); fclose(f); continue; }
                if (fread(int8_data, 1, (size_t)int8_n, f) != (size_t)int8_n) {
                    free(int8_data); free(scales); fclose(f); continue;
                }
                if (fread(scales, sizeof(float), (size_t)M, f) != (size_t)M) {
                    free(int8_data); free(scales); fclose(f); continue;
                }
                fclose(f);
                llama_set_prt_sidecar_int8(l, int8_data, scales, M, K);
                g_prt_int8_sidecar_buffers.push_back(int8_data);
                g_prt_int8_scale_buffers.push_back(scales);
                loaded++;
                total_sidecar_bytes += (size_t)raw_bytes;

                // Phase 15E: INT8 read timing
                auto t_int8_read = std::chrono::high_resolution_clock::now();
                double ms_int8_read = std::chrono::duration<double, std::milli>(t_int8_read - t_file_open).count();
                // Phase 15F: SHA — use manifest if valid, otherwise compute
                double ms_sha = 0.0;
                std::string sha_hex;
                if (hash_mode_manifest) {
                    // Fast path: look up from manifest
                    for (auto & ml : manifest_layers) {
                        if (ml.layer == l) { sha_hex = ml.sha256; break; }
                    }
                } else {
                    // Full hash fallback
                    auto t_sha_start = std::chrono::high_resolution_clock::now();
                    sha_hex = file_sha256_hex(path.c_str());
                    ms_sha = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_sha_start).count();
                    ms_sidecar_hash += ms_sha;
                }
                if (!sha_hex.empty()) unique_sha_set.insert(sha_hex);
                bool is_fallback = false;
                for (int fn : force_native_layers) { if (fn == l) { is_fallback = true; break; } }
                if (g_prt_log_file) {
                    double ms_layer_total = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_layer_start).count();
                    fprintf(g_prt_log_file, "[PRT_SIDECAR_LAYER] layer=%d file=ffn_up_layer%d_prt.int8 size=%ld sha256=%s status=loaded fallback=%s read_ms=%.2f hash_ms=%.2f layer_total_ms=%.2f\n",
                            l, l, (long)raw_bytes, sha_hex.c_str(), is_fallback ? "true" : "false",
                            ms_int8_read, ms_sha, ms_layer_total);
                    fflush(g_prt_log_file);
                }
            } else if (use_int6) {
                // INT6 packed sidecar: .int6 file with PRT6 header + packed payload + scales
                // File layout: [4 magic][4 version][4 rows][4 cols][4 reserved][M*4 scales][packed bytes]
                // Packing: 4 INT6 values -> 3 bytes (offset-32 encoding, 6 bits per value)
                // For 7B: M=18944, K=3584, packed_payload=ceil(18944*3584/4)*3=50997132, scales=18944*4=75776
                // Total: 16 + 75776 + 50997132 = 51072924
                std::string path = sidecar_dir + "/ffn_up_layer" + std::to_string(l) + "_prt.int6";
                struct stat st;
                if (stat(path.c_str(), &st) != 0) continue;
                int64_t raw_bytes = st.st_size;

                // Phase 15G: mmap-based loading for INT6
                bool use_mmap = true;
                const uint8_t * mmap_base = nullptr;
                size_t mmap_len = 0;
                int fd = -1;
                if (use_mmap) {
                    fd = open(path.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        mmap_base = (const uint8_t *)mmap(nullptr, (size_t)raw_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
                        if (mmap_base == MAP_FAILED || mmap_base == nullptr) {
                            close(fd); fd = -1; mmap_base = nullptr;
                        } else {
                            mmap_len = (size_t)raw_bytes;
                            madvise((void*)mmap_base, mmap_len, MADV_SEQUENTIAL);
                        }
                    }
                }

                if (mmap_base) {
                    // mmap path: read directly from memory
                    if (mmap_len < 16) { munmap((void*)mmap_base, mmap_len); close(fd); continue; }
                    uint8_t magic[4] = { mmap_base[0], mmap_base[1], mmap_base[2], mmap_base[3] };
                    if (magic[0] != 'P' || magic[1] != 'R' || magic[2] != 'T' || magic[3] != '6') {
                        munmap((void*)mmap_base, mmap_len); close(fd); continue;
                    }
                    uint32_t version = *(uint32_t*)(mmap_base + 4);
                    uint32_t rows = *(uint32_t*)(mmap_base + 8);
                    uint32_t cols = *(uint32_t*)(mmap_base + 12);

                    int64_t total_7b = 50997268;
                    int64_t total_14b = 53139472;
                    int64_t total_05b = 3288080;  // Phase 19B: Qwen2.5-0.5B FFN_UP INT6
                    int64_t total_3b = 0;
                    int M = 0, K = 0;
                    if (raw_bytes == total_7b) { M = 18944; K = 3584; }
                    else if (raw_bytes == total_14b) { M = 13824; K = 5120; }
                    else if (raw_bytes == total_05b) { M = 4864; K = 896; }
                    else if (raw_bytes == total_3b && total_3b > 0) { M = 11008; K = 2048; }
                    else {
                        munmap((void*)mmap_base, mmap_len); close(fd); continue;
                    }

                    size_t scale_off = 16;
                    size_t packed_off = scale_off + (size_t)M * 4;
                    size_t packed_n = (size_t)((((int64_t)M * K + 3) / 4) * 3);
                    if (packed_off + packed_n > mmap_len) {
                        munmap((void*)mmap_base, mmap_len); close(fd); continue;
                    }

                    float * scales = (float *)malloc((size_t)M * sizeof(float));
                    if (!scales) { munmap((void*)mmap_base, mmap_len); close(fd); continue; }
                    memcpy(scales, mmap_base + scale_off, (size_t)M * sizeof(float));

                    const uint8_t * packed_src = mmap_base + packed_off;
                    int8_t * int8_data = (int8_t *)malloc((size_t)M * K);
                    if (!int8_data) { free(scales); munmap((void*)mmap_base, mmap_len); close(fd); continue; }

                    // Phase 15H: optimized INT6 unpack
                    // Two optimizations over scalar:
                    //   1. 64-entry LUT: replaces 6-bit arithmetic with array lookup
                    //   2. 4x loop unrolling: reduces loop overhead and branch mispredictions
                    // M*K is divisible by 4, so boundary checks are only for the final partial group.
                    auto t_unpack_start = std::chrono::high_resolution_clock::now();
                    {
                        // 64-entry decode LUT: maps 6-bit pattern to offset-32 int8
                        int8_t lut[64];
                        for (int ui = 0; ui < 64; ui++) lut[ui] = (int8_t)(ui - 32);

                        const uint8_t * p_src = packed_src;
                        int64_t total = (int64_t)M * K;
                        int64_t i = 0;

                        // 4x unrolled body: 16 elements per iteration (12 source bytes)
                        for (; i + 15 < total; i += 16) {
                            // Group 0
                            uint8_t b0 = *p_src++; uint8_t b1 = *p_src++; uint8_t b2 = *p_src++;
                            int8_data[i]     = lut[b0 & 0x3F];
                            int8_data[i + 1] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
                            int8_data[i + 2] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
                            int8_data[i + 3] = lut[(b2 >> 2) & 0x3F];
                            // Group 1
                            b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
                            int8_data[i + 4] = lut[b0 & 0x3F];
                            int8_data[i + 5] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
                            int8_data[i + 6] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
                            int8_data[i + 7] = lut[(b2 >> 2) & 0x3F];
                            // Group 2
                            b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
                            int8_data[i + 8]  = lut[b0 & 0x3F];
                            int8_data[i + 9]  = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
                            int8_data[i + 10] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
                            int8_data[i + 11] = lut[(b2 >> 2) & 0x3F];
                            // Group 3
                            b0 = *p_src++; b1 = *p_src++; b2 = *p_src++;
                            int8_data[i + 12] = lut[b0 & 0x3F];
                            int8_data[i + 13] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
                            int8_data[i + 14] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
                            int8_data[i + 15] = lut[(b2 >> 2) & 0x3F];
                        }

                        // Scalar tail: 1-3 groups remaining (total % 16 != 0 only if M*K not divisible by 16)
                        // For 7B: M*K = 18944*3584 = 67,854,848 = divisible by 16, so no tail needed.
                        // But keep the path for completeness.
                        for (; i < total; i += 4) {
                            uint8_t b0 = *p_src++; uint8_t b1 = *p_src++; uint8_t b2 = *p_src++;
                            int8_data[i]     = lut[b0 & 0x3F];
                            int8_data[i + 1] = lut[((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F];
                            int8_data[i + 2] = lut[((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F];
                            int8_data[i + 3] = lut[(b2 >> 2) & 0x3F];
                        }
                    }
                    double ms_unpack = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_unpack_start).count();
                    ms_sidecar_unpack += ms_unpack;

                    munmap((void*)mmap_base, mmap_len); close(fd); mmap_base = nullptr; fd = -1;

                    if (g_prt_predecode_f32_enabled) {
                    llama_set_prt_sidecar_int6_predecode_f32(l, int8_data, scales, M, K);
                } else {
                    llama_set_prt_sidecar_int6(l, int8_data, scales, M, K);
                }
                    // Phase 19D-C: loader audit (layer 0 only to avoid spam)
                    if (l == 0) {
                        fprintf(stderr, "[PRT_LOAD_AUDIT] layer=%d M=%d K=%d int8_data=%p scales=%p int8_0=%d scale_0=%.6f\n",
                                l, M, K, (void*)int8_data, (void*)scales,
                                (int)int8_data[0], scales[0]);
                        // Dump first 16 packed bytes
                        fprintf(stderr, "[PRT_LOAD_AUDIT] packed_bytes: ");
                        for (int di = 0; di < 16; di++) fprintf(stderr, " %02x", (unsigned char)int8_data[di]);
                        fprintf(stderr, "\n");
                    }
                    g_prt_int6_sidecar_buffers.push_back(int8_data);
                    g_prt_int6_scale_buffers.push_back(scales);
                    loaded++;
                    total_sidecar_bytes += (size_t)raw_bytes;

                    auto t_int6_done = std::chrono::high_resolution_clock::now();
                    double ms_mmap = std::chrono::duration<double, std::milli>(t_int6_done - t_file_open).count();
                    // Phase 15F: SHA — use manifest if valid, otherwise compute
                    double ms_sha = 0.0;
                    std::string sha_hex;
                    if (hash_mode_manifest) {
                        for (auto & ml : manifest_layers) { if (ml.layer == l) { sha_hex = ml.sha256; break; } }
                    } else {
                        auto t_sha_start = std::chrono::high_resolution_clock::now();
                        sha_hex = file_sha256_hex(path.c_str());
                        ms_sha = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t_sha_start).count();
                        ms_sidecar_hash += ms_sha;
                    }
                    if (!sha_hex.empty()) unique_sha_set.insert(sha_hex);
                    bool is_fallback = false;
                    for (int fn : force_native_layers) { if (fn == l) { is_fallback = true; break; } }
                    if (g_prt_log_file) {
                        double ms_layer_total = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t_layer_start).count();
                        fprintf(g_prt_log_file, "[PRT_SIDECAR_LAYER] layer=%d file=ffn_up_layer%d_prt.int6 size=%ld sha256=%s status=loaded fallback=%s mmap_ms=%.2f unpack_ms=%.2f hash_ms=%.2f layer_total_ms=%.2f load_mode=mmap unpack_kernel=lut4x\n",
                                l, l, (long)raw_bytes, sha_hex.c_str(), is_fallback ? "true" : "false",
                                ms_mmap, ms_unpack, ms_sha, ms_layer_total);
                        fflush(g_prt_log_file);
                    }
                    continue; // skip the fread block
                }

                // Fallback: fread path
                // Known sizes for 7B and 3B
                // NOTE: generator uses ((n_elements+3)/4)*3 which gives 50921472 for 7B
                // but the actual files are 50997268 due to an off-by-4 write in generator.
                // Use actual file sizes to match generator output.
                int64_t scale_bytes_7b = (int64_t)18944 * 4;
                int64_t header_bytes = 16;
                // Actual file size from generator (includes off-by-4 padding):
                int64_t total_7b = 50997268;
                int64_t total_14b = 53139472;
                int64_t total_05b = 3288080;  // Phase 19B: Qwen2.5-0.5B FFN_UP INT6
                int64_t total_3b = 0;  // 3B not yet tested

                int M = 0, K = 0;
                if (raw_bytes == total_7b) { M = 18944; K = 3584; }
                else if (raw_bytes == total_14b) { M = 13824; K = 5120; }
                else if (raw_bytes == total_05b) { M = 4864; K = 896; }
                else if (raw_bytes == total_3b && total_3b > 0) { M = 11008; K = 2048; }
                else {
                    fprintf(stderr, "[PRT] Unknown INT6 sidecar size %ld for layer %d, skipping\n", (long)raw_bytes, l);
                    continue;
                }

                FILE * f = fopen(path.c_str(), "rb");
                if (!f) continue;

                // Read and verify header
                uint8_t magic[4];
                uint32_t version, rows, cols, reserved;
                if (fread(magic, 1, 4, f) != 4 || magic[0] != 'P' || magic[1] != 'R' || magic[2] != 'T' || magic[3] != '6') {
                    fprintf(stderr, "[PRT] Invalid INT6 magic for layer %d\n", l);
                    fclose(f); continue;
                }
                if (fread(&version, 4, 1, f) != 1 || fread(&rows, 4, 1, f) != 1 ||
                    fread(&cols, 4, 1, f) != 1 || fread(&reserved, 4, 1, f) != 1) {
                    fprintf(stderr, "[PRT] Failed to read INT6 header for layer %d\n", l);
                    fclose(f); continue;
                }

                // Read scales
                float * scales = (float *)malloc((size_t)M * sizeof(float));
                if (!scales) { fclose(f); continue; }
                if (fread(scales, sizeof(float), (size_t)M, f) != (size_t)M) {
                    free(scales); fclose(f); continue;
                }

                // Read packed payload
                size_t packed_n = (size_t)((((int64_t)M * K + 3) / 4) * 3);
                uint8_t * packed_data = (uint8_t *)malloc(packed_n);
                if (!packed_data) { free(scales); fclose(f); continue; }
                if (fread(packed_data, 1, packed_n, f) != packed_n) {
                    free(packed_data); free(scales); fclose(f); continue;
                }
                fclose(f);

                // Unpack INT6 -> INT8 (unpacked values are still INT6 range but stored as int8)
                // Call llama_set_prt_sidecar_int6 to handle via INT6 path in llama.cpp
                int64_t n_elements = (int64_t)M * K;
                int8_t * int8_data = (int8_t *)malloc((size_t)n_elements);
                if (!int8_data) { free(packed_data); free(scales); continue; }

                size_t packed_idx = 0;
                for (int64_t i = 0; i < n_elements; i += 4) {
                    uint8_t b0 = packed_data[packed_idx++];
                    uint8_t b1 = packed_data[packed_idx++];
                    uint8_t b2 = packed_data[packed_idx++];

                    // Unpack 4 INT6 values (offset-32)
                    // Byte 0: bits 0-5 = v0, bits 6-7 = v1[0:1]
                    // Byte 1: bits 0-3 = v1[2:5], bits 4-7 = v2[0:3]
                    // Byte 2: bits 0-1 = v2[4:5], bits 2-7 = v3[0:5]
                    int8_t v0 = (int8_t)(b0 & 0x3F) - 32;
                    int8_t v1 = (int8_t)(((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F) - 32;
                    int8_t v2 = (int8_t)(((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F) - 32;
                    int8_t v3 = (int8_t)((b2 >> 2) & 0x3F) - 32;

                    int8_data[i] = v0;
                    if (i + 1 < n_elements) int8_data[i + 1] = v1;
                    if (i + 2 < n_elements) int8_data[i + 2] = v2;
                    if (i + 3 < n_elements) int8_data[i + 3] = v3;
                }

                free(packed_data);
                if (g_prt_predecode_f32_enabled) {
                    llama_set_prt_sidecar_int6_predecode_f32(l, int8_data, scales, M, K);
                } else {
                    llama_set_prt_sidecar_int6(l, int8_data, scales, M, K);
                }
                g_prt_int6_sidecar_buffers.push_back(int8_data);
                g_prt_int6_scale_buffers.push_back(scales);
                loaded++;
                total_sidecar_bytes += (size_t)raw_bytes;

                // Phase 15E: INT6 total timing
                auto t_int6_done = std::chrono::high_resolution_clock::now();
                double ms_int6_read = std::chrono::duration<double, std::milli>(t_int6_done - t_file_open).count();
                // Phase 15F: SHA — use manifest if valid, otherwise compute
                double ms_sha = 0.0;
                std::string sha_hex;
                if (hash_mode_manifest) {
                    for (auto & ml : manifest_layers) {
                        if (ml.layer == l) { sha_hex = ml.sha256; break; }
                    }
                } else {
                    auto t_sha_start = std::chrono::high_resolution_clock::now();
                    sha_hex = file_sha256_hex(path.c_str());
                    ms_sha = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_sha_start).count();
                    ms_sidecar_hash += ms_sha;
                }
                if (!sha_hex.empty()) unique_sha_set.insert(sha_hex);
                bool is_fallback = false;
                for (int fn : force_native_layers) { if (fn == l) { is_fallback = true; break; } }
                if (g_prt_log_file) {
                    double ms_layer_total = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_layer_start).count();
                    fprintf(g_prt_log_file, "[PRT_SIDECAR_LAYER] layer=%d file=ffn_up_layer%d_prt.int6 size=%ld sha256=%s status=loaded fallback=%s read_ms=%.2f hash_ms=%.2f layer_total_ms=%.2f\n",
                            l, l, (long)raw_bytes, sha_hex.c_str(), is_fallback ? "true" : "false",
                            ms_int6_read, ms_sha, ms_layer_total);
                    fflush(g_prt_log_file);
                }

            } else {
                // Float32 sidecar: .bin file
                std::string path = sidecar_dir + "/ffn_up_layer" + std::to_string(l) + "_prt.bin";
                struct stat st;
                if (stat(path.c_str(), &st) != 0) continue;
                int64_t bytes = st.st_size;
                int M = 0, N = 0;
                if      (bytes == (int64_t)896  * 4864 * 4) { M = 896;  N = 4864; }
                else if (bytes == (int64_t)1536 * 8960 * 4) { M = 1536; N = 8960; }
                else if (bytes == (int64_t)2048 * 11008 * 4) { M = 2048; N = 11008; }
                else {
                    fprintf(stderr, "[PRT] Unknown sidecar size %ld for layer %d, skipping\n", (long)bytes, l);
                    continue;
                }
                FILE * f = fopen(path.c_str(), "rb");
                if (!f) continue;
                size_t n = (size_t)M * N;
                float * data = nullptr;
                if (params.prt_sidecar_mmap) {
                    int fd = fileno(f);
                    data = (float *)mmap(nullptr, n * sizeof(float), PROT_READ, MAP_PRIVATE, fd, 0);
                    if (data == MAP_FAILED) {
                        data = (float *)malloc(n * sizeof(float));
                        if (data && fread(data, sizeof(float), n, f) != n) { free(data); data = nullptr; }
                    }
                    fclose(f);
                } else {
                    data = (float *)malloc(n * sizeof(float));
                    if (!data) { fclose(f); continue; }
                    if (fread(data, sizeof(float), n, f) != n) { free(data); fclose(f); continue; }
                    fclose(f);
                }
                if (!data) continue;
                llama_set_prt_sidecar(l, data, M, N);
                g_prt_sidecar_buffers.push_back(data);
                loaded++;
                total_sidecar_bytes += (size_t)bytes;

                // Phase 15C: provenance logging
                std::string sha_hex = file_sha256_hex(path.c_str());
                if (!sha_hex.empty()) unique_sha_set.insert(sha_hex);
                bool is_fallback = false;
                for (int fn : force_native_layers) { if (fn == l) { is_fallback = true; break; } }
                if (g_prt_log_file) {
                    fprintf(g_prt_log_file, "[PRT_SIDECAR_LAYER] layer=%d file=ffn_up_layer%d_prt.bin size=%ld sha256=%s status=loaded fallback=%s\n",
                            l, l, (long)bytes, sha_hex.c_str(), is_fallback ? "true" : "false");
                    fflush(g_prt_log_file);
                }
            }
        }
        auto sidecar_load_end = std::chrono::high_resolution_clock::now();
        double sidecar_load_ms = std::chrono::duration<double, std::milli>(
            sidecar_load_end - sidecar_load_start).count();
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT_TIMING] sidecar_load_ms=%.2f sidecar_unpack_total_ms=%.2f sidecar_hash_total_ms=%.2f provenance_header_ms=%.2f\n",
                    sidecar_load_ms, ms_sidecar_unpack, ms_sidecar_hash, ms_provenance_begin);
            fflush(g_prt_log_file);
        }
        fprintf(stderr, "[PRT] Loaded %d/%d sidecars from %s\n", loaded, n_layer, sidecar_dir.c_str());
        if (loaded > 0) {
            extern int g_prt_sidecar_M[36];
            extern int g_prt_sidecar_N[36];
            int M = (loaded > 0 && g_prt_sidecar_M[0] > 0) ? g_prt_sidecar_M[0] : 896;
            int N = (loaded > 0 && g_prt_sidecar_N[0] > 0) ? g_prt_sidecar_N[0] : 4864;
            size_t bytes_per_layer = (loaded > 0) ? total_sidecar_bytes / loaded : 0;
            if (g_prt_log_file) {
                fprintf(g_prt_log_file, "[PRT_FORMAT] sidecar_format=%s scale_scheme=%s\n",
                        provenance_fmt_str.c_str(), use_int8 ? "per_row" : (use_int6 ? "per_row" : "none"));
                fprintf(g_prt_log_file, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer, M, N);
                // Phase 14C-VERIFY: Add model-consistent detail log
                int model_hidden = (M < N) ? M : N;  // smaller dimension = hidden
                int model_ffn = (M > N) ? M : N;       // larger dimension = ffn
                fprintf(g_prt_log_file, "[PRT_SHAPE_DETAIL] n_layer=%d hidden=%d ffn=%d sidecar_rows=%d sidecar_cols=%d runtime_M=%d runtime_N=%d format=%s\n",
                        n_layer, model_hidden, model_ffn, M, N, M, N, provenance_fmt_str.c_str());
                fprintf(g_prt_log_file, "[PRT_LOAD] sidecars_loaded=%d/%d sidecar_bytes_per_layer=%zu total_sidecar_bytes=%zu\n",
                        loaded, n_layer, bytes_per_layer, total_sidecar_bytes);
                fprintf(g_prt_log_file, "[PRT_PROVENANCE] loaded_count=%d unique_sha_count=%zu\n",
                        loaded, unique_sha_set.size());
                if (unique_sha_set.size() < (size_t)loaded) {
                    fprintf(g_prt_log_file, "[PRT_PROVENANCE_WARNING] duplicate_sidecar_hashes=true unique_sha_count=%zu loaded_count=%d\n",
                            unique_sha_set.size(), loaded);
                }
                // Log model path (SHA256 = not computed at runtime for large GGUF)
                fprintf(g_prt_log_file, "[PRT_PROVENANCE] model_path=%s model_sha256=not_computed\n",
                        params.model.path.c_str());
                fprintf(g_prt_log_file, "[PRT_PROVENANCE_END]\n");
                fflush(g_prt_log_file);
            } else {
                fprintf(stderr, "[PRT_FORMAT] sidecar_format=%s scale_scheme=%s\n",
                        use_int8 ? "int8" : "float32", use_int8 ? "per_row" : "none");
                fprintf(stderr, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer, M, N);
                // Phase 14C-VERIFY: Add model-consistent detail log
                // Model semantics: hidden=896 for 0.5B, ffn=4864; hidden=2048 for 3B, ffn=11008
                // Map: Float32 loader M=hidden,N=ffn → CORRECT; INT8 loader M=ffn,N=hidden → SWAPPED
                // The AVX2 kernel bug compensates for the INT8 swap (Phase 13Y)
                // INT8: M is larger (ffn), N is smaller (hidden); Float32: M is smaller (hidden), N is larger (ffn)
                int model_hidden = (M < N) ? M : N;  // smaller dimension = hidden
                int model_ffn = (M > N) ? M : N;       // larger dimension = ffn
                fprintf(stderr, "[PRT_SHAPE_DETAIL] n_layer=%d hidden=%d ffn=%d sidecar_rows=%d sidecar_cols=%d runtime_M=%d runtime_N=%d format=%s\n",
                        n_layer, model_hidden, model_ffn, M, N, M, N, use_int8 ? "int8" : "float32");
                fprintf(stderr, "[PRT_LOAD] sidecars_loaded=%d/%d sidecar_bytes_per_layer=%zu total_sidecar_bytes=%zu\n",
                        loaded, n_layer, bytes_per_layer, total_sidecar_bytes);
            }
        }

        // Set force-native layers if specified
        if (!params.prt_force_native.empty()) {
            std::vector<int> layers;
            std::string s = params.prt_force_native;
            size_t start = 0;
            for (size_t i = 0; i <= s.size(); i++) {
                if (i == s.size() || s[i] == ',') {
                    std::string tok = s.substr(start, i - start);
                    try { layers.push_back(std::stoi(tok)); } catch (...) {}
                    start = i + 1;
                }
            }
            if (!layers.empty()) {
                llama_set_prt_force_native_layers((int)layers.size(), layers.data());
            }
        }

        // Phase 13W: pre-touch sidecar pages if requested
        if (params.prt_pretouch_sidecars) {
            llama_pretouch_prt_sidecars();
        }

        // Phase 13X: log AVX2/FMA build configuration
        llama_dump_prt_build_info();
    }

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_meta();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    auto add_system_prompt = [&]() {
        if (!params.system_prompt.empty()) {
            ctx_cli.messages.push_back({
                {"role",    "system"},
                {"content", params.system_prompt}
            });
        }
    };
    add_system_prompt();

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /read <file>        add a text file\n");
    console::log("  /glob <pattern>     add text files using globbing pattern\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("\n");

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::string marker = ctx_cli.load_input_file(fname, false);
        if (marker.empty()) {
            console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
            return false;
        }
        if (inf.fim_sep_token != LLAMA_TOKEN_NULL) {
            cur_msg += common_token_to_piece(ctx_cli.ctx_server.get_llama_context(), inf.fim_sep_token, true);
            cur_msg += fname;
            cur_msg.push_back('\n');
        } else {
            cur_msg += "--- File: ";
            cur_msg += fname;
            cur_msg += " ---\n";
        }
        cur_msg += marker;
        console::log("Loaded text from '%s'\n", fname.c_str());
        return true;
    };

    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            add_system_prompt();

            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, "~")) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = std::string(home) + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                if (!add_text_file((rel_path / rel).string())) {
                    continue;
                }

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    console::error("Maximum number of globbed files allowed (%zu) reached.\n", FILE_GLOB_MAX_RESULTS);
                    break;
                }
            }
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second, timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    // Phase 13V: dump PRT per-call timing summary
    if (params.prt_mode > 0) {
        llama_dump_prt_timing_summary();
    }

    return 0;
}
