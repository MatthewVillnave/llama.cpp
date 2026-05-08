#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
// #include "log.h"

// PRT (Perturbation) API - only available when libllama has PRT support
extern "C" void llama_set_prt_debug_mode(int mode);
extern "C" void llama_set_prt_sidecar(int layer, const float * data, int M, int N);
extern "C" void llama_set_prt_sidecar_int8(int layer, const int8_t * int8_data, const float * scales, int M, int N);
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

    // PRT Phase 14B: separate buffers for float32 and INT8 sidecars
    static std::vector<float *> g_prt_sidecar_buffers;
    static std::vector<int8_t *> g_prt_int8_sidecar_buffers;
    static std::vector<float *> g_prt_int8_scale_buffers;
    bool use_int8 = (params.prt_sidecar_format == "int8");
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
        // Note: llama_set_prt_debug_mode logs "Debug mode set to N" internally via g_prt_log_file

        // Load sidecars
        // Sidecar files are pure float arrays without headers.
        // Detect M/N from file size: M*N*4 = bytes.
        // Known shapes: Qwen2.5-0.5B=896*4864, Qwen2.5-1.5B=1536*8960, Qwen2.5-3B=2048*11008
        // Phase 14B: log format selection
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT_FORMAT] sidecar_format=%s scale_scheme=%s\n",
                    use_int8 ? "int8" : "float32", use_int8 ? "per_row" : "none");
            fflush(g_prt_log_file);
        }
        auto sidecar_load_start = std::chrono::high_resolution_clock::now();
        std::string sidecar_dir = params.prt_sidecar_dir.empty() ? "/tmp/prt_sidecars/" : params.prt_sidecar_dir;
        const llama_model * model = llama_get_model(ctx_cli.ctx_server.get_llama_context());
        int n_layer = llama_model_n_layer(model);
        int loaded = 0;
        size_t total_sidecar_bytes = 0;
        for (int l = 0; l < n_layer; l++) {
            // Phase 14B: branch based on format
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
                int M = 0, K = 0;
                int64_t int8_bytes_3b = (int64_t)11008 * 2048;
                int64_t scale_bytes_3b = (int64_t)11008 * 4;
                int64_t int8_bytes_05b = (int64_t)4864 * 896;
                int64_t scale_bytes_05b = (int64_t)4864 * 4;
                int64_t total_3b = int8_bytes_3b + scale_bytes_3b;
                int64_t total_05b = int8_bytes_05b + scale_bytes_05b;
                if (raw_bytes == total_3b) { M = 11008; K = 2048; }
                else if (raw_bytes == total_05b) { M = 4864; K = 896; }
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
            }
        }
        auto sidecar_load_end = std::chrono::high_resolution_clock::now();
        double sidecar_load_ms = std::chrono::duration<double, std::milli>(
            sidecar_load_end - sidecar_load_start).count();
        if (g_prt_log_file) {
            fprintf(g_prt_log_file, "[PRT_TIMING] sidecar_load_ms=%.2f\n", sidecar_load_ms);
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
                        use_int8 ? "int8" : "float32", use_int8 ? "per_row" : "none");
                fprintf(g_prt_log_file, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer, M, N);
                fprintf(g_prt_log_file, "[PRT_LOAD] sidecars_loaded=%d/%d sidecar_bytes_per_layer=%zu total_sidecar_bytes=%zu\n",
                        loaded, n_layer, bytes_per_layer, total_sidecar_bytes);
                fflush(g_prt_log_file);
            } else {
                fprintf(stderr, "[PRT_FORMAT] sidecar_format=%s scale_scheme=%s\n",
                        use_int8 ? "int8" : "float32", use_int8 ? "per_row" : "none");
                fprintf(stderr, "[PRT_SHAPE] n_layer=%d M=%d N=%d\n", n_layer, M, N);
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
