#!/usr/bin/env python3
"""
PRT Phase 21J: PTY Capture Helper v4
Uses pexpect-style forkpty with env pass-through via execve.
"""

import os, sys, time, select, fcntl, json

os.chdir("/home/matthew-villnave/llama.cpp")

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
PROMPT = "The capital of France is"

def pty_capture(arg_list, extra_env=None, timeout=30):
    """Run llama-cli under PTY with extra env vars. Returns (exit_code, output, wall, to, killed)."""
    all_env = os.environ.copy()
    if extra_env:
        all_env.update(extra_env)
    
    pid, fd = os.forkpty()
    if pid == 0:
        os.execve("./build/bin/llama-cli", ["./build/bin/llama-cli"] + arg_list, all_env)
        os._exit(1)
    
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    
    output = b""
    start = time.time()
    timed_out = False
    
    while time.time() - start < timeout:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid != 0:
            exit_code = os.WEXITSTATUS(status)
            break
        r, _, _ = select.select([fd], [], [], 0.5)
        if r:
            try:
                chunk = os.read(fd, 8192)
                if chunk:
                    output += chunk
            except OSError:
                break
    else:
        timed_out = True
    
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk: break
            output += chunk
    except OSError:
        pass
    
    killed = False
    wpid, status = os.waitpid(pid, os.WNOHANG)
    if wpid == 0:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        killed = True
        exit_code = -1
    else:
        exit_code = os.WEXITSTATUS(status)
    
    os.close(fd)
    return exit_code, output, time.time() - start, timed_out, killed

def filter_out(text):
    import re
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x08.', '', text)
    lines = text.split('\n')
    result = []
    for line in lines:
        s = line.strip()
        if not s: continue
        if any(p in s for p in ['[PRT-NATIVE]', '[PRT_V2', 'IL=', 'up=0x', '| /', '====', '----']): continue
        if 'add a text' in s or 'add text files' in s: continue
        if s in ('build', 'model', 'avail', 'command'): continue
        result.append(s)
    return result

results = {}

tests = [
    ("native_n8",    ["-m", MODEL, "-p", PROMPT, "-n", "8",  "--temp", "0"],                  None,                              30),
    ("f32_n1",       ["-m", MODEL, "-p", PROMPT, "-n", "1",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    30),
    ("f32_n4",       ["-m", MODEL, "-p", PROMPT, "-n", "4",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    60),
    ("f32_n8",       ["-m", MODEL, "-p", PROMPT, "-n", "8",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    60),
    ("int8_n1",      ["-m", MODEL, "-p", PROMPT, "-n", "1",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    30),
    ("int8_n2",      ["-m", MODEL, "-p", PROMPT, "-n", "2",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    60),
    ("int8_n4",      ["-m", MODEL, "-p", PROMPT, "-n", "4",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    60),
    ("int8_n8",      ["-m", MODEL, "-p", PROMPT, "-n", "8",  "--temp", "0"],                  {"PRT_GGML_TEST_LAYER": "0"},    120),
]

for name, args, env, timeout in tests:
    print(f"\n=== {name} ===")
    sys.stdout.flush()
    ec, out, wt, to, killed = pty_capture(args, env, timeout=timeout)
    text = filter_out(out.decode('utf-8', errors='replace'))
    results[name] = {'exit': ec, 'wall': round(wt, 1), 'timed_out': to, 'killed': killed, 'text': text}
    print(f"Exit: {ec}, Wall: {wt:.1f}s, To: {to}, Killed: {killed}")
    for t in text[-5:]:
        print(f"  {t}")
    sys.stdout.flush()

print("\n=== SUMMARY ===")
for k, r in results.items():
    txt = ' | '.join(r['text'][:3])
    print(f"{k}: exit={r['exit']} killed={r['killed']} | {txt}")

with open("/tmp/phase21j_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved /tmp/phase21j_results.json")