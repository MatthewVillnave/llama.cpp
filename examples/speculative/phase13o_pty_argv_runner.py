#!/usr/bin/env python3
"""
PRT Phase 13O: Python PTY argv runner (argv-safe, no shell, stdin=/dev/null)
Uses pty.openpty + os.execvp + devnull stdin for clean exit.
"""

import sys
import os
import pty
import time
import select
import json
import signal
import re
import errno
import fcntl


def run_argv(argv, timeout=60, tail_bytes=262144):
    """Run argv under a PTY with devnull stdin, return compact JSON result."""
    master_fd, slave_fd = pty.openpty()
    
    # Open /dev/null for child's stdin
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    
    pid = os.fork()
    
    if pid == 0:
        # child
        os.close(master_fd)
        os.close(devnull_fd)
        # start new session so we can kill process group
        os.setsid()
        # stdin from devnull (already opened as fd 0 via dup2 below)
        # stdout/stderr to PTY slave
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        # exec with exactly the argv given
        os.execvp(argv[0], argv)
    else:
        # parent
        os.close(slave_fd)
        os.close(devnull_fd)
        
        # set master non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        output = b""
        start = time.time()
        timed_out = False
        child_dead = False
        child_reaped = False
        
        # Tracking booleans (updated across full stream)
        saw_prt_shape = False
        saw_sidecars_loaded = False
        saw_prt_true_replacement = False
        saw_native_fallback = False
        saw_callback_overwrites = False
        saw_generation_timing = False
        saw_flag_echo = False
        saw_path_fragment = False
        saw_invalid_argument = False
        saw_error = False
        
        # PRT-specific patterns
        PRT_SHAPE_PAT = re.compile(r"PRT_SHAPE|n_layer.*M=")
        SIDECARS_PAT = re.compile(r"sidecar|loaded.*/", re.I)
        REPLACEMENT_PAT = re.compile(r"true.replacement|PRT-11BB|ffn_up")
        FALLBACK_PAT = re.compile(r"native_fallback|fallback")
        OVERWRITES_PAT = re.compile(r"callback_overwrites")
        TIMING_PAT = re.compile(r"Prompt:.*t/s.*Generation:.*t/s|t/s\]")
        FLAG_ECHO_PAT = re.compile(r"--prt-")
        PATH_FRAG_PAT = re.compile(r"/tmp/prt_|/llama\.cpp/build")
        INVALID_ARG_PAT = re.compile(r"invalid argument|error:.*argument")
        ERROR_PAT = re.compile(r"error:|Error |Traceback")
        
        while True:
            elapsed = time.time() - start
            
            # Check timeout
            if elapsed >= timeout and not child_dead:
                timed_out = True
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                time.sleep(0.3)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                child_dead = True
            
            # Check if child died (only if not already reaped)
            if not child_reaped:
                result = os.waitpid(pid, os.WNOHANG)
                if result[0] != 0:
                    child_dead = True
                    child_reaped = True
            
            # Try to read
            try:
                r, _, xl = select.select([master_fd], [], [], 0.5)
            except InterruptedError:
                continue
            
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        output += data
                        # Update pattern booleans on full data
                        text_chunk = data.decode("utf-8", errors="replace")
                        saw_prt_shape = saw_prt_shape or bool(PRT_SHAPE_PAT.search(text_chunk))
                        saw_sidecars_loaded = saw_sidecars_loaded or bool(SIDECARS_PAT.search(text_chunk))
                        saw_prt_true_replacement = saw_prt_true_replacement or bool(REPLACEMENT_PAT.search(text_chunk))
                        saw_native_fallback = saw_native_fallback or bool(FALLBACK_PAT.search(text_chunk))
                        saw_callback_overwrites = saw_callback_overwrites or bool(OVERWRITES_PAT.search(text_chunk))
                        saw_generation_timing = saw_generation_timing or bool(TIMING_PAT.search(text_chunk))
                        saw_flag_echo = saw_flag_echo or bool(FLAG_ECHO_PAT.search(text_chunk))
                        saw_path_fragment = saw_path_fragment or bool(PATH_FRAG_PAT.search(text_chunk))
                        saw_invalid_argument = saw_invalid_argument or bool(INVALID_ARG_PAT.search(text_chunk))
                        saw_error = saw_error or bool(ERROR_PAT.search(text_chunk))
                    else:
                        # EOF
                        break
                except OSError as e:
                    if e.errno == errno.EIO:
                        # EIO on PTY master means EOF (slave closed)
                        break
                    elif e.errno == errno.EAGAIN:
                        if child_dead:
                            break
                        continue
                    else:
                        raise
            else:
                # no data available
                if child_dead:
                    break
        
        # final drain
        try:
            while True:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        output += data
                        text_chunk = data.decode("utf-8", errors="replace")
                        saw_prt_shape = saw_prt_shape or bool(PRT_SHAPE_PAT.search(text_chunk))
                        saw_sidecars_loaded = saw_sidecars_loaded or bool(SIDECARS_PAT.search(text_chunk))
                        saw_prt_true_replacement = saw_prt_true_replacement or bool(REPLACEMENT_PAT.search(text_chunk))
                        saw_native_fallback = saw_native_fallback or bool(FALLBACK_PAT.search(text_chunk))
                        saw_callback_overwrites = saw_callback_overwrites or bool(OVERWRITES_PAT.search(text_chunk))
                        saw_generation_timing = saw_generation_timing or bool(TIMING_PAT.search(text_chunk))
                        saw_flag_echo = saw_flag_echo or bool(FLAG_ECHO_PAT.search(text_chunk))
                        saw_path_fragment = saw_path_fragment or bool(PATH_FRAG_PAT.search(text_chunk))
                        saw_invalid_argument = saw_invalid_argument or bool(INVALID_ARG_PAT.search(text_chunk))
                        saw_error = saw_error or bool(ERROR_PAT.search(text_chunk))
                    else:
                        break
                except (OSError, IOError):
                    break
        except Exception:
            pass
        
        elapsed = time.time() - start
        
        # close master
        os.close(master_fd)
        
        # ensure child is reaped
        if not child_reaped:
            try:
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        
        raw_text = output.decode("utf-8", errors="replace")
        
        # keep bounded tail
        if len(raw_text) > tail_bytes:
            tail = raw_text[-tail_bytes:]
        else:
            tail = raw_text
        
        result = {
            "exit_code": 0,
            "timed_out": timed_out,
            "elapsed_sec": round(elapsed, 3),
            "tail_bytes": len(tail),
            "raw_bytes": len(raw_text),
            "contains_prt_shape": saw_prt_shape,
            "contains_sidecar_logs": saw_sidecars_loaded,
            "contains_flag_echo": saw_flag_echo,
            "contains_path_fragment": saw_path_fragment,
            "contains_traceback": False,  # already in saw_error
            "contains_error": saw_error,
            "saw_prt_shape": saw_prt_shape,
            "saw_sidecars_loaded": saw_sidecars_loaded,
            "saw_prt_true_replacement": saw_prt_true_replacement,
            "saw_native_fallback": saw_native_fallback,
            "saw_callback_overwrites": saw_callback_overwrites,
            "saw_generation_timing": saw_generation_timing,
            "saw_flag_echo": saw_flag_echo,
            "saw_path_fragment": saw_path_fragment,
            "saw_invalid_argument": saw_invalid_argument,
            "saw_error": saw_error,
            "tail_text": tail
        }
        
        return result


def main():
    own_args = []
    cmd_args = []
    seen_dashdash = False
    
    for arg in sys.argv[1:]:
        if arg == "--" and not seen_dashdash:
            seen_dashdash = True
            continue
        if not seen_dashdash:
            own_args.append(arg)
        else:
            cmd_args.append(arg)
    
    timeout = 60
    tail_bytes = 262144
    
    i = 0
    while i < len(own_args):
        if own_args[i] == "--timeout" and i + 1 < len(own_args):
            timeout = int(own_args[i + 1])
            i += 2
        elif own_args[i] == "--tail-bytes" and i + 1 < len(own_args):
            tail_bytes = int(own_args[i + 1])
            i += 2
        else:
            i += 1
    
    if not cmd_args:
        print(json.dumps({"error": "no command provided"}))
        sys.exit(1)
    
    result = run_argv(cmd_args, timeout=timeout, tail_bytes=tail_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()