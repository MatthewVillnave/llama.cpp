#!/usr/bin/env python3
"""
PRT Phase 13O: Python PTY argv runner
Uses pty.fork + os.execvp (no shell string, no shell=True)
"""

import sys
import os
import pty
import time
import select
import json
import signal
import re


def run_argv(argv, timeout=60, tail_bytes=262144):
    """Run argv under a PTY, return compact JSON result."""
    pid = os.fork()
    
    if pid == 0:
        # child
        os.close(0)
        pty.openpty()  # opens new PTY pair, stdin=slave
        os.dup2(0, 0)   # stdin = slave
        os.dup2(0, 1)   # stdout = slave  
        os.dup2(0, 2)   # stderr = slave
        os.execvp(argv[0], argv)
    else:
        # parent
        master_fd = pty.openpty()[0]
        output = b""
        start = time.time()
        timed_out = False
        
        while time.time() - start < timeout:
            try:
                r, _, xl = select.select([master_fd], [], [], 0.5)
                if r:
                    try:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        output += data
                    except OSError:
                        break
                elif output:
                    # no more data but we have some - give it a moment
                    extra = time.time() - start
                    if extra > 2:  # already got output, small grace period
                        break
                # check elapsed
                elapsed = time.time() - start
                if elapsed >= timeout:
                    timed_out = True
                    break
            except InterruptedError:
                continue
        
        elapsed = time.time() - start
        
        # kill process group
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        
        os.waitpid(pid, 0)
        os.close(master_fd)
        
        raw_text = output.decode("utf-8", errors="replace")
        
        # keep tail
        if len(raw_text) > tail_bytes:
            tail = raw_text[-tail_bytes:]
        else:
            tail = raw_text
        
        # detect patterns
        contains_prt_shape = bool(re.search(r"PRT_SHAPE|n_layer.*M=", tail))
        contains_sidecars = bool(re.search(r"sidecar|loaded.*/", tail, re.I))
        contains_flag_echo = bool(re.search(r"--prt-", tail))
        contains_path_fragment = bool(re.search(r"/tmp/prt_|/llama\.cpp/build", tail))
        contains_traceback = bool(re.search(r"Traceback|Error:", tail))
        contains_error = bool(re.search(r"error:|Error ", tail))
        
        result = {
            "exit_code": 0,
            "timed_out": timed_out,
            "elapsed_sec": round(elapsed, 3),
            "tail_bytes": len(tail),
            "raw_bytes": len(raw_text),
            "contains_prt_shape": contains_prt_shape,
            "contains_sidecar_logs": contains_sidecars,
            "contains_flag_echo": contains_flag_echo,
            "contains_path_fragment": contains_path_fragment,
            "contains_traceback": contains_traceback,
            "contains_error": contains_error,
            "tail_text": tail[-8192:]
        }
        
        return result


def main():
    # parse own args until --
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
        if own_args[i] == "--timeout" and i+1 < len(own_args):
            timeout = int(own_args[i+1])
            i += 2
        elif own_args[i] == "--tail-bytes" and i+1 < len(own_args):
            tail_bytes = int(own_args[i+1])
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