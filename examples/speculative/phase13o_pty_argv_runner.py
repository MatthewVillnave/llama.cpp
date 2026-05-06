#!/usr/bin/env python3
"""
PRT Phase 13O: Python PTY argv runner (argv-safe, no shell)
Uses pty.openpty + os.execvp, no shell string execution.
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
    """Run argv under a PTY, return compact JSON result."""
    master_fd, slave_fd = pty.openpty()
    
    pid = os.fork()
    
    if pid == 0:
        # child
        os.close(master_fd)
        # start new session so we can kill process group
        os.setsid()
        # dup slave to stdin/stdout/stderr
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        # exec with exactly the argv given
        os.execvp(argv[0], argv)
    else:
        # parent
        os.close(slave_fd)
        
        # set master non-blocking
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        output = b""
        start = time.time()
        timed_out = False
        child_dead = False
        child_reaped = False
        
        while True:
            elapsed = time.time() - start
            
            # Check if child died (only if not already reaped)
            if not child_reaped:
                result = os.waitpid(pid, os.WNOHANG)
                if result[0] != 0:
                    child_dead = True
                    child_reaped = True
            
            # Check timeout (only if child not dead yet)
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
                # wait for reaping
                try:
                    os.waitpid(pid, 0)
                    child_reaped = True
                except ProcessLookupError:
                    child_reaped = True
                child_dead = True
            
            # Try to read with timeout
            try:
                r, _, xl = select.select([master_fd], [], [], 0.5)
            except InterruptedError:
                continue
            
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        output += data
                    else:
                        # EOF
                        break
                except OSError as e:
                    if e.errno == errno.EIO:
                        # EIO means PTY closed (normal EOF)
                        break
                    elif e.errno == errno.EAGAIN:
                        # non-blocking, no data yet
                        if child_dead:
                            # child is dead and no more data - we're done
                            break
                        continue
                    else:
                        raise
            else:
                # no data available
                if child_dead:
                    # child is dead and no more data - we're done
                    break
        
        # final drain
        try:
            while True:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        output += data
                    else:
                        break
                except (OSError, IOError):
                    break
        except Exception:
            pass
        
        elapsed = time.time() - start
        
        # close master
        os.close(master_fd)
        
        # ensure child is reaped (only if not already)
        if not child_reaped:
            try:
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        
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
            "tail_text": tail
        }
        
        return result


def main():
    # parse own args before --
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