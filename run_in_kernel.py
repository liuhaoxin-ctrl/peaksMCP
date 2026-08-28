#!/usr/bin/env python
"""Execute code in the live peaksMCP kernel (zmq, by connection file) and return stdout/stderr/result."""
import sys, json, os
import jupyter_client

CONN = "/Users/haoxin/Library/Jupyter/runtime/kernel-17e0ec3f-c166-4386-a44d-9ea23f7bb9db.json"


def run(code: str, timeout: float = 600.0):
    cf = json.load(open(CONN))
    cf["key"] = cf.get("key", "")  # some versions omit
    kc = jupyter_client.BlockingKernelClient()
    kc.load_connection_info(cf)
    kc.start_channels()
    kc.wait_for_ready(timeout=30)
    msg_id = kc.execute(code, store_history=False)
    outputs = []
    try:
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
            except Exception as e:
                outputs.append(f"[TIMEOUT] {e}")
                break
            parent = msg["parent_header"].get("msg_id")
            if parent != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                outputs.append(content["text"])
            elif mtype in ("execute_result", "display_data"):
                txt = content.get("data", {}).get("text/plain")
                if txt is not None:
                    outputs.append(txt)
                for k, v in content.get("data", {}).items():
                    if k.startswith("image/"):
                        outputs.append(f"[{k} rendered]")
            elif mtype == "error":
                outputs.append("".join(content.get("traceback", [])) or f"{content.get('ename')}: {content.get('evalue')}")
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
    finally:
        kc.stop_channels()
    return "".join(outputs)


if __name__ == "__main__":
    code = sys.stdin.read() if not sys.argv[1:] else sys.argv[1]
    print(run(code))
