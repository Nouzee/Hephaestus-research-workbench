"""PostToolUse hook — log every Bash/Edit/Write operation to a file."""
import json
import os
import sys
import traceback
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "tool_use.log")
ERR_FILE = os.path.join(LOG_DIR, "hook_errors.log")
os.makedirs(LOG_DIR, exist_ok=True)

def log_error(msg: str):
    with open(ERR_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            log_error("empty stdin — hook called without input")
            sys.exit(0)

        hook_input = json.loads(raw)
        tool_name = hook_input.get("tool_name", "unknown")
        tool_input = hook_input.get("tool_input", {})

        if tool_name not in ("Bash", "Edit", "Write"):
            sys.exit(0)

        summary = _summarize(tool_name, tool_input)
        entry = {
            "ts": datetime.now().isoformat(),
            "tool": tool_name,
            "summary": summary,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        log_error(traceback.format_exc())

    sys.exit(0)


def _summarize(tool_name: str, inp: dict) -> str:
    if tool_name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        return f"{desc} | cmd={cmd[:120]}"
    elif tool_name == "Edit":
        fp = inp.get("file_path", "")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        return f"{fp} | replace {len(old)}→{len(new)} chars"
    elif tool_name == "Write":
        fp = inp.get("file_path", "")
        content = inp.get("content", "")
        return f"{fp} | {len(content)} chars"
    return str(inp)[:200]


if __name__ == "__main__":
    main()
