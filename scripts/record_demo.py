"""Record scripts/demo.py (steps 2-5: attack, without SMRITI, with SMRITI, laundering) as
HTML for the project page: site/demo.html.

    python scripts/record_demo.py

Runs the real demo against the model endpoint and NVIDIA hosted models (from .env), with the
in-memory store. Terminal colors become CSS classes; nothing else is changed.
"""
import contextlib
import html
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import demo  # noqa: E402

from smriti.store.fake_store import FakeStore  # noqa: E402

CLASSES = {"31": "r", "32": "g", "33": "y", "36": "c", "2": "d", "1": "b"}
ANSI = re.compile(r"\x1b\[(\d+)m")


def to_html(text: str) -> str:
    out, open_spans = [], 0
    pos = 0
    for m in ANSI.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        code = m.group(1)
        if code == "0":
            out.append("</span>" * open_spans)
            open_spans = 0
        elif code in CLASSES:
            out.append(f'<span class="{CLASSES[code]}">')
            open_spans += 1
        pos = m.end()
    out.append(html.escape(text[pos:]))
    out.append("</span>" * open_spans)
    return "".join(out)


def main() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            demo.main([], make_store=FakeStore)
        except Exception as e:           # steps after 5 need results files; not recorded
            print(f"\n[stopped: {type(e).__name__}]")
    text = buf.getvalue()
    end = text.find("== 6.")
    if end != -1:
        text = text[:text.rfind("\n", 0, end)]
    text = text.strip("\n")
    (ROOT / "site" / "demo.html").write_text(to_html(text) + "\n")
    print(text)


if __name__ == "__main__":
    main()
