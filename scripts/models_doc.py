"""Write the 'How the models work' section of README.md from src/niveshrl/model_registry.py (records are read from
report/results/, so re-run after any evaluation to refresh the numbers).

    python scripts/models_doc.py
"""
from __future__ import annotations

from niveshrl import model_registry as MR
from niveshrl.config import ROOT

BEGIN, END = "<!-- models:begin -->", "<!-- models:end -->"


def main() -> None:
    p = ROOT / "README.md"
    text = p.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{MR.markdown()}\n{END}"
    if BEGIN in text and END in text:
        a, b = text.index(BEGIN), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        anchor = "## Architecture"
        i = text.index(anchor) if anchor in text else len(text)
        text = text[:i] + block + "\n\n" + text[i:]
    p.write_text(text, encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "models.md").write_text(MR.markdown() + "\n", encoding="utf-8")
    print(f"README section and docs/models.md written ({len(MR.CARDS)} models)")


if __name__ == "__main__":
    main()
