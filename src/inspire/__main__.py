"""Run the public example and print the evidence block."""

from pathlib import Path
import json

from src.inspire.pipeline import prepare


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    case = json.loads((ROOT / "examples" / "fictional_case.json").read_text(encoding="utf-8"))
    documents = json.loads((ROOT / "examples" / "fictional_snippets.json").read_text(encoding="utf-8"))
    result = prepare(case["case_text"], documents)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
