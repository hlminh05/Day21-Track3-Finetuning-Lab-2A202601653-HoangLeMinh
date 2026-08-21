#!/usr/bin/env python3
"""Export full base-(b) versus correct-LoRA predictions for report examples.

NB5 intentionally prints only short previews. The report, however, asks for paired
qualitative examples, including losses. This deterministic replay uses the already
frozen prompt and eval set, then stores full outputs and per-example task scores.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report  # noqa: E402
from labkit.config import get_tier  # noqa: E402


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    frozen = json.loads((ROOT / "results" / "baselines_frozen.json").read_text(
        encoding="utf-8"))

    prompt_sha = hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16]
    assert prompt_sha == frozen["optimized_prompt_sha"], "frozen prompt changed"
    assert len(target) == frozen["n_target"], "frozen eval slice changed"

    base, tok = generate.load_base(tier)
    preds_b, _ = generate.generate_batch(
        base, tok, [row["input"] for row in target],
        system=generate.OPTIMIZED_PROMPT, label="qual/base-b")
    del base
    generate.free_memory()

    from peft import PeftModel

    tuned, tok = generate.load_base(tier)
    tuned = PeftModel.from_pretrained(tuned, str(ROOT / "adapters" / "correct"))
    tuned.eval()
    preds_ft, _ = generate.generate_batch(
        tuned, tok, [row["input"] for row in target],
        system=generate.NAIVE_PROMPT, label="qual/fine-tune")

    rows = []
    for i, (row, pred_b, pred_ft) in enumerate(zip(target, preds_b, preds_ft)):
        b_score = ev.triage_field_accuracy(pred_b, row["label"])
        ft_score = ev.triage_field_accuracy(pred_ft, row["label"])
        relation = "win" if ft_score > b_score else "loss" if ft_score < b_score else "tie"
        rows.append({
            "i": i,
            "ticket": row["input"],
            "label": row["label"],
            "baseline_b_pred": pred_b,
            "baseline_b_score": round(b_score, 2),
            "fine_tune_pred": pred_ft,
            "fine_tune_score": round(ft_score, 2),
            "fine_tune_relation": relation,
        })

    report.write_json(rows, "qualitative_comparison.json", results_dir=ROOT / "results")
    counts = {key: sum(row["fine_tune_relation"] == key for row in rows)
              for key in ("win", "loss", "tie")}
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
