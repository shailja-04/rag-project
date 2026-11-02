"""Retrieval evaluation: hit-rate@k on a labeled question set.

Each question lists expected keywords that MUST appear in the retrieved
chunks for it to count as a hit. Run:  python -m src.eval
"""

import json
from pathlib import Path

from .config import RAGConfig
from .retrieve import retrieve


def evaluate(questions_path: str = "eval/questions.json", cfg: RAGConfig | None = None, model=None) -> dict:
    cfg = cfg or RAGConfig()
    questions = json.loads(Path(questions_path).read_text())

    hits, details = 0, []
    for q in questions:
        chunks = retrieve(q["question"], cfg, model=model)
        retrieved_text = " ".join(c["text"] for c in chunks).lower()
        matched = [kw for kw in q["expected_keywords"] if kw.lower() in retrieved_text]
        is_hit = len(matched) > 0
        hits += is_hit
        details.append(
            {
                "question": q["question"],
                "hit": is_hit,
                "matched_keywords": matched,
                "top_score": chunks[0]["score"] if chunks else 0,
            }
        )

    total = len(questions)
    report = {
        "hit_rate": round(hits / total, 3) if total else 0,
        "hits": hits,
        "total": total,
        "top_k": cfg.top_k,
        "details": details,
    }

    print(f"\nRetrieval eval: {hits}/{total} hits "
          f"(hit-rate@{cfg.top_k} = {report['hit_rate']})")
    for d in details:
        mark = "PASS" if d["hit"] else "FAIL"
        print(f"  [{mark}] {d['question'][:70]}")
    return report


if __name__ == "__main__":
    evaluate()
