"""
STAGE 2: AUTOMATED EVALUATION

Runs every question in the evaluation set through the retriever and checks
the results against the expected papers and pages you wrote by hand.

Usage:
    python evaluate.py                     # retrieval metrics at k = 3, 5, 10 (free, no LLM calls)
    python evaluate.py --k 5 10 20         # choose your own k values
    python evaluate.py --generate          # also get LLM answers at k = config.TOP_K (uses API calls)
    python evaluate.py --questions eval/question.json

Outputs go to eval/results/:
    <run>_retrieval.csv   one row per question per k (open in Excel / Numbers)
    <run>_summary.json    averaged metrics + the settings used for this run
    <run>_answers.json    LLM answers next to your reference answers (with --generate)
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import config
from ask import build_prompt, generate, retrieve

REFUSAL_TEXT = "couldn't find this in the papers"
RESULTS_DIR = Path("eval/results")


# ---------------------------------------------------------------
# Load questions and catch typos in file names early.
# A misspelled file name would silently count as a retrieval miss.
# ---------------------------------------------------------------
def load_questions(path):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    papers = {p.name for p in Path(config.PAPERS_DIR).glob("*.pdf")}
    for q in questions:
        for src in q["expected_sources"]:
            if src not in papers:
                print(f"WARNING {q['id']}: '{src}' is not in {config.PAPERS_DIR}/ (typo?)")
    return questions


# ---------------------------------------------------------------
# Score one question at one value of k.
#
#   paper_hit : 1 if ANY expected paper appears in the top k, else 0
#   page_hit  : 1 if ANY retrieved chunk comes from an expected page
#   coverage  : fraction of expected papers found (matters for multi-paper questions)
#   rr        : reciprocal rank = 1 / position of the first relevant chunk
#               (1.0 if it's ranked 1st, 0.5 if 2nd, 0.33 if 3rd, 0 if missing).
#               Averaged over questions this becomes MRR (Mean Reciprocal Rank).
#   top_score : similarity of the best chunk (compare in-scope vs out-of-scope!)
# ---------------------------------------------------------------
def score_retrieval(q, hits, k):
    top = hits[:k]
    expected = set(q["expected_sources"])
    expected_pages = {(src, p) for src, pages in q.get("expected_pages", {}).items() for p in pages}

    row = {
        "id": q["id"],
        "type": q["type"],
        "k": k,
        "top_score": round(top[0]["score"], 3) if top else "",
        "retrieved": "; ".join(f"{h['source']}:p{h['page']}" for h in top),
        "paper_hit": "", "page_hit": "", "coverage": "", "rr": "",
    }
    if not expected:  # out-of-scope: there is no "right" chunk to find
        return row

    ranks = [i for i, h in enumerate(top, start=1) if h["source"] in expected]
    found = {h["source"] for h in top} & expected

    row["paper_hit"] = int(bool(ranks))
    row["rr"] = round(1 / ranks[0], 3) if ranks else 0.0
    row["coverage"] = round(len(found) / len(expected), 3)
    if expected_pages:
        row["page_hit"] = int(any((h["source"], h["page"]) in expected_pages for h in top))
    return row


def mean(values):
    values = [v for v in values if v != ""]
    return round(sum(values) / len(values), 3) if values else None


# ---------------------------------------------------------------
# Average the per-question scores, per k and per question type.
# ---------------------------------------------------------------
def summarize(rows, ks):
    groups = sorted({r["type"] for r in rows if r["type"] != "out_of_scope"})
    summary = []
    for k in ks:
        at_k = [r for r in rows if r["k"] == k]
        for group in groups + ["ALL in-scope", "out_of_scope"]:
            if group == "ALL in-scope":
                sel = [r for r in at_k if r["type"] != "out_of_scope"]
            else:
                sel = [r for r in at_k if r["type"] == group]
            if not sel:
                continue
            summary.append({
                "k": k, "group": group, "n": len(sel),
                "paper_hit": mean([r["paper_hit"] for r in sel]),
                "page_hit": mean([r["page_hit"] for r in sel]),
                "coverage": mean([r["coverage"] for r in sel]),
                "mrr": mean([r["rr"] for r in sel]),
                "top_score": mean([r["top_score"] for r in sel]),
            })
    return summary


def print_summary(summary):
    def fmt(v):
        return f"{v:>9.3f}" if isinstance(v, float) else f"{'-':>9}"

    print(f"\n{'k':>3}  {'group':<14}{'n':>3}{'PaperHit':>10}{'PageHit':>9}"
          f"{'Coverage':>10}{'MRR':>9}{'TopScore':>10}")
    last_k = None
    for s in summary:
        if last_k is not None and s["k"] != last_k:
            print()
        last_k = s["k"]
        print(f"{s['k']:>3}  {s['group']:<14}{s['n']:>3} {fmt(s['paper_hit'])}{fmt(s['page_hit'])}"
              f" {fmt(s['coverage'])}{fmt(s['mrr'])} {fmt(s['top_score'])}")


# ---------------------------------------------------------------
# Optional: generate answers and check refusal behaviour.
#   out-of-scope questions SHOULD refuse
#   in-scope questions should NOT refuse (a "false refusal")
# Answer correctness and faithfulness are judged later, by you or an LLM judge.
# ---------------------------------------------------------------
def run_generation(questions, hits_by_id, k, sleep):
    records = []
    for q in questions:
        hits = hits_by_id[q["id"]][:k]
        try:
            answer = generate(build_prompt(q["question"], hits))
        except Exception as e:  # e.g. rate limit: record it and keep going
            answer = f"[ERROR] {e}"
        refused = REFUSAL_TEXT in answer.lower().replace("\u2019", "'")
        should_refuse = q["type"] == "out_of_scope"
        records.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "retrieved": [f"{h['source']}:p{h['page']}" for h in hits],
            # the exact text the LLM saw - judge.py needs it to check faithfulness
            "context": [{"source": h["source"], "page": h["page"], "text": h["text"]} for h in hits],
            "answer": answer, "reference_answer": q["reference_answer"],
            "refused": refused, "refusal_correct": refused == should_refuse,
        })
        status = "ok" if refused == should_refuse else "WRONG"
        print(f"  {q['id']}  refused={refused!s:<5}  [{status}]")
        time.sleep(sleep)  # stay under the free-tier rate limit
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls")
    parser.add_argument("--name", default="", help="optional label for this run, e.g. chunk400")
    args = parser.parse_args()

    ks = sorted(set(args.k))
    questions = load_questions(args.questions)
    print(f"{len(questions)} questions | chunk {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} | k = {ks}")

    # Retrieve ONCE at the largest k. The top 3 are just the first 3 of the
    # top 10, so smaller k values are computed by slicing - no extra searches.
    hits_by_id = {q["id"]: retrieve(q["question"], k=max(ks)) for q in questions}
    rows = [score_retrieval(q, hits_by_id[q["id"]], k) for k in ks for q in questions]
    summary = summarize(rows, ks)
    print_summary(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.name}" if args.name else "")

    with open(RESULTS_DIR / f"{run}_retrieval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    settings = {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                "embed_model": config.EMBED_MODEL, "ks": ks, "questions": args.questions}
    output = {"settings": settings, "retrieval": summary}

    if args.generate:
        print(f"\nGenerating answers at k = {config.TOP_K} ...")
        records = run_generation(questions, hits_by_id, config.TOP_K, args.sleep)
        oos = [r for r in records if r["type"] == "out_of_scope"]
        ins = [r for r in records if r["type"] != "out_of_scope"]
        output["generation"] = {
            "llm_model": config.LLM_MODEL, "k": config.TOP_K,
            "correct_refusal_rate": mean([int(r["refused"]) for r in oos]),
            "false_refusal_rate": mean([int(r["refused"]) for r in ins]),
        }
        (RESULTS_DIR / f"{run}_answers.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCorrect refusals (out-of-scope): {output['generation']['correct_refusal_rate']}")
        print(f"False refusals (in-scope):       {output['generation']['false_refusal_rate']}")

    (RESULTS_DIR / f"{run}_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to {RESULTS_DIR}/{run}_*")


if __name__ == "__main__":
    main()
