"""
STAGE 2 - PART 2: LLM-AS-A-JUDGE

Grades the answers saved by `evaluate.py --generate` on three things:
  correctness       : does the answer match YOUR reference answer?      (0, 1 or 2)
  faithfulness      : what fraction of its claims are supported by the retrieved chunks?
  citation accuracy : what fraction of its citations point to a chunk that supports the claim?

Usage:
    python judge.py eval/results/<run>_answers.json
    python judge.py eval/results/<run>_judged.csv --agreement   # compare judge with YOUR hand grades

Outputs (next to the input file):
    <run>_judged.json   full verdicts, including every claim and whether it was supported
    <run>_judged.csv    one row per question, with empty columns for your own grades
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import config
from ask import llm, retrieve

JUDGE_PROMPT = """You are a strict evaluator of a question-answering system that answers questions about research papers.

You receive:
- QUESTION
- REFERENCE ANSWER: written by a human expert from the papers. Treat it as ground truth.
- CONTEXT: the passages the system retrieved. Each starts with [file.pdf, p.N].
- SYSTEM ANSWER: the answer you must grade.

Do three things.

1. Split the SYSTEM ANSWER into its individual factual claims (short, self-contained statements).
   For each claim decide if it is SUPPORTED by the CONTEXT: the context states it, or it follows
   directly from the context. A claim that is true but comes from outside knowledge is NOT supported.
   Also record the citation the answer attached to the claim (e.g. "paper.pdf, p.2"), or null if none,
   and whether that cited passage actually supports the claim (true/false, or null if no citation).

2. Grade CORRECTNESS against the REFERENCE ANSWER:
   2 = contains the key facts of the reference and nothing that contradicts it
   1 = partially correct: some key facts missing, or minor errors
   0 = wrong, contradicts the reference, or misses the main point
   Extra true details that are not in the reference do not lower correctness.

3. Reply with ONLY a JSON object and no other text, in exactly this shape:
{"claims": [{"claim": "...", "supported": true, "citation": "paper.pdf, p.2", "citation_correct": true}],
 "correctness": 2,
 "correctness_reason": "one short sentence"}"""


# ---------------------------------------------------------------
# Get the exact chunk text the answering LLM saw.
# New answer files store it; older ones only stored "file:pN", so we
# re-run retrieval (same database -> same chunks) and warn if it differs.
# ---------------------------------------------------------------
def get_context(rec):
    if "context" in rec:
        return rec["context"]
    hits = retrieve(rec["question"], k=len(rec["retrieved"]))
    if [f"{h['source']}:p{h['page']}" for h in hits] != rec["retrieved"]:
        print(f"    WARNING {rec['id']}: re-retrieved chunks differ (database changed since this run?)")
    return [{"source": h["source"], "page": h["page"], "text": h["text"]} for h in hits]


def build_judge_input(rec, context):
    ctx = "\n\n".join(f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in context)
    return (f"QUESTION:\n{rec['question']}\n\n"
            f"REFERENCE ANSWER:\n{rec['reference_answer']}\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"SYSTEM ANSWER:\n{rec['answer']}")


# LLMs sometimes wrap JSON in ``` fences or add a sentence around it.
def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge reply")
    return json.loads(text[start:end + 1])


def call_judge(user_input, sleep, retries=3):
    extra = {}
    if "gpt-oss" in config.JUDGE_MODEL:
        extra["reasoning_effort"] = config.JUDGE_REASONING_EFFORT
    for attempt in range(1, retries + 1):
        try:
            response = llm.chat.completions.create(
                model=config.JUDGE_MODEL,
                temperature=0,  # same input -> same grade, as far as possible
                max_tokens=config.JUDGE_MAX_TOKENS,
                messages=[{"role": "system", "content": JUDGE_PROMPT},
                          {"role": "user", "content": user_input}],
                extra_body=extra,
            )
            return parse_json(response.choices[0].message.content)
        except Exception as e:  # bad JSON, rate limit, network...
            print(f"    judge attempt {attempt} failed: {e}")
            time.sleep(sleep * (attempt + 1))
    return None


# ---------------------------------------------------------------
# Grade one answer. Refusals are graded by rule, without the judge:
#   out-of-scope + refused   -> correct (2)
#   in-scope     + refused   -> false refusal (0)
# An out-of-scope question that got an answer scores correctness 0,
# and the judge still checks which of its claims came from the context.
# ---------------------------------------------------------------
def grade(rec, sleep):
    out_of_scope = rec["type"] == "out_of_scope"
    result = {"id": rec["id"], "type": rec["type"], "refused": rec["refused"],
              "correctness": None, "judge_reason": "", "faithfulness": None,
              "citation_accuracy": None, "n_claims": 0, "unsupported_claims": [], "claims": []}

    if rec["refused"]:
        result["correctness"] = 2 if out_of_scope else 0
        result["judge_reason"] = "correct refusal" if out_of_scope else "false refusal"
        return result

    verdict = call_judge(build_judge_input(rec, get_context(rec)), sleep)
    if verdict is None:
        result["judge_reason"] = "JUDGE FAILED - grade this one by hand"
        return result

    claims = verdict.get("claims", [])
    cited = [c for c in claims if c.get("citation")]
    result.update({
        "claims": claims,
        "n_claims": len(claims),
        "unsupported_claims": [c.get("claim", "") for c in claims if c.get("supported") is not True],
        "faithfulness": round(sum(c.get("supported") is True for c in claims) / len(claims), 3) if claims else None,
        "citation_accuracy": round(sum(c.get("citation_correct") is True for c in cited) / len(cited), 3) if cited else None,
        "correctness": 0 if out_of_scope else verdict.get("correctness"),
        "judge_reason": "answered an out-of-scope question" if out_of_scope else verdict.get("correctness_reason", ""),
    })
    return result


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(results):
    groups = sorted({r["type"] for r in results if r["type"] != "out_of_scope"})
    rows = []
    for group in groups + ["ALL in-scope", "out_of_scope"]:
        sel = [r for r in results if (r["type"] != "out_of_scope" if group == "ALL in-scope" else r["type"] == group)]
        if not sel:
            continue
        graded = [r["correctness"] for r in sel if r["correctness"] is not None]
        rows.append({
            "group": group, "n": len(sel),
            # correctness reported on a 0-1 scale (score / 2)
            "correctness": mean([c / 2 for c in graded]),
            "fully_correct": mean([int(c == 2) for c in graded]),
            "faithfulness": mean([r["faithfulness"] for r in sel]),
            "citation_acc": mean([r["citation_accuracy"] for r in sel]),
        })
    return rows


def print_summary(rows):
    def fmt(v):
        return f"{v:>13.3f}" if isinstance(v, float) else f"{'-':>13}"
    print(f"\n{'group':<14}{'n':>3}{'Correctness':>13}{'FullyCorrect':>13}{'Faithfulness':>13}{'CitationAcc':>13}")
    for s in rows:
        print(f"{s['group']:<14}{s['n']:>3}{fmt(s['correctness'])}{fmt(s['fully_correct'])}"
              f"{fmt(s['faithfulness'])}{fmt(s['citation_acc'])}")
    print("(Faithfulness and citation accuracy only count questions that were answered, not refused.)")


# ---------------------------------------------------------------
# Agreement check: how often does the judge's correctness grade match yours?
# Fill the human_correctness column in the CSV (0, 1 or 2) for 8-10 questions.
# ---------------------------------------------------------------
def agreement(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness"].strip() and r["correctness"].strip()]
    if not rows:
        print("No rows have both human_correctness and correctness filled in.")
        return
    same = [r for r in rows if int(float(r["correctness"])) == int(float(r["human_correctness"]))]
    within_one = [r for r in rows if abs(int(float(r["correctness"])) - int(float(r["human_correctness"]))) <= 1]
    print(f"Hand-graded questions: {len(rows)}")
    print(f"Exact agreement:       {len(same)}/{len(rows)} = {len(same) / len(rows):.2f}")
    print(f"Within one point:      {len(within_one)}/{len(rows)} = {len(within_one) / len(rows):.2f}")
    for r in rows:
        if r not in same:
            print(f"  {r['id']}: judge={r['correctness']}  you={r['human_correctness']}  ({r['judge_reason']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="<run>_answers.json, or <run>_judged.csv with --agreement")
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--sleep", type=float, default=3.0, help="seconds between judge calls")
    args = parser.parse_args()

    if args.agreement:
        agreement(args.path)
        return

    in_path = Path(args.path)
    records = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"Judging {len(records)} answers with {config.JUDGE_MODEL} ...")

    results = []
    for rec in records:
        r = grade(rec, args.sleep)
        results.append(r)
        print(f"  {r['id']}  correctness={r['correctness']}  faithfulness={r['faithfulness']}"
              f"  unsupported={len(r['unsupported_claims'])}")
        if not rec["refused"]:
            time.sleep(args.sleep)

    summary = summarize(results)
    print_summary(summary)

    base = in_path.name.replace("_answers.json", "")
    json_path = in_path.with_name(f"{base}_judged.json")
    csv_path = in_path.with_name(f"{base}_judged.csv")
    json_path.write_text(json.dumps({"judge_model": config.JUDGE_MODEL, "summary": summary,
                                     "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["id", "type", "refused", "correctness", "faithfulness", "citation_accuracy",
              "n_claims", "n_unsupported", "judge_reason", "human_correctness", "human_notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r.get(k) for k in fields if k in r},
                             "n_unsupported": len(r["unsupported_claims"]),
                             "human_correctness": "", "human_notes": ""})
    print(f"\nSaved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
