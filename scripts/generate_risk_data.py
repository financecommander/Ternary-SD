"""
Generate synthetic credit risk training data using Gemini API.

Creates labeled (Borrower_History_Notes -> Risk_Flag + Audit_Memo) examples
for fine-tuning a local ternary LLM to replace cloud API risk analysis.

Usage:
    python scripts/generate_risk_data.py --count 1000 --output data/risk_training.jsonl

Requires GEMINI_API_KEY in .env or environment.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Synthetic borrower profile components ───────────────────────────

POSITIVE_TRAITS = [
    "excellent track record with multiple successful projects",
    "consistent on-time payments across all prior loans",
    "strong net worth with diversified real estate portfolio",
    "proven experience in commercial real estate development",
    "reliable borrower with 15+ years industry experience",
    "timely repayment history on prior $5M+ facilities",
    "experienced developer with institutional-quality reporting",
    "successful exits on 3 prior multifamily developments",
    "strong sponsor with $50M+ AUM across CRE portfolio",
    "good credit history and stable cash flow from operations",
]

NEUTRAL_TRAITS = [
    "standard borrower with limited operating history",
    "first-time commercial borrower with residential experience",
    "moderate experience in the asset class",
    "borrower has completed one prior project of similar scope",
    "adequate financial statements provided",
    "regional operator expanding into new market",
    "borrower has mixed reviews from prior lenders",
    "limited third-party reporting available",
    "sponsor has adequate but not exceptional liquidity",
    "operating history is under 5 years in CRE",
]

NEGATIVE_TRAITS = [
    "history of late payments on prior facilities",
    "prior default on $2M construction loan in 2019",
    "bankruptcy filing in 2017, since discharged",
    "foreclosure on prior investment property",
    "delinquent on multiple obligations in past 3 years",
    "poor credit score below 620",
    "failed to complete prior development project",
    "litigation pending related to prior loan default",
    "borrower has significant tax liens outstanding",
    "concern about borrower's ability to fund cost overruns",
]

PROJECT_TYPES = [
    "Multifamily", "Office", "Retail", "Industrial",
    "Mixed-Use", "Hotel", "Senior Housing", "Student Housing",
    "Self-Storage", "Medical Office",
]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "San Francisco", "Austin", "Denver", "Miami", "Seattle",
    "Nashville", "Charlotte", "Dallas", "Atlanta", "Portland",
    "Rural Kentucky", "Small town Iowa", "Suburban Ohio",
]


def build_borrower_notes(risk_level: str) -> str:
    """Generate realistic borrower history notes for a given risk level."""
    if risk_level == "Low":
        traits = random.sample(POSITIVE_TRAITS, k=random.randint(2, 4))
        # Occasionally add a minor neutral note
        if random.random() < 0.3:
            traits.append(random.choice(NEUTRAL_TRAITS))
    elif risk_level == "Medium":
        pos = random.sample(POSITIVE_TRAITS, k=random.randint(1, 2))
        neu = random.sample(NEUTRAL_TRAITS, k=random.randint(1, 2))
        neg = random.sample(NEGATIVE_TRAITS, k=random.randint(0, 1))
        traits = pos + neu + neg
        random.shuffle(traits)
    else:  # High
        neg = random.sample(NEGATIVE_TRAITS, k=random.randint(2, 4))
        # Occasionally add a positive note (mixed signals)
        if random.random() < 0.3:
            neg.append(random.choice(NEUTRAL_TRAITS))
        traits = neg

    project = random.choice(PROJECT_TYPES)
    location = random.choice(LOCATIONS)
    amount = random.choice(["$500K", "$1.2M", "$3.5M", "$7M", "$15M", "$25M"])

    notes = (
        f"Borrower seeking {amount} for {project} project in {location}. "
        + ". ".join(t.capitalize() for t in traits) + "."
    )
    return notes


def generate_with_gemini(notes: str, api_key: str) -> dict | None:
    """Call Gemini to produce a labeled risk assessment for the given notes."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai package required. pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    prompt = (
        "You are an independent credit risk auditor. Review these borrower notes. "
        "Summarize the borrower's 'Execution Risk' in one sentence and assign a "
        "'Risk Flag' (Low, Medium, High). "
        'Return response in JSON format with keys: "Risk_Flag" and "Audit_Memo".\n\n'
        f"Borrower Notes:\n{notes}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()

        # Strip markdown code fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        if "Risk_Flag" in result and "Audit_Memo" in result:
            return result
    except Exception as e:
        print(f"  Gemini error: {e}")
    return None


def generate_rule_based(notes: str, intended_risk: str) -> dict:
    """Fallback: generate a deterministic label without API call."""
    lower = notes.lower()

    neg_count = sum(1 for kw in [
        "default", "late", "poor", "failed", "bankruptcy",
        "foreclosure", "delinquent", "lien", "litigation", "concern",
    ] if kw in lower)

    pos_count = sum(1 for kw in [
        "excellent", "strong", "proven", "reliable", "consistent",
        "timely", "successful", "experienced", "good",
    ] if kw in lower)

    if neg_count >= 2:
        flag = "High"
        memo = "Multiple risk indicators including adverse credit events."
    elif neg_count == 1 and pos_count <= 1:
        flag = "Medium"
        memo = "Mixed borrower profile with some risk factors noted."
    elif pos_count >= 2:
        flag = "Low"
        memo = "Strong borrower profile with solid track record."
    else:
        flag = intended_risk
        memo = {
            "Low": "Adequate borrower profile with no major risk indicators.",
            "Medium": "Moderate risk profile; additional due diligence recommended.",
            "High": "Elevated risk profile; significant concerns identified.",
        }[intended_risk]

    return {"Risk_Flag": flag, "Audit_Memo": memo}


def main():
    parser = argparse.ArgumentParser(description="Generate credit risk training data")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", type=str, default="data/risk_training.jsonl")
    parser.add_argument("--use-gemini", action="store_true",
                        help="Use Gemini API for labels (slower, costs ~$1-2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY", "")

    if args.use_gemini and not api_key:
        print("ERROR: Set GEMINI_API_KEY to use --use-gemini")
        sys.exit(1)

    # Distribution: 40% Low, 35% Medium, 25% High
    risk_distribution = (
        ["Low"] * int(args.count * 0.40)
        + ["Medium"] * int(args.count * 0.35)
        + ["High"] * (args.count - int(args.count * 0.40) - int(args.count * 0.35))
    )
    random.shuffle(risk_distribution)

    examples = []
    gemini_calls = 0

    print(f"Generating {args.count} training examples...")
    print(f"  Mode: {'Gemini API' if args.use_gemini else 'Rule-based (fast)'}")
    print(f"  Output: {args.output}")

    for i, intended_risk in enumerate(risk_distribution):
        notes = build_borrower_notes(intended_risk)

        if args.use_gemini:
            label = generate_with_gemini(notes, api_key)
            gemini_calls += 1
            if label is None:
                label = generate_rule_based(notes, intended_risk)
            # Rate limit
            if gemini_calls % 10 == 0:
                time.sleep(1)
        else:
            label = generate_rule_based(notes, intended_risk)

        example = {
            "input": notes,
            "output": {
                "Risk_Flag": label["Risk_Flag"],
                "Audit_Memo": label["Audit_Memo"],
            },
        }
        examples.append(example)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.count} generated")

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    flags = [ex["output"]["Risk_Flag"] for ex in examples]
    print(f"\nDone! Wrote {len(examples)} examples to {args.output}")
    print(f"  Low:    {flags.count('Low'):4d} ({flags.count('Low')/len(flags)*100:.0f}%)")
    print(f"  Medium: {flags.count('Medium'):4d} ({flags.count('Medium')/len(flags)*100:.0f}%)")
    print(f"  High:   {flags.count('High'):4d} ({flags.count('High')/len(flags)*100:.0f}%)")

    if args.use_gemini:
        print(f"  Gemini API calls: {gemini_calls}")


if __name__ == "__main__":
    main()
