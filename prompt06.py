"""
Structured output with Pydantic (LangChain + Groq).

Free-text answers suit people but not programs: to use a price inside code you
would have to parse prose and hope its format never changes.
`with_structured_output` instead asks the model to fill in a Pydantic schema,
so the reply comes back as a validated Python object with typed fields.

The script follows one messy store email through a small pipeline:

1. free_text_facts - ask for the pricing facts and get prose back.
2. extract_facts   - the same request with a PricingFacts schema: float prices
                     and a Literal urgency level that code can use directly.
3. decide_price    - a second schema, PriceDecision, turns the facts into an
                     action (HOLD / CUT / RAISE) and a whole-rupee price.
4. check_decision  - because the decision is data, plain Python re-checks it
                     against the business rules before anything is changed,
                     instead of trusting the model's arithmetic.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt06.py
"""

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# temperature=0 -> repeatable output; max_retries lets the client ride out rate limits.
# gpt-oss reasons before it answers and those tokens count towards max_tokens.
# Groq's default cap can cut long answers off; 4096 leaves room for them while
# keeping prompt + max_tokens within the free tier's 8,000 tokens-per-minute budget.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=4096, max_retries=10)

EMAIL = """Hi team, quick one. Store 12 called - the Coke 500ml (SKU COKE500) is not moving,
the shop across the road is at 35 bucks and we are still at 40. Finance says our landed
cost is 30. Can we do something before the weekend? Thanks, Priya"""

# Business rules, stated once and used both in the prompt and in the check.
MIN_MARKUP_ON_COST = 0.20     # price must be at least 20% above cost
MAX_GAP_TO_COMPETITOR = 0.05  # price must be within 5% of the competitor
RULES = (
    f"margin at least {MIN_MARKUP_ON_COST:.0%} on cost, "
    f"within {MAX_GAP_TO_COMPETITOR:.0%} of the competitor price, whole rupees only"
)


class PricingFacts(BaseModel):
    """Facts pulled out of the email."""
    sku: str
    our_price: float
    competitor_price: float
    cost: float
    urgency: Literal["low", "medium", "high"] = Field(description="how urgent the sender sounds")


class PriceDecision(BaseModel):
    """What to do with the shelf price."""
    action: Literal["HOLD", "CUT", "RAISE"]
    new_price: int = Field(description="new shelf price in whole rupees")
    reason: str = Field(description="max 10 words")


def free_text_facts(email):
    """Ask for the facts without a schema: the answer is prose that code cannot use."""
    return llm.invoke("What are the pricing facts in this email?\n" + email).content


def extract_facts(email):
    """Ask for the same facts as a validated PricingFacts object."""
    return llm.with_structured_output(PricingFacts).invoke("Extract the pricing facts.\n" + email)


def decide_price(facts):
    """Turn the facts into a typed PriceDecision."""
    return llm.with_structured_output(PriceDecision).invoke(
        f"You are a pricing analyst. Rules: {RULES}. Facts: {facts.model_dump()}"
    )


def check_decision(decision, facts):
    """Re-check the model's decision against the business rules in plain Python.

    Returns a list of rule violations; an empty list means the decision is safe to apply.
    """
    problems = []
    floor = facts.cost * (1 + MIN_MARKUP_ON_COST)
    if decision.new_price < floor:
        problems.append(f"Rs {decision.new_price} is below the margin floor of Rs {floor:.2f}")
    gap = abs(decision.new_price - facts.competitor_price) / facts.competitor_price
    if gap > MAX_GAP_TO_COMPETITOR:
        problems.append(
            f"Rs {decision.new_price} is {gap:.1%} away from the competitor's Rs {facts.competitor_price:.0f}"
        )
    return problems


if __name__ == "__main__":
    print("--- 1. free text: readable, but code cannot use it ---")
    print(free_text_facts(EMAIL).strip())

    print("\n--- 2. structured output: a validated Pydantic object ---")
    facts = extract_facts(EMAIL)
    print(repr(facts))
    # Typed fields mean ordinary arithmetic works on the model's output.
    print(f"type={type(facts).__name__} | price gap = Rs {facts.our_price - facts.competitor_price:.0f}")

    print("\n--- 3. structured output as the decision ---")
    decision = decide_price(facts)
    print(repr(decision))

    print("\n--- 4. code checks the decision before acting on it ---")
    problems = check_decision(decision, facts)
    if problems:
        print("REJECTED: " + "; ".join(problems))
        print(f"-> keep {facts.sku} at Rs {facts.our_price:.0f} and send for review")
    elif decision.action == "HOLD":
        print(f"-> rules satisfied; keep {facts.sku} at Rs {facts.our_price:.0f}")
    else:
        print(f"-> rules satisfied; update shelf price for {facts.sku} to Rs {decision.new_price}")
