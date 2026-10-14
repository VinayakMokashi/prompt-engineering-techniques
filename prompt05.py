"""
Prompt structure, system prompts and negative prompting (LangChain + Groq).

A bare one-line question leaves the model to guess the audience, the
constraints and the shape of the answer, so it tends to reply with a long,
generic essay. This script asks one retail-pricing question several ways and
shows how much of that guessing a well-structured prompt removes:

1. bare_vs_structured - the question asked bare, then rewritten with the four
   building blocks of a prompt: ROLE, TASK, CONTEXT and OUTPUT FORMAT.
2. system_prompts     - a system message that sets only a role, then one that
   adds hard rules and negative instructions ("Do NOT ...") to rule out
   behaviour we do not want: price ranges, caveats and follow-up questions.

Every answer is printed with a short report (word count, whether it gives a
price range, whether it contains a question), so the effect of each change
is measurable rather than a matter of impression.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt05.py
"""

import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# temperature=0 -> repeatable output; max_retries lets the client ride out rate limits.
# gpt-oss reasons before it answers and those tokens count towards max_tokens.
# Groq's default cap can cut long answers off; 4096 leaves room for them while
# keeping prompt + max_tokens within the free tier's 8,000 tokens-per-minute budget.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=4096, max_retries=10)

BARE_PROMPT = "Coke 500ml, ours Rs 40, competitor Rs 35. New price?"

# The same request, with every building block spelled out.
STRUCTURED_PROMPT = """ROLE: You are a pricing analyst at an Indian retail chain.
TASK: Recommend a new shelf price for one product.
CONTEXT: Product Coke 500ml. Our price Rs 40. Competitor price Rs 35. Our cost is Rs 30 and we never sell below cost.
OUTPUT FORMAT: One line only, exactly: Price: Rs <number> | Reason: <max 10 words>"""

USER_QUESTION = "Coke 500ml sells at Rs 40 with us and Rs 35 at the competitor. What should we do?"

SYSTEM_ROLE_ONLY = "You are a pricing analyst at an Indian retail chain."

# Role + positive rules + negative rules. Negative instructions name the
# failure modes explicitly instead of hoping the model avoids them.
SYSTEM_WITH_RULES = """You are a pricing analyst at an Indian retail chain.
Answer in one sentence with a single price in Rs.
Do NOT give a price range.
Do NOT mention the competitor by name or suggest checking their website.
Do NOT add caveats, disclaimers or follow-up questions."""

# Matches a price range such as "Rs 36-38", "₹36 – ₹38" or "Rs 36 to Rs 38".
PRICE_RANGE = re.compile(r"(?:Rs\.?|₹)\s*\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*(?:Rs\.?|₹)?\s*\d+")


def report(text):
    """Summarise an answer: length, and whether it breaks the rules we care about."""
    words = len(text.split())
    has_range = "yes" if PRICE_RANGE.search(text) else "no"
    has_question = "yes" if "?" in text else "no"
    return f"{words} words | price range: {has_range} | contains a question: {has_question}"


def show(label, text):
    """Print one labelled answer followed by its report."""
    print(f"--- {label} ---")
    print(text.strip())
    print(f"[{report(text)}]\n")


def bare_vs_structured():
    """Compare a bare question with a role + task + context + output-format prompt."""
    show("bare prompt", llm.invoke(BARE_PROMPT).content)
    show("role + task + context + output format", llm.invoke(STRUCTURED_PROMPT).content)


def system_prompts():
    """Compare a role-only system message with one that adds rules and negative prompts."""
    user_message = HumanMessage(USER_QUESTION)
    show("system prompt: role only",
         llm.invoke([SystemMessage(SYSTEM_ROLE_ONLY), user_message]).content)
    show("system prompt: role + rules + negative prompting",
         llm.invoke([SystemMessage(SYSTEM_WITH_RULES), user_message]).content)


if __name__ == "__main__":
    print("=" * 60)
    print("1. Bare prompt vs structured prompt")
    print("=" * 60 + "\n")
    bare_vs_structured()

    print("=" * 60)
    print("2. System prompts and negative prompting")
    print("=" * 60 + "\n")
    system_prompts()
