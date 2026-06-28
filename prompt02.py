"""
Chain-of-Thought (CoT) reasoning with an LLM (LangChain + Groq).

Asking a model to "think step by step" usually improves accuracy on reasoning
and math problems compared with asking for a direct answer. This script compares
several prompting styles on the same kinds of questions:

1. standard_prompt          - ask for a direct, concise answer.
2. cot_prompt               - ask for a step-by-step answer.
3. advanced_cot_prompt      - enforce a strict per-step structure
                              (state goal -> formula -> calculate -> explain).
4. logical_reasoning_prompt - guide a formal logical deduction (used on a
                              truth-teller / liar puzzle).

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt02.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# temperature=0 -> deterministic output; max_retries lets the client ride out rate limits.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=10)

# Direct-answer prompt (baseline, no reasoning shown).
standard_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question concisely: {question}."
)

# Simple chain-of-thought prompt: ask the model to reason step by step.
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question step by step concisely: {question}"
)

standard_chain = standard_prompt | llm
cot_chain = cot_prompt | llm

question = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"

# Advanced CoT: force an explicit, uniform structure for every step.
advanced_cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Solve the following problem step by step. For each step:
1. State what you're going to calculate
2. Write the formula you'll use (if applicable)
3. Perform the calculation
4. Explain the result

Question: {question}

Solution:"""
)

advanced_cot_chain = advanced_cot_prompt | llm

# A multi-leg problem where step-by-step reasoning really helps.
complex_question = "A car travels 150 km at 60 km/h, then another 100 km at 50 km/h. What is the average speed for the entire journey?"

advanced_cot_response = advanced_cot_chain.invoke(complex_question).content

# Logical-reasoning prompt: a detailed template that walks the model through a
# formal deduction (list facts -> enumerate scenarios -> eliminate contradictions).
logical_reasoning_prompt = PromptTemplate(
    input_variables=["scenario"],
    template="""Analyze the following logical puzzle thoroughly. Follow these steps in your analysis:

List the Facts:
Summarize all the given information and statements clearly.
Identify all the characters or elements involved.

Identify Possible Roles or Conditions:
Determine all possible roles, behaviors, or states applicable to the characters or elements (e.g., truth-teller, liar, alternator).

Note the Constraints:
Outline any rules, constraints, or relationships specified in the puzzle.

Generate Possible Scenarios:
Systematically consider all possible combinations of roles or conditions for the characters or elements.
Ensure that all permutations are accounted for.

Test Each Scenario:
For each possible scenario:
Assume the roles or conditions you've assigned.
Analyze each statement based on these assumptions.
Check for consistency or contradictions within the scenario.

Eliminate Inconsistent Scenarios:
Discard any scenarios that lead to contradictions or violate the constraints.
Keep track of the reasoning for eliminating each scenario.

Conclude the Solution:
Identify the scenario(s) that remain consistent after testing.
Summarize the findings.

Provide a Clear Answer:
State definitively the role or condition of each character or element.
Explain why this is the only possible solution based on your analysis.

Scenario:

{scenario}

Analysis:"""
)

logical_reasoning_chain = logical_reasoning_prompt | llm

# Run each chain so the differences between the prompting styles are visible.
standard_response = standard_chain.invoke(question).content
cot_response = cot_chain.invoke(question).content

puzzle = "Alice, Bob, and Carol each either always tell the truth or always lie. Alice says, 'Bob is a liar.' Bob says, 'Carol is a liar.' Carol says, 'Alice and Bob are both liars.' Who is a truth-teller and who is a liar?"
logical_reasoning_response = logical_reasoning_chain.invoke(puzzle).content

print("Standard Answer:\n", standard_response)
print("\nChain-of-Thought Answer:\n", cot_response)
print("\nAdvanced Chain-of-Thought Answer:\n", advanced_cot_response)
print("\nLogical Reasoning Answer:\n", logical_reasoning_response)
