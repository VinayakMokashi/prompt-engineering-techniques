"""
Self-consistency with an LLM (LangChain + Groq).

Instead of trusting a single answer, this technique asks the model to solve the
same problem several different ways, then has the model cross-check those
attempts against each other. This often catches mistakes that one isolated
answer would miss.

Pipeline (see solve_problem):
1. generate_multiple_paths - solve the problem N times, each "using a unique
   approach" (temperature=0.3 introduces variety between attempts).
2. aggregate_results       - feed all attempts back and pick the most consistent answer.
3. self_consistency_check  - critique that answer for logic, facts, and bias.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt03.py

Note: this script is API-heavy. Each problem triggers ~5 model calls
(N reasoning paths + aggregation + consistency check), and it is run on several
problems, so on Groq's free tier it may pause to respect rate limits. max_retries
on the model lets it wait and resume rather than crash.
"""

import os
import sys
from pathlib import Path
import random
from collections import Counter

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# temperature=0.3 -> slight randomness so the reasoning paths actually differ;
# max_retries lets the client wait out the free-tier rate limit instead of failing.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_retries=10)


def generate_multiple_paths(problem, num_paths=3):
    """
    Generate multiple reasoning paths for a given problem using slight randomness.
    Encourages diverse approaches to problem solving.

    Args:
    problem (str): The problem statement.
    num_paths (int): Number of reasoning paths to generate.

    Returns:
    list: A list of generated reasoning paths.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "path_number"],
        template="""Solve the following problem using a unique approach. This is reasoning path {path_number}.
        Problem: {problem}
        Reasoning path {path_number}:"""
    )

    # Ask the model the same question num_paths times to collect diverse attempts.
    paths = []
    for i in range(num_paths):
        chain = prompt_template | llm
        response = chain.invoke({"problem": problem, "path_number": i+1}).content
        paths.append(response)

    return paths


problem = "A ball is thrown upwards with an initial velocity of 20 m/s. How high will it go?"

paths = generate_multiple_paths(problem)

# Show each individual reasoning path before they are aggregated.
print("Problem:\n", problem)
print("\nGenerated Reasoning Paths:")
for i, path in enumerate(paths, 1):
    print(f"\n--- Reasoning Path {i} ---\n", path)
print("\n" + "=" * 50 + "\n")


def aggregate_results(paths):
    """
    Analyze multiple reasoning paths and return the most consistent answer.

    Args:
    paths (list): List of reasoning paths.

    Returns:
    str: The most consistent answer determined via aggregation.
    """
    prompt_template = PromptTemplate(
        input_variables=["paths"],
        template="""Analyze the following reasoning paths and determine the most consistent answer. If there are discrepancies, explain why and provide the most likely correct answer.
        Reasoning paths:
        {paths}

        Most consistent answer:"""
    )

    # Join the separate attempts into one block for the model to compare.
    chain = prompt_template | llm
    response = chain.invoke({"paths": "\n".join(paths)}).content
    return response


aggregated_result = aggregate_results(paths)
print("Aggregated Result:\n", aggregated_result)


def self_consistency_check(problem, aggregated_result):
    """
    Evaluate the consistency and reliability of the final result.

    Args:
    problem (str): The original problem statement.
    aggregated_result (str): The combined answer.

    Returns:
    str: Evaluation report on the consistency and logic.
    """
    prompt_template = PromptTemplate(
        input_variables=["problem", "result"],
        template="""Evaluate the consistency and reliability of the following result for the given problem.
        Problem: {problem}
        Result: {result}

        Evaluation (consider factors like logical consistency, adherence to known facts, and potential biases):"""
    )

    chain = prompt_template | llm
    response = chain.invoke({"problem": problem, "result": aggregated_result}).content
    return response


consistency_evaluation = self_consistency_check(problem, aggregated_result)
print("Self-Consistency Evaluation:\n", consistency_evaluation)


def solve_problem(problem):
    """
    Solve a problem using multiple reasoning paths, aggregation, and self-consistency check.

    Args:
    problem (str): The problem statement.

    Returns:
    tuple: (aggregated_result, consistency_evaluation)
    """
    paths = generate_multiple_paths(problem)
    aggregated_result = aggregate_results(paths)
    consistency_evaluation = self_consistency_check(problem, aggregated_result)
    return aggregated_result, consistency_evaluation


# Run the full self-consistency pipeline over a few example problems.
problems = [
    "What is the capital of France?",
    "Explain the concept of supply and demand in economics.",
    "If a train travels at 60 km/h, how long will it take to cover 180 km?"
]

for problem in problems:
    print(f"Problem: {problem}")
    result, evaluation = solve_problem(problem)
    print("Aggregated Result:\n", result)
    print("\nConsistency Evaluation:\n", evaluation)
    print("\n" + "-"*50 + "\n")
