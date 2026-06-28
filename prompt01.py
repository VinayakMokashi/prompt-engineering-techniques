"""
Few-shot and in-context learning with an LLM (LangChain + Groq).

This script demonstrates three related prompt-engineering techniques where the
model is guided by examples placed directly in the prompt:

1. few_shot_sentiment_classification - classify sentiment by first showing the
   model a few labelled examples.
2. multi_task_few_shot               - reuse the few-shot idea for different
   tasks (sentiment, language detection, ...) chosen at call time.
3. in_context_learning               - teach the model an arbitrary task purely
   from examples provided at runtime (here: converting words to pig latin).

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt01.py
"""

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


def few_shot_sentiment_classification(input_text):
    """Classify input_text as Positive, Negative, or Neutral using few-shot examples."""
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text"],
        template="""
        Classify the sentiment as Positive, Negative, or Neutral.

        Examples:
        Text: I love this product! It's amazing.
        Sentiment: Positive

        Text: This movie was terrible. I hated it.
        Sentiment: Negative

        Text: The weather today is okay.
        Sentiment: Neutral

        Now, classify the following:
        Text: {input_text}
        Sentiment:
        """
    )

    # "prompt | llm" builds a runnable chain: fill the template, then send it to the model.
    chain = few_shot_prompt | llm
    result = chain.invoke(input_text).content

    # The model may answer "Sentiment: Positive"; keep only the label after the colon.
    result = result.strip()
    if ':' in result:
        result = result.split(':')[1].strip()

    return result


def multi_task_few_shot(input_text, task):
    """Run a chosen task (e.g. 'sentiment' or 'language') on input_text via few-shot examples."""
    few_shot_prompt = PromptTemplate(
        input_variables=["input_text", "task"],
        template="""
        Perform the specified task on the given text.

        Examples:
        Text: I love this product! It's amazing.
        Task: sentiment
        Result: Positive

        Text: Bonjour, comment allez-vous?
        Task: language
        Result: French

        Now, perform the following task:
        Text: {input_text}
        Task: {task}
        Result:
        """
    )

    chain = few_shot_prompt | llm
    return chain.invoke({"input_text": input_text, "task": task}).content


def in_context_learning(task_description, examples, input_text):
    """Teach an arbitrary task from runtime examples, then apply it to input_text.

    examples is a list of {"input": ..., "output": ...} dicts that are formatted
    into the prompt so the model can infer the pattern.
    """
    # Turn the example dicts into a readable "Input/Output" block for the prompt.
    example_text = "".join([f"Input: {e['input']}\nOutput: {e['output']}\n\n" for e in examples])

    in_context_prompt = PromptTemplate(
        input_variables=["task_description", "examples", "input_text"],
        template="""
        Task: {task_description}

        Examples:
        {examples}

        Now, perform the task on the following input:
        Input: {input_text}
        Output:
        """
    )

    chain = in_context_prompt | llm
    return chain.invoke({"task_description": task_description, "examples": example_text, "input_text": input_text}).content


# --- Example usage: teach pig latin from two examples, then convert a new word ---
task_desc = "Convert the given text to pig latin."

examples = [
    {"input": "hello", "output": "ellohay"},
    {"input": "apple", "output": "appleay"}
]

test_input = "python"

result = in_context_learning(task_desc, examples, test_input)
print(result)

sentiment = few_shot_sentiment_classification("I really enjoyed the concert last night!")
print("Sentiment:", sentiment)

language = multi_task_few_shot("Hola, ¿cómo estás?", task="language")
print("Language:", language)
