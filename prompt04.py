"""
Prompt evaluation with an LLM + sentence embeddings (LangChain + Groq).

Where prompt01-03 focus on *writing* better prompts, this script focuses on
*measuring* prompt quality. The PromptEval class scores LLM responses on three
axes and uses them to compare competing prompts:

- relevance   - cosine similarity between the response embedding and an expected
                answer embedding (is it answering the right thing?).
- specificity - ratio of unique words to total words (rich wording vs. vague/repetitive).
- consistency - average pairwise similarity across repeated runs of the same prompt
                (does the model answer stably?).

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
   (this script also needs sentence-transformers / scikit-learn, which download a
   small embedding model, 'all-MiniLM-L6-v2', on first run).
3. Run:                   python prompt04.py

Note: comparing the prompts makes several model calls per prompt (one answer plus
repeated runs for the consistency score), so on Groq's free tier it may pause to
respect rate limits.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")


class PromptEval:
    def __init__(self):
        # temperature=0 -> deterministic output; max_retries lets the client ride out rate limits.
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_retries=10)

        # Compact, fast model that turns sentences into comparable embedding vectors.
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_specificity(self, response):
        """
        Measures how specific (lexically rich) a response is.
        Higher ratio of unique words = more specific.
        """
        words = response.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def compute_relevance(self, response, expected_content):
        """
        Uses cosine similarity to evaluate how semantically close
        the model's response is to the expected answer.
        """
        embeddings = self.sentence_model.encode([response, expected_content])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    def compute_consistency(self, prompt, num_responses=3):
        """
        Checks how consistent the model is when asked the same prompt multiple times.
        High similarity between responses = high consistency.
        """
        similarities = []
        # Ask the same prompt several times, then compare every pair of answers.
        responses = [self.llm.invoke(prompt).content for _ in range(num_responses)]
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self.compute_relevance(responses[i], responses[j])
                similarities.append(sim)
        return np.mean(similarities)

    def compare_prompts(self, prompts, expected_content):
        """
        Compares a list of prompts by evaluating:
        - Relevance: how close the response is to the expected answer
        - Specificity: how unique and non-generic the wording is
        - Consistency: how stable the model is when asked repeatedly
        Returns results as a pandas DataFrame.
        """
        results = []
        for prompt in prompts:
            response = self.llm.invoke(prompt).content
            relevance = self.compute_relevance(response, expected_content)
            specificity = self.compute_specificity(response)
            consistency = self.compute_consistency(prompt)
            results.append(
                {
                    "relevance": relevance,
                    "specificity": specificity,
                    "consistency": consistency,
                }
            )
        return pd.DataFrame.from_dict(results)


if __name__ == "__main__":
    evaluator = PromptEval()

    # Two passages with very different lexical richness, used to sanity-check
    # the specificity metric: the repetitive speech should score lower.
    joey_speech = """It's a love based on giving and receiving, as well as having and sharing. And the love that they give and have is shared and received. And through this having and giving and sharing and receiving, we too can share and love and have... and receive.
    When we share the giving and experience the receiving, we find ourselves having what was received and receiving what was shared. For in sharing what we have and having what we share, the giving becomes receiving and the receiving becomes giving.
    The beauty of their bond comes from receiving what is given and giving what is had. As they share in having and have in sharing, we witness how receiving the shared and sharing the received creates a circle of having what is given and giving what is had.
    Life together means giving to sharing and sharing in receiving. When they receive what is shared and share what is received, they discover having given and giving had. Through receiving the having and having the receiving, sharing gives and giving shares.
    Their commitment shows how having received and receiving had leads to sharing given and giving shared. As they give to having and have in giving, the sharing receives while the receiving shares. This exchange of having shared and sharing had completes the giving received and receiving given
    """

    quantum_mechanics = """Here's a concise speech on quantum mechanics without much repetition:
    Quantum mechanics reveals a universe fundamentally different from our everyday experience. At the subatomic level, particles exhibit wave-like properties, existing in multiple states simultaneously until measured. This principle, known as superposition, challenges our classical intuition.
    Heisenberg's uncertainty principle demonstrates we cannot precisely know both position and momentum of a particle. Quantum entanglement allows particles to maintain instantaneous connections regardless of distance, what Einstein called "spooky action at a distance."
    Quantum tunneling enables particles to pass through seemingly impenetrable barriers. Wave function collapse occurs when observation forces quantum systems to resolve into definite states.
    These counterintuitive phenomena form the backbone of modern technology - from lasers to MRI machines to transistors in computing devices. While mathematically robust, quantum theory continues to spark philosophical debates about reality's true nature and the role of consciousness in the universe.
    Understanding quantum mechanics requires abandoning classical determinism for probabilistic thinking, reminding us that the cosmos operates by rules far stranger and more beautiful than we initially imagined.
    """

    joey_specificity = evaluator.compute_specificity(joey_speech)
    quantum_specificity = evaluator.compute_specificity(quantum_mechanics)
    print("Joey Specificity:", joey_specificity)
    print("Quantum Specificity:", quantum_specificity)

    # Compare four differently-worded prompts that target the same underlying answer.
    prompts = [
        "List the types of machine learning.",
        "What are the main categories of machine learning algorithms?",
        "Explain the different approaches to machine learning.",
        "What types of machine learning can you think about? Explain like you are Shakespear, Darwin, or Einstein. Choose any of these personality"
    ]

    # The gold-standard answer that responses are measured against for relevance.
    expected_content = "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."

    compare_prompt_results = evaluator.compare_prompts(prompts, expected_content)
    print(compare_prompt_results)
