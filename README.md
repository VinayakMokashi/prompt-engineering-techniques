# Prompt Engineering Techniques

A small, hands-on collection of **prompt-engineering techniques** demonstrated with
[LangChain](https://python.langchain.com/) and [Groq](https://groq.com/)-hosted
LLaMA models. Each script is self-contained and prints its results, so you can run
them and watch how different prompting strategies change the model's output.

## What's inside

| Script | Technique | What it shows |
| --- | --- | --- |
| [`prompt01.py`](prompt01.py) | **Few-shot & in-context learning** | Guides the model with examples placed in the prompt: few-shot sentiment classification, a flexible multi-task classifier, and learning an arbitrary task (pig latin) purely from runtime examples. |
| [`prompt02.py`](prompt02.py) | **Chain-of-Thought (CoT) reasoning** | Compares a direct answer vs. step-by-step reasoning, a strictly structured step-by-step solver, and a formal logical-deduction prompt on a truth-teller/liar puzzle. |
| [`prompt03.py`](prompt03.py) | **Self-consistency** | Solves each problem several different ways, aggregates the attempts into the most consistent answer, then critiques that answer for logic, facts, and bias. |
| [`prompt04.py`](prompt04.py) | **Prompt evaluation** | Scores and compares competing prompts on three axes — relevance (vs. an expected answer), specificity (lexical richness), and consistency (stability across repeated runs) — using sentence embeddings and cosine similarity. |

Together they form a natural progression: **show examples** → **make the model reason out loud** → **answer multiple times and self-verify** → **measure how good a prompt actually is**.

## Tech stack

- Python 3.8+
- `langchain` + `langchain-groq` (LLM orchestration)
- `python-dotenv` (loads your API key from a `.env` file)
- `sentence-transformers` + `scikit-learn` + `pandas` / `numpy` (used by `prompt04.py` for embedding-based evaluation)
- Groq-hosted models: `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/prompt-engineering-techniques.git
   cd prompt-engineering-techniques
   ```

2. **(Recommended) create a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key.** Copy `.env.example` to `.env` and paste in your key
   (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   The scripts load `.env` from their own folder, so the key is picked up no matter
   which directory you launch them from. `.env` is git-ignored and never committed.

## How to run

Run any script directly:

```bash
python prompt01.py
python prompt02.py
python prompt03.py
python prompt04.py
```

Each prints clearly labelled sections for every technique it demonstrates.

> **Note on `prompt03.py` and `prompt04.py`:** they are API-heavy — each makes
> several model calls per problem/prompt (e.g. repeated runs for the consistency
> score). On Groq's free tier (6,000 tokens/minute) they may pause to respect rate
> limits; `max_retries` on the model lets them wait and resume rather than crash, so
> they simply take longer to finish. `prompt04.py` also downloads a small embedding
> model (`all-MiniLM-L6-v2`) the first time it runs.

## License

Released under the [MIT License](LICENSE).
