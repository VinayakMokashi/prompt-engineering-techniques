# Prompt Engineering Techniques

A small, hands-on collection of **prompt-engineering techniques** demonstrated with
[LangChain](https://python.langchain.com/) and [Groq](https://groq.com/)-hosted LLMs
(OpenAI's open-weight `gpt-oss` models). Each script is self-contained and prints its
results, so you can run them and watch how different prompting strategies change the
model's output.

## What's inside

| Script | Technique | What it shows |
| --- | --- | --- |
| [`prompt01.py`](prompt01.py) | **Few-shot & in-context learning** | Guides the model with examples placed in the prompt: few-shot sentiment classification, a flexible multi-task classifier, and learning an arbitrary task (pig latin) purely from runtime examples. |
| [`prompt02.py`](prompt02.py) | **Chain-of-Thought (CoT) reasoning** | Compares a direct answer vs. step-by-step reasoning, a strictly structured step-by-step solver, and a formal logical-deduction prompt on a truth-teller/liar puzzle. |
| [`prompt03.py`](prompt03.py) | **Self-consistency** | Solves each problem several different ways, aggregates the attempts into the most consistent answer, then critiques that answer for logic, facts, and bias. |
| [`prompt04.py`](prompt04.py) | **Prompt evaluation** | Scores and compares competing prompts on three axes — relevance (vs. an expected answer), specificity (lexical richness), and consistency (stability across repeated runs) — using sentence embeddings and cosine similarity. |
| [`prompt05.py`](prompt05.py) | **Prompt structure, system prompts & negative prompting** | Asks one retail-pricing question bare, then with an explicit ROLE / TASK / CONTEXT / OUTPUT FORMAT; then compares a role-only system message with one that adds rules and "Do NOT ..." instructions. Each answer is scored for length, price ranges and questions. |
| [`prompt06.py`](prompt06.py) | **Structured output with Pydantic** | Extracts pricing facts from a messy email as free text and then as a validated Pydantic object, turns them into a typed HOLD / CUT / RAISE decision, and re-checks that decision against the business rules in plain Python before acting on it. |

Together they form a natural progression: **show examples** → **make the model reason out loud** → **answer multiple times and self-verify** → **measure how good a prompt actually is** → **structure the prompt itself** → **get back typed, validated output that code can act on**.

```
prompt-engineering-techniques/
├── prompt01.py … prompt06.py   # one technique per script
├── requirements.txt
├── .env.example                # template for your Groq API key
└── LICENSE
```

## How the newer scripts work

### `prompt05.py` — prompt structure, system prompts and negative prompting

The script asks the same question (*our Coke 500ml is Rs 40, the competitor's is Rs 35: what should our price be?*) four ways:

1. **Bare prompt:** `Coke 500ml, ours Rs 40, competitor Rs 35. New price?`
2. **Structured prompt:** the same request split into `ROLE`, `TASK`, `CONTEXT` (including the cost and a never-below-cost rule) and an exact one-line `OUTPUT FORMAT`.
3. **System prompt, role only:** a `SystemMessage` that just sets the persona.
4. **System prompt with rules:** the persona plus positive rules ("one sentence, a single price") and negative ones ("Do NOT give a price range", "Do NOT add caveats, disclaimers or follow-up questions").

After each answer, `report()` prints its word count, whether it contains a price range (a regex that understands `Rs`/`₹` ranges) and whether it contains a question. From a sample run:

| Prompt | Words | Price range | Contains a question |
| --- | --- | --- | --- |
| Bare prompt | 624 | yes | no |
| Role + task + context + output format | 10 | no | no |
| System prompt: role only | 1422 | no | yes |
| System prompt: role + rules + negative prompting | 7 | no | no |

### `prompt06.py` — structured output with Pydantic

```
messy email ─► PricingFacts (Pydantic) ─► PriceDecision (Pydantic) ─► check_decision() in Python ─► act / reject
```

- `llm.with_structured_output(PricingFacts)` makes the model fill a schema — `sku: str`, `our_price` / `competitor_price` / `cost: float`, `urgency: Literal["low", "medium", "high"]` — so the reply is a validated object, not prose. (In a sample run, the free-text version rendered the email's "35 bucks" as **$35**; the schema version returns plain numbers.)
- A second schema, `PriceDecision`, holds `action: Literal["HOLD", "CUT", "RAISE"]`, a whole-rupee `new_price` and a short reason.
- `check_decision()` re-checks the model's decision against the same business rules given in the prompt — at least 20% above cost and within 5% of the competitor — so the model's arithmetic is verified before anything changes. A sample run:

```
PricingFacts(sku='COKE500', our_price=40.0, competitor_price=35.0, cost=30.0, urgency='high')
PriceDecision(action='CUT', new_price=36, reason='Align with margin and stay within competitor range')
-> rules satisfied; update shelf price for COKE500 to Rs 36
```

## Tech stack

- Python 3.10+
- `langchain-core` + `langchain-groq` 1.x (prompt templates, messages, structured output and the Groq chat model)
- `pydantic` 2 (schemas for structured output in `prompt06.py`)
- `python-dotenv` (loads your API key from a `.env` file)
- `sentence-transformers` + `scikit-learn` + `pandas` / `numpy` (used by `prompt04.py` for embedding-based evaluation)
- Groq-hosted models: `openai/gpt-oss-120b` (`prompt01`, `prompt02`, `prompt05`, `prompt06`) and `openai/gpt-oss-20b` (`prompt03`, `prompt04`)

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
python prompt05.py
python prompt06.py
```

Each prints clearly labelled sections for every technique it demonstrates.

> **Note on `prompt03.py` and `prompt04.py`:** they are API-heavy — each makes
> several model calls per problem/prompt (e.g. repeated runs for the consistency
> score). On Groq's free tier (8,000 tokens per minute per model) they may pause to
> respect rate limits; `max_retries` on the model lets them wait and resume rather
> than crash, so they simply take longer to finish. `prompt04.py` also downloads a
> small embedding model (`all-MiniLM-L6-v2`) the first time it runs.

## Notes on the models

- **Reasoning tokens.** The `gpt-oss` models reason before they answer, and those hidden
  reasoning tokens count towards the completion limit. Groq's default limit is low enough
  to cut long answers off (or leave them empty), so each script sets `max_tokens`
  explicitly, sized so that prompt + `max_tokens` stays within the free tier's budget.
- **Reasoning effort.** `prompt03.py` runs with `reasoning_effort="low"`: self-consistency
  already writes its reasoning paths out in the answers, and at the default effort the
  small model can get stuck generating "unique approaches" to a trivial question without
  ever answering.

## License

Released under the [MIT License](LICENSE).
