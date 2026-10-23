# Research agent RL workshop

Train Kimi K3 to research multi-hop questions with reinforcement learning on Fireworks serverless training, then use
the same agent from LangChain.

**Slides:** [Workshop: Research Agent](https://docs.google.com/presentation/d/1G18C-oDN9lJv9nqzXl10IvNl_NbXOieBGKQ5BONzCHs/edit?usp=sharing)

**The task.** Each MuSiQue question needs two facts chained together. The agent gets the question's 20 Wikipedia
paragraphs (2 answer it, 18 are look-alike distractors) and three tools: `search(query)`, `open(doc_id)`,
`answer(text)`. The tools run as plain Python on your machine (`research_env.py`); only the model calls go to Fireworks.

**The reward** (`correctness x efficiency`): full credit for a short answer containing the gold answer, given after
reading at least one document, in about 5 tool calls. Graded by plain string matching against the dataset's answers.

## Notebooks

| Notebook | What it does | Time |
|---|---|---|
| `workshop_train_research_agent.ipynb` | base test on 13 leak-free held-out questions -> 4 RL steps -> tuned test -> results and reward charts -> promotes the trained checkpoint to a model | ~20 min |
| `workshop_langchain_research_agent.ipynb` | the same agent as a LangChain `create_agent` with the same prompt, tools, turn budget and scoring; deploys the promoted model, runs base and tuned side by side, then deletes the deployment | ~20-30 min |

Four steps demonstrates the training loop; it is not enough to move accuracy reliably. A longer 18-step run of the same
recipe went from 16% to about 50% short, correct answers on the leak-free questions (see the slides).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set FIREWORKS_API_KEY
jupyter lab
```

- The MuSiQue dev set (~29 MB) downloads from Hugging Face on first run into `data/`.
- `runs/closedbook_*.json` caches which questions Kimi K3 answers from memory; with the cache present the data step is
  instant (without it, ~6 minutes).
- The training notebook writes the promoted model id to `runs/promoted_model.txt`; the LangChain notebook reads it.
- The LangChain notebook's deployment (8x B300) bills until its last cell deletes it; run that cell.

## Files

| File | Purpose |
|---|---|
| `research_env.py` | episode state, tools, BM25 search, grader, rewards, leakage check |
| `research_rollout.py` | multi-turn rollouts on the training sampler and conversion to training data |
| `runs/` | closed-book caches (committed); each run also writes per-episode logs and `promoted_model.txt` here (not committed) |
