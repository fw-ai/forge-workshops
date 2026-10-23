# Decision router workshop

Train a small model to decide, for every user prompt, whether a cheap model can answer it or whether it needs an
expensive one. Then put that router in front of a LangChain agent.

**Slides:** [Workshop: Decision Model](https://docs.google.com/presentation/d/1miFMQbDvXsZDzbak-FX3tfExoWnHH_3rjSrzIQ2MG5I/edit?usp=sharing)

---

## Run the workshop

| File | What it does | Time |
|---|---|---|
| [`workshop_train_router.ipynb`](workshop_train_router.ipynb) | Builds the data, trains one router (Qwen 3.8, 250 rows) with the cookbook's decision loop, deploys it, scores it, plots separation, score histograms and calibration | ~15-20 min |
| [`workshop_langchain_router.ipynb`](workshop_langchain_router.ipynb) | Loads that router, picks a threshold, runs 8 math questions through a LangChain agent that routes between `gpt-oss-120b` and `kimi-k3`, then deletes the deployment | ~5-10 min |
| [`requirements.txt`](requirements.txt) | The Fireworks training cookbook at a pinned commit (brings `fireworks-ai`, `transformers`, `datasets`, PyTorch and the decision trainer), plus LangChain | |
| [`.env.example`](.env.example) | Copy to `.env` and add `FIREWORKS_API_KEY`. That is the only credential. | |

```bash
cd decision-router
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # do this before the session: the first install downloads PyTorch
cp .env.example .env                # add FIREWORKS_API_KEY
jupyter lab
```

Run `workshop_train_router.ipynb` top to bottom. Its last cell prints a per-stage timing table (data, train, deploy,
score) so you can see where the minutes went. The deployment it creates is left running for the LangChain notebook,
which deletes it at the end. If you only run the training notebook, set `KEEP_FOR_LANGCHAIN = False` in its last cell
and run it, so the deployment stops billing.

**Expected time per stage (training notebook):**

| Stage | Typical | Notes |
|---|---|---|
| Install | 1-5 min, once | Do it before the session |
| Data | ~0.5-2 min | Downloads three public datasets from Hugging Face on first run (cached afterwards) |
| Train | ~3-6 min | 24 optimizer steps on the serverless pool. When the pool is busy this has taken up to ~18 min |
| Deploy | ~4 min | Matched deployment shape |
| Score | ~1 min | 360 requests, 8 at a time |

**What it costs:** serverless LoRA training on 250 short rows, about 15 minutes of a single-GPU deployment, and a few
cents of serverless inference in the LangChain notebook (questions sent to `kimi-k3` cost a few cents each).

**Verified:** the install, the setup cell (including looking up the account from the API key alone) and
the data cells were run in a fresh virtual environment. They rebuild exactly the same 250 training rows and 200
temperature-fit rows as the research run that produced the story's numbers. Training, deployment and scoring were not
run as part of that check.

## Files

| File | Purpose |
|---|---|
| `workshop_train_router.ipynb` | data, decision-loop training, deploy, score, results; saves `runs/workshop-<id>/workshop_router.json` |
| `workshop_langchain_router.ipynb` | the trained router as LangChain middleware, live agent run, teardown |
| `requirements.txt` | pinned cookbook commit plus LangChain |
| `runs/` | `replay_answers.jsonl` (committed cache of both models' answers); per run: training rows, recipe logs, charts, `workshop_router.json` (not committed) |
