# Forge workshops

Hands-on workshops for training and using models with Fireworks. Each workshop lives in its own folder with its own
notebooks, requirements and README, and needs only a Fireworks API key.

| Workshop | What you build | Time | Slides |
|---|---|---|---|
| [`research-agent-rl/`](research-agent-rl/) | Train a multi-hop research agent (Kimi K3) with reinforcement learning, then run it as a LangChain agent | ~20 min train, ~20-30 min LangChain (includes deploying the trained model) | [Research Agent](https://docs.google.com/presentation/d/1G18C-oDN9lJv9nqzXl10IvNl_NbXOieBGKQ5BONzCHs/edit?usp=sharing) |
| [`decision-router/`](decision-router/) | Train a model router (Qwen 3.8) with the cookbook's decision loop, then route a LangChain agent between a cheap and a strong model | ~15-20 min train, ~5-10 min LangChain | [Decision Model](https://docs.google.com/presentation/d/1miFMQbDvXsZDzbak-FX3tfExoWnHH_3rjSrzIQ2MG5I/edit?usp=sharing) |
| [`skill-to-weights/`](skill-to-weights/) | Mini workshop: train a `SKILL.md` into GLM 5.3 Flash's weights with one short SFT run, scored by a plain Python function | ~15 min | - |

All workshops use the Fireworks Training SDK (`fireworks-ai` 1.2.27) and install the
[Fireworks training cookbook](https://github.com/fw-ai/cookbook) as a pip dependency, so there is nothing to clone.

## Getting started

Use one virtual environment per workshop: their pinned package versions differ.

```bash
cd decision-router            # or: cd research-agent-rl, cd skill-to-weights
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # do this before the session; the first install downloads PyTorch
cp .env.example .env              # set FIREWORKS_API_KEY
jupyter lab
```

Then open the workshop's `workshop_train_*.ipynb` (or `workshop_skill_to_weights.ipynb`) and run it top to bottom. Each workshop's README lists its notebooks,
expected time per stage, files, and anything account-specific.

## Layout

```
forge-workshops/
  README.md                 this index
  .gitignore                .env, venvs, run outputs, downloaded data
  research-agent-rl/
    README.md  requirements.txt  .env.example
    workshop_train_research_agent.ipynb  workshop_langchain_research_agent.ipynb
    research_env.py  research_rollout.py
    data/                   MuSiQue downloads here on first run (not committed)
    runs/                   closed-book caches (committed); per-run logs (not committed)
  decision-router/
    README.md  requirements.txt  .env.example
    workshop_train_router.ipynb  workshop_langchain_router.ipynb
    runs/                   cached replay answers (committed); per-run rows, logs, charts (not committed)
  skill-to-weights/
    README.md  requirements.txt  .env.example
    workshop_skill_to_weights.ipynb
    skills/caveman/         vendored skill + license
    cases/caveman.jsonl     40 cases with grading labels
    runs/                   per-run replies and result.png (not committed)
```
