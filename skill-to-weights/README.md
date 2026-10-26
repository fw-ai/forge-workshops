# Skill to weights mini workshop

Train a skill file (`SKILL.md`) into a model's weights with one short SFT run on Fireworks serverless training.
Afterwards the model behaves as if the skill were in its prompt, even though it isn't.

**The recipe.** Sample the base model on train cases with the skill as its system prompt. Score every reply with a plain
Python function and keep the ones that pass. Fine-tune on those replies with the skill removed from the prompt. Then
compare three things on held-out cases: base without the skill, base with it, and the tuned model without it.

**The worked example** is [caveman](skills/caveman/SKILL.md) (`JuliusBrussee/caveman`, Apache-2.0): terse developer
answers with no ceremony and every command kept exact. It runs on 40 developer questions, 24 train and 16 held out, on
GLM 5.3 Flash.

## Notebook

| Notebook | What it does | Time |
|---|---|---|
| `workshop_skill_to_weights.ipynb` | headroom with and without the skill -> training rows -> 48-step LoRA SFT -> tuned eval -> pass@1 / pass@k / pass^k chart -> promotes the tuned checkpoint | ~15 min |

**What to expect.** On the reference run (16 held-out cases x 4 draws), held-out pass@1 was 9% for base, 62% with the
skill in the prompt, and 67% for the tuned model with no skill. Your numbers will move a little from run to run.

The grader checks the rules a script can check:
- the reply stays under a word limit;
- the required command or term is present;
- there is no ceremony or filler;
- it doesn't fake the caveman voice;
- dangerous commands carry a warning.

It doesn't check whether the explanation is technically right. On the earlier run it agreed with an LLM judge on 92-95% of
replies.

## Use your own skill

Edit only the notebook's first cell:

- `SKILL_PATH`: your `SKILL.md`.
- `CASES_PATH`: a JSONL file with `id`, `split` (`train` or `test`), `category`, `prompt`, and `labels`. The labels are
  whatever your grader needs; the model never sees them.
- `NEUTRAL_SYSTEM`: the system prompt without the skill. The tuned model is trained and scored under it.
- `score(case, reply) -> bool`: your quick eval. Plain Python is fastest. To use an LLM judge, call it inside `score`.
- `BASE_MODEL` and `TOKENIZER`: the model must be on serverless training and serverless inference, and `TOKENIZER` must be
  its chat template. The notebook stops if they disagree.

If the headroom step shows the skill doesn't beat base, stop there: training has nothing to copy.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set FIREWORKS_API_KEY
jupyter lab
```

- The run makes about 320 base generations, a short LoRA session, and 64 tuned generations. No judge is called.
- Each run writes its replies and `result.png` to `runs/<run id>/`, which is not committed.
- Promotion turns the checkpoint into a model in your account. The last cell closes the training session; run it.

## Files

| File | Purpose |
|---|---|
| `workshop_skill_to_weights.ipynb` | the whole workshop |
| `skills/caveman/` | the vendored skill, its license, and the pinned source commit |
| `cases/caveman.jsonl` | 40 cases with grading labels (`key_point`, `must_include`, `warning`, `max_words`) |
