"""Token-level research-agent rollouts for GRPO on the serverless training sampler.

The inference path in the headroom notebook gets text back. Training needs the sampled
token ids and their logprobs for every assistant turn, so rollouts run through the
``glm53`` renderer and a Tinker sampler instead. Same tools, same prompt, same score.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from fireworks.training import sdk as tinker
from fireworks.training.sdk import types as fw_types

from research_env import SYSTEM_PROMPT, TOOLS, Episode, Score, score
from training.renderer.glm5 import GLM53Renderer

TOOL_SPECS = [t["function"] for t in TOOLS]
LAST_TURN_NUDGE = "This is your last turn. Call answer(text) now with your best short answer."
LAST_TURN_NUDGE_LOOSE = "This is your last turn. Call answer(text) now."
TEXT_REPLY_NUDGE = "Use the tools. Call answer(text) when you know the answer."


class GLM53LowEffortRenderer(GLM53Renderer):
    """``glm53`` with the system line the chat template emits for ``reasoning_effort="low"``.

    The template defaults to Max; training at Max would optimize a slower policy than the
    low-effort one measured at inference.
    """

    _initial_prompt_text = "<|system|>Reasoning Effort: Low"


@dataclass
class Rollout:
    episode: Episode
    turns: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def sampler_failed(self) -> bool:
        return self.stop_reason.startswith(("sampler_", "missing_logprobs"))

    def score(self) -> Score:
        return score(self.episode)


def _sample(sampler: Any, prompt: tinker.ModelInput, params: tinker.SamplingParams, timeout_s: float):
    """One sample with a hard wall-clock cap; a stuck request must not hold a slot for minutes."""
    # fireworks-ai >= 1.2.27 accepts only its own ModelInput class, not tinker's.
    sdk_prompt = fw_types.ModelInput.from_ints(prompt.to_ints())
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: sampler.sample(prompt=sdk_prompt, num_samples=1, sampling_params=params).result())
        return future.result(timeout=timeout_s)


def run_rollout(
    sampler: Any,
    renderer: GLM53LowEffortRenderer,
    question: dict,
    *,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    sample_timeout_s: float,
    system_prompt: str = SYSTEM_PROMPT,
    tool_specs: list[dict] | None = None,
    last_turn_nudge: str | None = LAST_TURN_NUDGE,
    nudge_text_replies: bool = True,
) -> Rollout:
    """Run one multi-turn episode. Never raises: failures are recorded in ``stop_reason``.

    ``last_turn_nudge=None`` sends no last-turn message. ``nudge_text_replies=False`` ends the
    episode unanswered (scored 0) when the model replies without calling a tool."""
    rollout = Rollout(episode=Episode(question))
    messages = renderer.create_conversation_prefix_with_tools(tool_specs or TOOL_SPECS, system_prompt)
    messages.append({"role": "user", "content": question["question"]})
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                   stop=renderer.get_stop_sequences())

    for turn in range(max_turns):
        if turn == max_turns - 1 and last_turn_nudge:
            messages.append({"role": "user", "content": last_turn_nudge})
        prompt = renderer.build_generation_prompt(messages)
        try:
            result = _sample(sampler, prompt, params, sample_timeout_s)
        except FutureTimeout:
            rollout.stop_reason = "sampler_timeout"
            return rollout
        except Exception as exc:  # noqa: BLE001 - sampler trouble says nothing about the policy
            rollout.stop_reason = f"sampler_error: {type(exc).__name__}"
            return rollout

        seq = result.sequences[0]
        tokens = list(seq.tokens or [])
        if not tokens or seq.logprobs is None or len(seq.logprobs) != len(tokens):
            rollout.stop_reason = "missing_logprobs"
            return rollout
        rollout.turns.append({"prompt": prompt, "tokens": tokens, "logprobs": [float(x) for x in seq.logprobs]})

        message, _ = renderer.parse_response(tokens)
        messages.append(message)
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            if not nudge_text_replies:
                rollout.stop_reason = "text_reply"
                return rollout
            messages.append({"role": "user", "content": TEXT_REPLY_NUDGE})
            continue
        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            observation = rollout.episode.call(name, arguments)
            messages.append({"role": "tool", "content": observation,
                             "tool_call_id": getattr(call, "id", None) or "", "name": name})
            if rollout.episode.done:
                break
        if rollout.episode.done:
            rollout.stop_reason = "answered"
            return rollout
    rollout.stop_reason = "max_turns"
    return rollout


def rollout_to_datums(rollout: Rollout, advantage: float, max_seq_len: int) -> list[fw_types.Datum]:
    """One importance-sampling datum per assistant turn, all sharing the episode's advantage."""
    datums = []
    for turn in rollout.turns:
        prompt, tokens, logprobs = turn["prompt"], turn["tokens"], turn["logprobs"]
        if len(tokens) < 2:
            continue
        model_input = fw_types.ModelInput.from_ints(prompt.to_ints() + tokens[:-1])
        if model_input.length > max_seq_len:
            continue
        pad = prompt.length - 1   # targets are shifted one position left of the inputs
        datums.append(fw_types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": [0] * pad + tokens,
                "logprobs": [0.0] * pad + logprobs,
                "advantages": [0.0] * pad + [advantage] * len(tokens),
            },
        ))
    return datums
