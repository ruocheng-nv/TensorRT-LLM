# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-100 GSM8K harness shared by the HF reference and the TensorRT-LLM path.

The two paths must be scored by *one* implementation of sample selection,
prompt rendering, stop behaviour and answer extraction, or a config difference
shows up as a model difference. Everything either side needs lives here; the
runners only supply generation.

Sample selection is the first 100 rows of the GSM8K ``test`` split in dataset
order. Fixed by construction rather than by a seed: a seeded shuffle is
reproducible only as long as nobody changes the sampler, and the scores have to
stay comparable across iterations.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

NUM_SAMPLES = 100
#: Long enough that a full chain of thought is not truncated. Truncation
#: silently depresses the score and reads as a model defect.
MAX_SEQ_LEN = 2048
#: Large enough to keep the run fast, small enough not to change the score.
BATCH_SIZE = 8
#: The template's own knob: ``reasoning_effort`` in {'low', 'high'}, anything
#: else (including unset) resolves to 'max'. Whichever value is used must be
#: identical on both paths, which is why it is threaded explicitly rather than
#: left to the template default.
DEFAULT_REASONING_EFFORT = "max"

#: The label is the text after the final '#### ' marker.
_LABEL_RE = re.compile(r"####\s*([-0-9.,]+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
#: This model marks its final answer with ``\boxed{...}``, and then very often
#: keeps writing -- a restatement ("for the 16 glasses"), a units note, or a
#: verification line ("Check: 63 + 99 = 162"). Reading the *last* number in the
#: completion therefore scores the check, not the answer: it turned 5 of the
#: first 6 apparent failures into false negatives on the reference run. The
#: marker is read first for that reason.
#: One level of nesting is required, not optional: the model writes
#: ``\boxed{260 \text{ sheep}}``, and a brace-free pattern silently fails to
#: match it and falls through to the bold rule, which then reads the last
#: "**Step 3:**" heading as the answer.
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
#: Fallback marker for completions that bold the answer instead of boxing it.
_BOLD_RE = re.compile(r"\*\*([^*]*?)\*\*")


def load_samples(num_samples: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """The fixed evaluation slice: question, gold label, and stable index."""
    import datasets

    split = datasets.load_dataset("gsm8k", "main", split="test")
    rows = []
    for index in range(num_samples):
        item = split[index]
        match = _LABEL_RE.search(item["answer"])
        if match is None:
            raise ValueError(f"gsm8k row {index} has no '#### <answer>' marker")
        rows.append(
            {
                "index": index,
                "question": item["question"],
                "label": normalize_number(match.group(1)),
                "gold_solution": item["answer"],
            }
        )
    return rows


def select(samples: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    """The named subset, in dataset order.

    The harness's own accuracy-debugging method works on the handful of samples
    that actually drive a score gap rather than on the whole slice, so a run has
    to be able to name them. Indices are the dataset's, so a subset result is
    directly comparable with the same rows of a full run.
    """
    wanted = set(int(i) for i in indices)
    chosen = [s for s in samples if s["index"] in wanted]
    missing = wanted - {s["index"] for s in chosen}
    if missing:
        raise ValueError(f"gsm8k indices {sorted(missing)} are outside the loaded slice")
    return chosen


def normalize_number(text: str) -> Optional[str]:
    """Canonical form so '1,000', '1000' and '1000.00' compare equal."""
    cleaned = text.replace(",", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value == int(value):
        return str(int(value))
    return repr(value)


#: LaTeX the model writes *inside* a numeric answer. ``{,}`` is the LaTeX
#: thousands separator: ``\boxed{\$25{,}000}`` reads as the three separate
#: numbers 25, 000 with a brace between them unless it is normalized first.
_LATEX_UNIT_RE = re.compile(r"\\(?:text|mathrm|mbox)\{[^{}]*\}")


def _strip_latex(span: str) -> str:
    span = _LATEX_UNIT_RE.sub(" ", span)
    return span.replace("{,}", ",").replace("\\,", "").replace("\\$", "$").replace("\\%", "%")


def extract_answer(completion: str) -> Optional[str]:
    """The model's numeric answer, read from its own answer marker.

    Order matters and is not arbitrary: the model states its result and then
    frequently continues with a restatement or a verification arithmetic line,
    so the *last number* is often not the answer. Prefer the explicit
    ``\\boxed{}`` marker, then a bolded span, and only fall back to the last
    number when the completion carries no marker at all.

    Both paths call this, so the rule cannot open a gap between them -- but it
    can make the whole gate meaningless if it is wrong, which is why the
    fallbacks are ordered rather than merged.
    """
    for pattern in (_BOXED_RE, _BOLD_RE):
        spans = pattern.findall(completion)
        for span in reversed(spans):
            numbers = _NUMBER_RE.findall(_strip_latex(span))
            if numbers:
                return normalize_number(numbers[-1])
    matches = _NUMBER_RE.findall(completion)
    if not matches:
        return None
    return normalize_number(matches[-1])


def render(
    tokenizer: Any,
    question: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> str:
    """Render one question through the checkpoint's own chat template.

    Both paths call this, so a template or role difference cannot open a gap
    between them. The template always appends ``<think>`` after
    ``<|assistant|>``, so thinking mode is on for every run; ``reasoning_effort``
    is the only knob and it is reported with every result.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )


def rescore(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-derive ``predicted`` from the stored completion text.

    The runs persist completions, not just verdicts, so scoring stays a
    property of the harness rather than of whichever revision produced the
    artefact. Both sides are re-extracted by the same code at comparison time,
    so an extractor fix cannot land on one path and not the other -- which is
    exactly how a harness change would otherwise manufacture a score gap.
    """
    return [{**row, "predicted": extract_answer(row["completion"])} for row in rows]


def score(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exact-match score plus the per-sample record needed to debug a gap."""
    correct = [r for r in rows if r.get("predicted") is not None and r["predicted"] == r["label"]]
    truncated = [r for r in rows if r.get("truncated")]
    return {
        "num_samples": len(rows),
        "num_correct": len(correct),
        "score": 100.0 * len(correct) / len(rows) if rows else 0.0,
        "num_truncated": len(truncated),
        "truncated_indices": [r["index"] for r in truncated],
        "wrong_indices": [
            r["index"] for r in rows if r.get("predicted") is None or r["predicted"] != r["label"]
        ],
    }


def compare_runs(reference: Sequence[Dict[str, Any]], candidate: Sequence[Dict[str, Any]]):
    """Which samples the reference gets right and the candidate gets wrong.

    Those are the only samples that drive a score gap; ones both get right or
    both get wrong carry no signal, and naming them is what makes a follow-up
    teacher-forced comparison cheap instead of a whole-dataset rerun.
    """
    by_index = {r["index"]: r for r in candidate}
    discriminating = []
    for row in reference:
        other = by_index.get(row["index"])
        if other is None:
            continue
        ref_ok = row.get("predicted") == row["label"]
        cand_ok = other.get("predicted") == other["label"]
        if ref_ok and not cand_ok:
            discriminating.append(
                {
                    "index": row["index"],
                    "label": row["label"],
                    "reference_predicted": row.get("predicted"),
                    "candidate_predicted": other.get("predicted"),
                    "candidate_truncated": bool(other.get("truncated")),
                }
            )
    return discriminating


# ---------------------------------------------------------------------------
# Teacher-forced divergence classification
# ---------------------------------------------------------------------------
#
# When a teacher-forced step's greedy argmax disagrees with the reference, the
# question is whether the reference itself scored the two candidates as tied (so
# the choice carries no meaning) or made a resolvable decision this path flipped
# (which is what a real defect looks like).
#
# The earlier predicate was TAUTOLOGICAL and is deleted: it called a divergence a
# tie iff ``ref[hf]-ref[trt] <= |got[hf]-ref[hf]| + |got[trt]-ref[trt]|``. Writing
# the left side out,
#
#     ref[hf]-ref[trt] = (ref[hf]-got[hf]) + (got[hf]-got[trt]) + (got[trt]-ref[trt])
#
# and ``got[trt] >= got[hf]`` for any argmax inversion, so the middle term is
# <= 0 and the inequality holds by construction. Every inversion was labelled a
# tie, so "0 confident divergences" was non-evidence.
#
# The replacement calibrates noise on tokens OTHER than the inverted pair and
# adds a reference-only bf16-ULP measure that does not use this path's logits at
# all. See ``test_classify_divergence_is_not_tautological``.


def bf16_ulp(value: float) -> float:
    """The bf16 spacing at ``value``'s magnitude (7 fractional mantissa bits).

    The reference model's logits are bf16-valued (kept as float32), so a
    separation of one ULP means the two candidates are *adjacent representable
    values* in the reference's own readout -- a reference-only tie needing no
    assumption about this path's error.
    """
    if value == 0 or not math.isfinite(value):
        return 2.0**-133
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def classify_divergence(got: Any, ref: Any) -> Optional[Dict[str, Any]]:
    """Classify one teacher-forced step from the two full logit vectors.

    ``got`` and ``ref`` are this path's and the reference's logits over the whole
    vocabulary at one step. Returns ``None`` when the greedy argmax agrees (not a
    divergence), otherwise a record whose ``confident`` field is the honest,
    non-tautological verdict.

    ``confident`` is ``True`` when the reference's own separation between its
    top-2 candidates exceeds BOTH (a) the ambient this-path-vs-reference logit
    error measured on OTHER competitive tokens at the same step, scaled to a
    pairwise gap, and (b) one bf16 ULP. That is: the reference had a resolvable
    preference larger than this path's typical noise elsewhere, and this path
    flipped it anyway. A bit-identical or <=1-ULP separation is never confident:
    the reference itself cannot resolve those.
    """
    import torch

    got = got.float().flatten()
    ref = ref.float().flatten()
    hf_tok = int(ref.argmax())
    trt_tok = int(got.argmax())
    if hf_tok == trt_tok:
        return None

    hf_sep = float(ref[hf_tok] - ref[trt_tok])  # >= 0 by construction
    ulp = bf16_ulp(max(abs(float(ref[hf_tok])), abs(float(ref[trt_tok]))))
    hf_sep_ulps = hf_sep / ulp if ulp else float("inf")
    bit_identical = bool(ref[hf_tok] == ref[trt_tok])

    # Independent noise: the error on the top competitive tokens by reference
    # logit, EXCLUDING the two candidates. Those are where argmax decisions
    # happen, so their spread is the relevant "could generic noise have flipped
    # this" scale -- and it does not look at the inverted pair, which is what
    # made the old test vacuous.
    err = got - ref
    order = torch.argsort(ref, descending=True)
    competitive = [int(t) for t in order[:66].tolist() if t not in (hf_tok, trt_tok)][:64]
    noise_competitive = float(err[torch.tensor(competitive)].std())
    # The separation is a pairwise gap; a typical gap between two independent
    # competitive tokens has std ~ sqrt(2) x the per-logit std.
    noise_pairwise = math.sqrt(2.0) * noise_competitive

    confident = bool(hf_sep > noise_pairwise and hf_sep_ulps > 1.0 and not bit_identical)
    return {
        "hf_token": hf_tok,
        "trt_token": trt_tok,
        "hf_separation": hf_sep,
        "hf_sep_bf16_ulps": hf_sep_ulps,
        "hf_bit_identical": bit_identical,
        "noise_pairwise": noise_pairwise,
        "sep_over_noise": hf_sep / noise_pairwise if noise_pairwise else float("inf"),
        "confident": confident,
    }
