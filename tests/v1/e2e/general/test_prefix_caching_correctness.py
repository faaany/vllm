# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

# One small model per cache type the prefix cache handles differently.
MODELS = [
    pytest.param("Qwen/Qwen3-0.6B", id="full_attention"),
    pytest.param("google/gemma-3-1b-it", id="sliding_window"),
    pytest.param("Qwen/Qwen3.5-0.8B", id="gdn_hybrid"),
    pytest.param("tiiuae/Falcon-H1-0.5B-Base", id="mamba2_hybrid"),
]

# (start offset, length) of each shared prefix in the corpus. Different starts
# give different first blocks, so no two prefixes share a cached block.
PREFIXES = [(0, 2048), (1000, 2001)]

QUESTIONS = [
    "\n\nQuestion: In one sentence, what is the text above mainly about?\nAnswer:",
    "\n\nQuestion: Name three people mentioned in the text above.\nAnswer:",
    "\n\nQuestion: Suggest a short title for the text above.\nAnswer:",
]
NUM_WARMUP_QUESTIONS = 2

MAX_TOKENS = 16
NUM_LOGPROBS = 5

CORPUS = Path(__file__).parents[3] / "prompts" / "summary.txt"


def _to_tokens_text_logprobs(output):
    completion = output.outputs[0]
    return list(completion.token_ids), completion.text, completion.logprobs


@pytest.mark.parametrize("model", MODELS)
def test_prefix_cache_hit_matches_uncached(vllm_runner, model: str) -> None:
    with vllm_runner(
        model,
        max_model_len=4096,
        block_size=None,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
    ) as runner:
        llm = runner.get_llm()
        tokenizer = llm.get_tokenizer()
        corpus = tokenizer.encode(CORPUS.read_text(), add_special_tokens=False)
        questions = [tokenizer.encode(q, add_special_tokens=False) for q in QUESTIONS]

        hit_params = SamplingParams(
            temperature=0.0, max_tokens=MAX_TOKENS, logprobs=NUM_LOGPROBS
        )
        ref_params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            logprobs=NUM_LOGPROBS,
            skip_reading_prefix_cache=True,
        )

        hits, refs, failures = [], [], []
        for start, length in PREFIXES:
            assert start + length <= len(corpus), "corpus too short for the prefix"
            prefix = corpus[start : start + length]
            prompts = [TokensPrompt(prompt_token_ids=prefix + q) for q in questions]

            # Sent one at a time: the first fills the cache, the second lets
            # hybrid models checkpoint the junction where the prompts diverge.
            for prompt in prompts[:NUM_WARMUP_QUESTIONS]:
                llm.generate(prompt, hit_params, use_tqdm=False)

            for qi, prompt in enumerate(prompts):
                hit = llm.generate(prompt, hit_params, use_tqdm=False)[0]
                ref = llm.generate(prompt, ref_params, use_tqdm=False)[0]
                label = f"prefix ({start}, {length}) question {qi}"
                if ref.num_cached_tokens:
                    failures.append(f"{label}: reference read the cache")
                # The first two prompts repeat the warm-up verbatim; the last
                # one is new and can only reuse the shared prefix.
                if not hit.num_cached_tokens:
                    failures.append(f"{label}: no prefix-cache hit")
                hits.append(_to_tokens_text_logprobs(hit))
                refs.append(_to_tokens_text_logprobs(ref))

    assert not failures, "\n".join(failures)
    check_logprobs_close(
        outputs_0_lst=refs,
        outputs_1_lst=hits,
        name_0="uncached",
        name_1="cached",
    )
