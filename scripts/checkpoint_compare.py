#!/usr/bin/env python3
"""Compare base model vs fine-tuned checkpoint using the Tinker SDK."""

import os
import tinker
from tinker import types

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_checkpoint(sc: tinker.ServiceClient, path: str) -> bool:
    """Check if checkpoint is accessible via REST API."""
    try:
        rc = sc.create_rest_client()
        result = rc.get_checkpoint_archive_url_from_tinker_path(path).result()
        print(f"Checkpoint verified: {result.url[:50]}...")
        return True
    except Exception as e:
        print(f"WARNING: Could not verify checkpoint: {e}")
        return False


def generate(sampler, tokenizer, prompt: str, temperature: float, max_tokens: int) -> str:
    tokens = tokenizer.encode(prompt)
    response = sampler.sample(
        prompt=types.ModelInput.from_ints(tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    ).result()
    seqs = response.sequences
    if len(seqs) != 1:
        raise ValueError(f"Expected 1 sample, got {len(seqs)}")
    return tokenizer.decode(seqs[0].tokens)


def main() -> None:
    # Configuration - update via env vars.
    checkpoint_path = os.environ.get("TINKER_CHECKPOINT", "")
    base_model = os.environ.get("TINKER_BASE_MODEL", "meta-llama/Llama-3.2-1B")
    test_prompts = [
        "Write a haiku about coding:",
        "Write a haiku about the ocean:",
        "Write a short poem about artificial intelligence:",
    ]
    temperature = 0.7
    max_tokens = int(os.environ.get("TINKER_TEST_MAX_TOKENS", "128"))

    sc = tinker.ServiceClient()
    print("\nSetting up samplers...")
    print(f"Base model: {base_model}")
    print(f"Checkpoint: {checkpoint_path}")

    base_sampler = sc.create_sampling_client(base_model=base_model)
    ft_sampler = None

    if checkpoint_path:
        checkpoint_valid = verify_checkpoint(sc, checkpoint_path)
        if checkpoint_valid:
            try:
                ft_sampler = sc.create_sampling_client(model_path=checkpoint_path)
                print("Checkpoint sampler created successfully")
            except Exception as e:
                print(f"\nERROR: Failed to create checkpoint sampler: {e}")
                print("The checkpoint exists but cannot be loaded for sampling.")
                print("This may mean it was saved with save_state() instead of save_weights_for_sampler().")
        else:
            print("\nSkipping checkpoint - verification failed.")
            print("The checkpoint may have been deleted or the path is incorrect.")

    tokenizer = base_sampler.get_tokenizer()

    for prompt in test_prompts:
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        print("\n[BASE MODEL]")
        print("-" * 50)
        try:
            print(generate(base_sampler, tokenizer, prompt, temperature, max_tokens))
        except Exception as e:
            print(f"ERROR: Base model sampling failed: {e}")

        if ft_sampler:
            print("\n[FINE-TUNED]")
            print("-" * 50)
            try:
                print(generate(ft_sampler, tokenizer, prompt, temperature, max_tokens))
            except Exception as e:
                print(f"ERROR: Checkpoint sampling failed: {e}")
        print()

    print("TEST COMPLETE")


if __name__ == "__main__":
    main()
