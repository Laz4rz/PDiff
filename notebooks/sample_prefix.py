from sample_model import (
    ROOT,
    load_model_from_run,
    print_run_metadata,
    print_samples,
    run_sampling,
)

from discrete_diffusion.data.datasets import generate_prefix_dataset

NUM_SAMPLES = 24
NUM_STEPS = None  # use model default
RUN_DIR = ROOT / "outputs/prefix/2026.03.05/144345"
SHOW_STEPS = True
STEP_EVERY = 64
STEP_MAX_SAMPLES = 24
SKIP_SPECIAL_TOKENS = False
EPS = None  # set to a float to override (e.g., 1e-5)


model, tokenizer, ckpt_path, ckpt_cfg, _ = load_model_from_run(RUN_DIR)
print_run_metadata(ckpt_path, ckpt_cfg)

validation = generate_prefix_dataset(samples=None)["validation"]
prefixes, completions = list(validation["prefixes"]), list(validation["completions"])

samples, prefix_prompt_texts, prefix_token_ids = run_sampling(
    model,
    tokenizer,
    num_samples=NUM_SAMPLES,
    num_steps=NUM_STEPS,
    eps=EPS,
    step_every=STEP_EVERY,
    max_samples=STEP_MAX_SAMPLES,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
    show_steps=SHOW_STEPS,
    prefix_prompts=prefixes
)

print_samples(
    samples,
    tokenizer,
    prefix_prompt_texts=prefix_prompt_texts,
    prefix_token_ids=prefix_token_ids,
)
