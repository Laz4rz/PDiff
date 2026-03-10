from sample_model import (
    ROOT,
    load_model_from_run,
    print_run_metadata,
    print_samples,
    run_sampling,
)

from discrete_diffusion.data.datasets import get_star_graph_dataset

NUM_SAMPLES = 24
NUM_STEPS = None  # use model default
RUN_DIR = ROOT / "outputs/star_graph/2026.03.06/213118"
SHOW_STEPS = True
STEP_EVERY = 64
STEP_MAX_SAMPLES = 24
SKIP_SPECIAL_TOKENS = False
EPS = None  # set to a float to override (e.g., 1e-5)


def _resolve_star_dataset_config(ckpt_cfg) -> dict:
    data_cfg = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else ckpt_cfg.data
    raw_cfg = (
        data_cfg.get("dataset_config", {})
        if isinstance(data_cfg, dict)
        else getattr(data_cfg, "dataset_config", {})
    )
    if raw_cfg is None:
        return {}
    try:
        cfg = dict(raw_cfg)
    except TypeError:
        return {}
    # Leave only explicit overrides; the loader applies defaults for missing keys.
    return {k: v for k, v in cfg.items() if v not in (None, "")}


model, tokenizer, ckpt_path, ckpt_cfg, _ = load_model_from_run(RUN_DIR)
print_run_metadata(ckpt_path, ckpt_cfg)

star_dataset_config = _resolve_star_dataset_config(ckpt_cfg)
print("Star dataset config:", star_dataset_config if star_dataset_config else "<defaults>")
validation = get_star_graph_dataset(dataset_config=star_dataset_config)["validation"]
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
    expected_completions=completions,
)
