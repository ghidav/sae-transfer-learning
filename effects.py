import argparse
import gc
import math
import os
import logging

import torch as t
from tqdm import tqdm
from functools import partial

from loading_utils import load_examples, load_saes
from hooks import sae_hook, sae_ig_patching_hook

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def patching_effect_attrib(clean, patch, model, modules, dictionaries, metric_fn, metric_kwargs=dict()):
    hidden_states_clean = {}
    grads = {}

    sae_hooks = []

    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks.append((module, partial(sae_hook, sae=dictionary, cache=hidden_states_clean)))

    # Forward pass with hooks
    logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits, **metric_kwargs)

    # Backward pass
    metric_clean.sum().backward()

    # Collect gradients
    for module in modules:
        if module in hidden_states_clean:
            grads[module] = hidden_states_clean[module].grad

    if patch is None:
        hidden_states_patch = {k: t.zeros_like(v) for k, v in hidden_states_clean.items()}
    else:
        hidden_states_patch = {}
        sae_hooks = []
        for i, module in enumerate(modules):
            dictionary = dictionaries[module]
            sae_hooks.append((module, partial(sae_hook, sae=dictionary, cache=hidden_states_patch)))

        with t.no_grad():
            corr_logits = model.run_with_hooks(patch, fwd_hooks=sae_hooks)

    effects = {}
    deltas = {}
    for module in modules:
        patch_state, clean_state, grad = hidden_states_patch[module], hidden_states_clean[module], grads[module]
        delta = patch_state - clean_state.detach()
        effect = delta * grad
        effects[module] = effect
        deltas[module] = delta
        grads[module] = grad

    del hidden_states_clean, hidden_states_patch
    gc.collect()

    return effects


def patching_effect_ig(clean, patch, model, modules, dictionaries, metric_fn, steps=10, metric_kwargs=dict()):

    hidden_states_clean = {}
    sae_hooks = []

    # Forward pass through the clean input with hooks to capture hidden states
    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks.append((module, partial(sae_hook, sae=dictionary, cache=hidden_states_clean)))

    # First pass to get clean logits and metric
    logits_clean = model.run_with_hooks(clean, fwd_hooks=sae_hooks)

    hidden_states_patch = {}
    sae_hooks_patch = []
    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks_patch.append((module, partial(sae_hook, sae=dictionary, cache=hidden_states_patch)))

    with t.no_grad():
        logits_patch = model.run_with_hooks(patch, fwd_hooks=sae_hooks_patch)

    # Integrated gradients computation
    grads = {}
    effects = {}
    deltas = {}

    for module in modules:
        dictionary = dictionaries[module]
        clean_state = hidden_states_clean[module].detach()
        patch_state = hidden_states_patch[module].detach() if patch is not None else None
        delta = (patch_state - clean_state.detach()) if patch_state is not None else -clean_state.detach()

        for step in range(steps + 1):
            interpolated_state_cache = {}
            alpha = step / steps
            interpolated_state = (
                clean_state * (1 - alpha) + patch_state * alpha if patch is not None else clean_state * (1 - alpha)
            )

            interpolated_state.requires_grad_(True)
            interpolated_state.retain_grad()

            sae_hook_ = [
                (
                    module,
                    partial(
                        sae_ig_patching_hook, sae=dictionary, patch=interpolated_state, cache=interpolated_state_cache
                    ),
                )
            ]

            # Forward pass with hooks
            logits_interpolated = model.run_with_hooks(clean, fwd_hooks=sae_hook_)
            metric = metric_fn(logits_interpolated, **metric_kwargs)

            # Sum the metrics and backpropagate
            metric.sum().backward(retain_graph=True)

            if module not in grads:
                grads[module] = interpolated_state_cache[module].grad.clone()
            else:
                grads[module] += interpolated_state_cache[module].grad

            if step % (steps // 5) == 0:  # Print every 20% of steps
                del interpolated_state_cache
                t.cuda.empty_cache()

            model.zero_grad(set_to_none=True)

        # Calculate gradients
        grads[module] /= steps

        # Compute effects
        effect = grads[module] * delta
        effects[module] = effect
        deltas[module] = delta

    del hidden_states_clean, hidden_states_patch
    gc.collect()

    return effects


def metric_fn(logits):
    return t.gather(logits[:, -1, :], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - t.gather(
        logits[:, -1, :], dim=-1, index=clean_answer_idxs.view(-1, 1)
    ).squeeze(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", "-k", type=int, default=-1, help="The number of cluster to be used.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ioi",
        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.",
    )
    parser.add_argument(
        "--component", "-c", type=str, default="resid_post", help="The component to test for downstream effects."
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=192,
        help="The number of examples from the --dataset over which to average indirect effects.",
    )
    parser.add_argument(
        "--example_length",
        "-l",
        type=int,
        default=None,
        help="The max length (if using sum aggregation) or exact length (if not aggregating) of examples.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="pythia-160m-deduped",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--dict_path", type=str, default="saes", help="Path to all dictionaries for your language model."
    )
    parser.add_argument("--dict_size", type=int, default=32768, help="The width of the dictionary encoder.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of examples to process at once when running circuit discovery.",
    )
    parser.add_argument(
        "--method", "-mt", type=str, default="attrib", help="Method to use to compute effects ('attrib' or 'ig')."
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--direction", type=str, default="baseline")
    parser.add_argument("--ckpt", type=str, default="1B")

    args = parser.parse_args()
    dict_path = os.path.join(args.dict_path, args.model)

    effects_dir = "effects"
    os.makedirs(effects_dir, exist_ok=True)

    device = args.device
    patching_effect = patching_effect_attrib if args.method == "attrib" else patching_effect_ig

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.tokenizer.padding_side = "left"

    nl = model.cfg.n_layers
    nh = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    modules = [get_act_name(args.component, args.layer)]
    dictionaries = None
    cluster = args.K != -1

    # loading saes
    dictionaries = load_saes(
        dict_path,
        args.model,
        model.cfg,
        modules,
        layer=args.layer,
        ckpt=args.ckpt,
        device=device,
        debug=True,
        direction=args.direction,
    )
    logger.info(f"{len(dictionaries)} dictionaries loaded.")

    data_path = f"tasks/{args.dataset}.json"
    save_basename = args.dataset

    examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    print(f"Loaded {len(examples)} examples from dataset {args.dataset}.")

    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [examples[batch * batch_size : (batch + 1) * batch_size] for batch in range(n_batches)]

    if num_examples < args.num_examples:  # warn the user
        logger.warning(
            f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead."
        )
    print("Collecting effects for", args.component, "on", args.dataset, "with", args.method, "method.")
    print("A total of", num_examples, "examples will be used.")

    running_nodes = None
    for batch in tqdm(batches, desc="Batches", total=n_batches):
        clean_inputs = t.cat([e["clean_prefix"] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor([e["clean_answer"] for e in batch], dtype=t.long, device=device)

        if not args.example_length:
            args.example_length = clean_inputs.shape[1]

        patch_inputs = t.cat([e["patch_prefix"] for e in batch], dim=0).to(device)
        patch_answer_idxs = t.tensor([e["patch_answer"] for e in batch], dtype=t.long, device=device)
        effects = patching_effect(clean_inputs, patch_inputs, model, modules, dictionaries, metric_fn)

        nodes = {}
        for module in modules:
            nodes[module] = effects[module]
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}

        if running_nodes is None:
            running_nodes = {k: len(batch) * nodes[k].to("cpu") for k in nodes.keys() if k != "y"}
        else:
            for k in nodes.keys():
                if k != "y":
                    running_nodes[k] += len(batch) * nodes[k].to("cpu")

        del nodes
        gc.collect()

    nodes = {k: v.to(device) / num_examples for k, v in running_nodes.items()}
    save_dict = {"examples": examples, "nodes": nodes}
    save_path = f"{effects_dir}/{save_basename}_n{num_examples}_{args.method}_{args.direction}_L{args.layer}_{args.ckpt}.pt"
    with open(save_path, "wb") as outfile:
        t.save(save_dict, outfile)
