import torch
from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformer
import argparse
from loading_utils import load_saes, load_examples
from transformer_lens.utils import get_act_name

from functools import partial
import numpy as np
from hooks import sae_features_hook, sae_hook
import os

parser = argparse.ArgumentParser()
parser.add_argument("-K", "--K", nargs="+", default=[123, 246, 368, 492])
parser.add_argument("-c", "--component", type=str, default="resid_post")
parser.add_argument("-n", "--n", type=int, default=1024)
parser.add_argument("-m", "--method", type=str, default="attrib")
parser.add_argument("-w", "--what", type=str, default="faithfulness")
parser.add_argument("-d", "--direction", type=str, default="baseline")
parser.add_argument("--layer", type=int, default=10)
parser.add_argument("--task", type=str, default="ioi")
parser.add_argument("--ckpt", type=str, default="1B")
args = parser.parse_args()

os.makedirs("faithfulness", exist_ok=True)


def test_circuit(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    saes,
    node_threshold=0,
    use_resid=False,
    what="faithfulness",
    device="cuda",
):

    hooks = []
    masks = []
    K = node_threshold

    for hook_name in nodes.keys():

        # feature_mask = (nodes[hook_name].abs() > node_threshold).sum(0) > 0
        _, topk_idxes = torch.topk(nodes[hook_name], K, dim=1)
        feature_mask = torch.zeros_like(nodes[hook_name], dtype=torch.bool)
        feature_mask.scatter_(1, topk_idxes, 1)

        hooks.append(
            (
                hook_name,
                partial(
                    sae_features_hook,
                    sae=saes[hook_name],
                    feature_mask=feature_mask,
                    feature_avg=feature_avg[hook_name],
                    resid=use_resid,
                    ablation=what,
                ),
            )
        )

        masks.append(feature_mask.type(torch.int32))

    # masks = [(m > 0).sum().item() for m in masks]
    masks = [K for m in masks]

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens.to(device),
            fwd_hooks=hooks,
        ).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))

    return (clean_ans_logits - patch_ans_logits).squeeze(), np.mean(masks)


def faithfulness(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    dictionaries,
    node_threshold,
    use_resid=False,
    device="cuda",
):

    # Get the model's logit diff - m(M)
    with torch.no_grad():
        logits = model(tokens.to(device)).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))

    M = (clean_ans_logits - patch_ans_logits).squeeze().mean().item()

    # Get the circuit's logit diff - m(C)
    C, N = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        node_threshold=node_threshold,
        use_resid=use_resid,
        what=args.what,
        device=device,
    )

    # Get the ablated circuit's logit diff - m(zero)
    zero, _ = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        node_threshold=node_threshold,
        use_resid=use_resid,
        what="empty",
        device=device,
    )

    return (C.mean().item() - zero.mean().item()) / (M - zero.mean().item() + 1e-7), N


##########
## Main ##
##########

print(args)
task = args.task
args.K = [int(k) for k in args.K]
model = HookedTransformer.from_pretrained("pythia-160m-deduped", device="cuda")
modules = [get_act_name(args.component, args.layer)]
lengths = {"ioi": 15, "greater_than": 12, "subject_verb": 6}

n = args.n
device = "cuda" if torch.cuda.is_available() else "cpu"

dictionaries = load_saes(
    "saes/pythia-160m-deduped",
    "pythia-160m-deduped",
    model.cfg,
    modules,
    ckpt=args.ckpt,
    device=device,
    direction=args.direction,
    layer=args.layer,
)

train_examples = load_examples(f"tasks/{task}.json", 2 * n, model, length=lengths[task])[:n]
test_examples = load_examples(f"tasks/{task}.json", 2 * n, model, length=lengths[task])[n : 2 * n]
if len(test_examples) < 64:
    test_examples = load_examples(f"tasks/{task}.json", 2 * n, model, length=lengths[task])

assert (
    len(test_examples) > 64
), f"Not enough examples can be loaded, train length ({len(train_examples)}) test length ({len(test_examples)})"

train_tokens = torch.cat([e["clean_prefix"] for e in train_examples])

feature_cache = {}
hooks = [
    (
        hook_name,
        partial(
            sae_hook,
            sae=sae,
            cache=feature_cache,
        ),
    )
    for hook_name, sae in dictionaries.items()
]

with torch.no_grad():
    model.run_with_hooks(
        train_tokens.to(device),
        fwd_hooks=hooks,
    )

feature_avg = {k: v.mean(0) for k, v in feature_cache.items()}

effects = torch.load(f"effects/{task}_n{n}_{args.method}_{args.direction}_L{args.layer}_{args.ckpt}.pt")["nodes"]
# effects = {k: v for k, v in effects.items() if str(model.cfg.n_layers - 1) not in k}

test_tokens = torch.cat([e["clean_prefix"] for e in test_examples])
clean_answers = torch.tensor([e["clean_answer"] for e in test_examples])
patch_answers = torch.tensor([e["patch_answer"] for e in test_examples])

scores = []
Ns = []

# for T in tqdm(np.exp(np.linspace(-10, np.log(100), 64))):
for T in tqdm(args.K):
    # for T in tqdm([1024]):
    score, N = faithfulness(
        test_tokens,
        clean_answers,
        patch_answers,
        effects,
        dictionaries,
        node_threshold=T,
        use_resid=True,
        device=device,
    )
    scores.append(score)
    Ns.append(N)

score_df = pd.DataFrame({"score": scores, "N": Ns})
score_df.to_csv(
    f"faithfulness/{args.direction}_{task}_{args.method}_{args.component}_{args.what}_L{args.layer}_{args.ckpt}_K_{'_'.join(str(int(s)) for s in args.K)}.csv",
    index=False,
)
