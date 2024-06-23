import torch
import pandas as pd
from sae_lens import HookedSAETransformer
from tqdm import tqdm
import json

from circuit import IOICircuit
from tasks.ioi.dictionary import supervised_dictionary

device = "cuda"

def logits_diff(logits, correct_answer, incorrect_answer=None):
    correct_index = model.to_single_token(correct_answer)
    if incorrect_answer is None:
        return logits.cpu().numpy()[0, -1, correct_index]
    else:
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits.cpu().numpy()[0, -1, correct_index] - logits.cpu().numpy()[0, -1, incorrect_index]

def logits_score(clean_logits, patched_logits, corr_logits, correct_answer, incorrect_answer=None):
    clean_score = logits_diff(clean_logits, correct_answer, incorrect_answer)
    patched_score = logits_diff(patched_logits, correct_answer, incorrect_answer)
    corr_score = logits_diff(corr_logits, correct_answer, incorrect_answer)

    score = (patched_score - clean_score) / (corr_score - clean_score)
    return score

# Load task
with open('tasks/ioi/task.json') as f:
    task = json.load(f)

# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

ioi_circuit = IOICircuit(model, task)
ioi_circuit.load_saes('z')
ioi_circuit.load_saes('resid_pre')

# Compute supervised dictionary
bs = 64
prompts = [p['prompt'] for p in task['prompts']]
activations = {i: [] for i in ['q', 'k', 'v', 'z']}
for b in tqdm(range(0, len(prompts), bs)):
    tokens = model.to_tokens(prompts[b:b+bs])
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    for key in activations.keys():
        activations[key].append(cache.stack_activation(key))

activations = {key: torch.cat(values, 1) for key, values in activations.items()}
task_df = ioi_circuit.create_task_df()
io_vec, s_vec, pos_vec = supervised_dictionary(task_df, activations)


all_nodes_labels = ['bNMH-z', 'bNMH-q', 'bNMH-qk', 'IH+DTH-q', 'SIH-q', 'SIH-v', 'bNMH-z', 'bNMH-q', 'bNMH-qk', 'IH+DTH-q', 'SIH-q', 'SIH-v']
all_nodes = [['bNMH.z'], ['bNMH.q'], ['bNMH.qk'], ['IH.q', 'DTH.q'], ['SIH.q'], ['SIH.v'], ['bNMH.z'], ['bNMH.q'], ['bNMH.qk'], ['IH.q', 'DTH.q'], ['SIH.q'], ['SIH.v']]
attributes = ['IO', 'S', 'S', 'S', 'S', 'S', 'Pos', 'Pos', 'Pos', 'Pos', 'Pos']

for node_names, node_label, attribute in zip(all_nodes, all_nodes_labels, attributes):
    scores_df = {
        'clean_ld': [],
        'corr_ld': [],
        'patch_corr_ld': [],
        'patch_corr_score': [],
        'patch_supervised_ld': [],
        'patch_supervised_score': [],
        'patch_all_sae_ld': [],
        'patch_all_sae_score': [],
        'patch_sae_fs_ld': [],
        'patch_sae_fs_score': []
    }

    for idx in tqdm(range(512)):

        example = task['prompts'][idx]

        io = example['variables']['IO']
        s = example['variables']['S2']

        pos = example['variables']['Pos']
        neg_pos = 'ABB' if pos == 'BAB' else 'BAB'

        pos_id = 0 if pos == 'ABB' else 1

        ### CORRUPTED
        new_attr, clean_logits, patched_logits, corr_logits = ioi_circuit.run_with_patch(
            example, 
            node_names=node_names,
            attribute=attribute,
            method='corr',
            verbose=False 
            )

        scores_df['clean_ld'].append(logits_diff(clean_logits, ' '+io, ' '+new_attr).item())
        scores_df['corr_ld'].append(logits_diff(corr_logits, ' '+io, ' '+new_attr).item())
        scores_df['patch_corr_ld'].append(logits_diff(patched_logits, ' '+io, ' '+new_attr).item())
        scores_df['patch_corr_score'].append(logits_score(clean_logits, patched_logits, corr_logits, ' '+io, ' '+new_attr))

        ### SUPERVISED DICTIONARY
        patches = []
        for node in node_names:
            node, component = node.split('.')

            if attribute == 'IO':
                in_patch = io_vec[pos][new_attr]
                out_patch = s_vec[pos][io]
            elif attribute == 'S':
                in_patch = io_vec[pos][new_attr]
                out_patch = s_vec[pos][s]
            else:
                in_patch = pos_vec[neg_pos]
                out_patch = pos_vec[pos]

            patches.append((in_patch, out_patch))

        _, clean_logits, patched_logits, _ = ioi_circuit.run_with_patch(
            example, 
            node_names=node_names,
            attribute=attribute,
            method='feature',
            patches=patches,
            verbose=False
            )

        scores_df['patch_supervised_ld'].append(logits_diff(patched_logits, ' '+io, ' '+new_attr).item())
        scores_df['patch_supervised_score'].append(logits_score(clean_logits, patched_logits, corr_logits, ' '+io, ' '+new_attr))

        ### ALL SAE FEATURES
        _, clean_logits, patched_logits, corr_logits = ioi_circuit.run_with_patch(
            example, 
            node_names=node_names,
            attribute=attribute,
            method='sae-all',
            verbose=False,
            new_attr=new_attr
            )

        scores_df['patch_all_sae_ld'].append(logits_diff(patched_logits, ' '+io, ' '+new_attr).item())
        scores_df['patch_all_sae_score'].append(logits_score(clean_logits, patched_logits, corr_logits, ' '+io, ' '+new_attr))

        ### SAE FEATURE SEARCH 
        N = 10

        _, clean_logits, patched_logits, corr_logits = ioi_circuit.run_with_patch(
            example, 
            node_names=node_names,
            attribute=attribute,
            method='sae-fs',
            verbose=False,
            new_attr=new_attr
            )

        scores_df['patch_sae_fs_ld'].append(logits_diff(patched_logits, ' '+io, ' '+new_attr).item())
        scores_df['patch_sae_fs_score'].append(logits_score(clean_logits, patched_logits, corr_logits, ' '+io, ' '+new_attr))

    scores_df = pd.DataFrame(scores_df)
    scores_df.to_json(f'tasks/ioi/sc-scores/{attribute}_{node_label}.json')
