# Base class for circuits
import re
import os
from functools import partial
import random
import torch
import pandas as pd
from transformer_lens import utils
from sae_lens import SAE
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from hooks import patching_hook, feature_patching_hook, editing_hook
import json
import time

with open('tasks/ioi/names.json') as f:
    NAMES = json.load(f)

DEVICE = 'cuda'

class IOIPrompt:
    def __init__(self, prompt, id):
        self.id = id
        self.text = prompt['prompt']
        self.variables = prompt['variables']

    def get_variable(self, variable_name):
        return self.variables[variable_name]

    def tokenize(self, model):
        self.tokens =  model.to_tokens(self.text)
        self.str_tokens = model.to_str_tokens(self.text)

    def get_variable(self, variable_name):
        return self.variables[variable_name]

    def generate_corrupted(self, attribute, model):

        if not hasattr(self, 'tokens'):
            self.tokenize(model)

        if attribute == 'IO' or attribute == 'S':
            attr = self.get_variable(attribute)[1:]
            new_attr = random.choice(list(set(NAMES) - {attr}))
            corr_prompt = {
                'prompt': self.text.replace(attr, new_attr),
                'variables': self.variables.copy()
            }
            for key in corr_prompt['variables']:
                if attribute in key:
                    corr_prompt['variables'][key] = new_attr
            return IOIPrompt(corr_prompt)
        elif attribute == 'POS':
            io_pos = self.get_variable('IO_pos')
            s1_pos = self.get_variable('S1_pos')
            neg_pos = 'ABB' if self.get_variable('POS') == 'BAB' else 'BAB'

            new_str_tokens = self.str_tokens.copy()
            new_str_tokens[io_pos] = self.str_tokens[s1_pos]
            new_str_tokens[s1_pos] = self.str_tokens[io_pos]

            corr_prompt = {
                'prompt': model.tokenizer.convert_tokens_to_string(new_str_tokens[1:]),
                'variables': self.variables.copy()
            }
            corr_prompt['variables']['POS'] = neg_pos
            corr_prompt['variables']['IO_pos'] = s1_pos
            corr_prompt['variables']['S1_pos'] = io_pos
            return  IOIPrompt(corr_prompt)


class TransformerCircuit:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.saes = {}

    def get_node(self, node_name):
        for node in self.cfg['nodes']:
            if node['name'] == node_name:
                return node
        return None

    def get_variable(self, variable_name):
        for variable in self.cfg['variables']:
            if variable['name'] == variable_name:
                return variable
        return None

    def get_node_attr(self, node_name, component):
        node = self.get_node(node_name)
        if node is not None:
            attr = node['q' if component in 'q' else 'kv']
            if '+' in attr: 
                var, offset = attr.split('+')
            elif '-' in attr:
                var, offset = attr.split('-')
            else:
                var, offset = attr, 0
            return var, int(offset)
        return None

    @classmethod
    def read_variable(self, x):
        if '+' in x:
            offset = int(x.split('+')[-1])
        elif '-' in x:
            offset = int(x.split('-')[-1])
        else:
            offset = 0

        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, x)[0], offset

    def load_saes(self, component, device='cuda'):
        if component == 'z':
            sae_id = "gpt2-small-hook-z-kk"
        elif component == 'resid_pre':
            sae_id = "gpt2-small-res-jb"

        layers = []
        for node in self.cfg['nodes']:
            for head in node['heads']:
                l, h = head.split('.')
                if l not in layers:
                    layers.append(l)
        
        for layer in tqdm(layers):
            sae, cfg_dict, _ = SAE.from_pretrained(
                sae_id,
                f'blocks.{layer}.hook_{component}',
                device=device
            )
            self.saes[sae.cfg.hook_name] = sae

# Class for IOI circuit
class IOICircuit(TransformerCircuit):
    def __init__(self, model, cfg, prompts, names):
        super().__init__(model, cfg)
        self.prompts = [IOIPrompt(p, id=i) for i, p in enumerate(prompts)]
        self.names = names
        self.components = ['q', 'k', 'v', 'z']

    def create_task_df(self):
        task_df = {
            'prompt': [],
            'IO': [],
            'S': [],
            'POS': [],
            'IO_pos': [],
            'S1_pos': [],
            'S1+1_pos': [],
            'S2_pos': [],
            'END': [],
        }

        for prompt in self.prompts:
            task_df['prompt'].append(prompt.text)
            task_df['POS'].append(prompt.get_variable('POS'))
            task_df['IO'].append(prompt.get_variable('IO'))
            task_df['S'].append(prompt.get_variable('S'))
            task_df['IO_pos'].append(prompt.get_variable('IO_pos'))
            task_df['S1_pos'].append(prompt.get_variable('S1_pos'))
            task_df['S1+1_pos'].append(prompt.get_variable('S1_pos') + 1)
            task_df['S2_pos'].append(prompt.get_variable('S2_pos'))
            task_df['END'].append(prompt.get_variable('END'))

        self.task_df = pd.DataFrame(task_df)
    

    def run_with_patch(
            self, 
            prompt: Dict[str, Union[str, Dict[str, str]]], 
            node_names: List[str], 
            attribute: str, 
            method: str = 'zero',
            patches: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
            verbose: bool = False,
            new_attr: Optional[str] = None,
            N: int = 4
        ) -> Tuple[Optional[str], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Function to run the circuit with a patch applied to the specified nodes.

        Args:
            prompt (dict): The prompt instance to be used.
            node_names (list): The names of the nodes to be patched.
            attribute (str): The attribute of the task to be patched.
            method (str): The method to be used for patching (one of 'zero', 'corr', 'feature', 'sae-fs', 'sae-all'). Default is 'zero'.
            patches (list, optional): The patches to be applied to the nodes if using 'feature' method. Default is None.
            verbose (bool): Whether to print verbose information. Default is False.
            new_attr (str, optional): The new attribute value for corruption. Default is None.
            N (int): Number of features for greedy feature selection in 'sae-fs' method. Default is 4.

        Returns:
            tuple: A tuple containing the new attribute, clean logits, patched logits, and corrupted logits.
        """
        
        def validate_inputs():
            assert method in ['zero', 'corr', 'feature', 'sae-fs', 'sae-all'], "Invalid method: must be 'zero', 'corr', 'feature', 'sae-fs', or 'sae-all'"
            assert attribute in ['IO', 'S', 'Pos'], "Invalid attribute: must be 'IO', 'S', or 'Pos'"
            for node_name in node_names:
                node, components = node_name.split('.')
                assert self.get_node(node) is not None, f"Node {node} not found"
                for c in components:
                    assert c in self.components, f"Component {c} not found in the task"

        def generate_corr_prompt(new_attr) -> str:
            if attribute == 'IO':
                new_io = ' ' + random.choice(list(set(self.names) - {io.strip(), s.strip()})) if new_attr is None else ' ' + new_attr
                new_attr = new_io.strip()
                return clean_prompt.replace(io, new_io), new_attr
            elif attribute == 'S':
                new_s = ' ' + random.choice(list(set(self.names) - {io.strip(), s.strip()})) if new_attr is None else ' ' + new_attr
                new_attr = new_s.strip()
                return clean_prompt.replace(s, new_s), new_attr
            elif attribute == 'Pos':
                new_template = self.task['templates'][1 - pos]
                return new_template.format(IO=io.strip(), S1=s.strip(), S2=s.strip()), s.strip()

        validate_inputs()

        clean_prompt = prompt['prompt']
        clean_tokens = self.model.to_tokens(clean_prompt)
        clean_str_tokens = self.model.to_str_tokens(clean_prompt)
        io = clean_str_tokens[prompt['variables']['IO']]
        s = clean_str_tokens[prompt['variables']['S1']]
        pos = 0 if prompt['variables']['POS'] == "ABB" else 1
        
        if verbose: 
            print(f"Clean prompt: {clean_prompt}")

        # Generate corrupted prompt if required
        corr_logits = None
        if method in ['corr', 'sae-fs', 'sae-all']:
            corr_prompt, new_attr = generate_corr_prompt(new_attr)
            corr_tokens = self.model.to_tokens(corr_prompt)

            assert clean_tokens.shape[-1] == corr_tokens.shape[-1], "Clean and corrupted prompts must have the same length"

            if verbose: 
                print(f"Corrupted prompt: {corr_prompt}")
            with torch.no_grad():
                corr_logits, corr_cache = self.model.run_with_cache(corr_tokens)

        # Get clean logits and cache
        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)

        hooks = []
        z_vectors = [[] for _ in range(self.model.cfg.n_layers)]

        # Iterate over the nodes
        for i, name in enumerate(node_names):
            node_name, components = name.split('.')
            node = self.get_node(node_name)

            for component_name in components:
                var, offset = self.read_variable(node['q']) if component_name in ['q', 'z'] else self.read_variable(node['kv'])
                var_pos = prompt['variables'][var] + offset

                for head in node['heads']:
                    l, h = map(int, head.split('.'))
                    hook_name = utils.get_act_name(component_name, l)

                    if method == 'corr':
                        hook_fn = partial(patching_hook, pos=var_pos, head=h, patch=corr_cache[hook_name][:, var_pos, h])
                    elif method == 'feature':
                        f_in, f_out = patches[i][0][component_name][l], patches[i][1][component_name][l]
                        hook_fn = partial(feature_patching_hook, pos=var_pos, head=h, f_in=f_in, f_out=f_out)
                    elif 'sae' in method:
                        clean_acts = torch.matmul(clean_cache[hook_name][:, var_pos, h], self.model.W_O[l, h]) if component_name == "z" else clean_cache[utils.get_act_name('resid_pre', l)][:, var_pos] # [1 dm]
                        corr_acts = torch.matmul(corr_cache[hook_name][:, var_pos, h], self.model.W_O[l, h]) if component_name == "z" else corr_cache[utils.get_act_name('resid_pre', l)][:, var_pos] # [1 dm]

                        sae = self.saes[utils.get_act_name('resid_pre', l)] if component_name != "z" else self.saes[hook_name]

                        if method == 'sae-all':
                            with torch.no_grad():
                                clean_sae_acts = sae(clean_acts)
                                corr_sae_acts = sae(corr_acts)
                            edit = corr_sae_acts - clean_sae_acts
                        else:
                            edit = greedy_fs(sae, clean_acts, corr_acts, var_pos, h, N)

                        if component_name == "z":
                            z_vectors[l].append((var_pos, edit))
                        else:
                            M = getattr(self.model, f'W_{component_name.upper()}')[l, h]
                            hook_fn = partial(editing_hook, pos=var_pos, head=h, edit=torch.matmul(edit, M))

                    if not ('sae' in method and component_name == "z"):
                        hooks.append((hook_name, hook_fn))
                    
                    if verbose:
                        print(f"Hooking L{l}H{h} {component_name} at position {var_pos}")

        if 'sae' in method:
            for l in range(self.model.cfg.n_layers):
                pos_dict = {}
                hook_name = utils.get_act_name('attn_out', l)
                if z_vectors[l]:
                    for pos, feature_vector in z_vectors[l]:
                        pos_dict[pos] = pos_dict.get(pos, 0) + feature_vector
                    for pos, feature_vector in pos_dict.items():
                        hooks.append((hook_name, partial(editing_hook, pos=pos, edit=feature_vector)))

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        return new_attr, clean_logits, patched_logits, corr_logits

    def run_with_reconstruction(
            self,
            prompt: IOIPrompt,
            method: str,
            node_names: List[str],
            cache: Dict[str, torch.Tensor],
            reconstruction: str = 'full',
            verbose: bool = False
        ) -> Tuple[Optional[str], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Function to run the circuit with a reconstructed activations from the SAE.

        Args:
            prompt (cls): The prompt instance to be used.
            method (str): The method to be used for patching (either 'supervised' or 'sae').
            node_names (list): The names of the nodes to be patched.
            reconstruction (str): The type of reconstruction to be used (either 'full' or 'average'). Default is 'full'.
            verbose (bool): Whether to print verbose information. Default is False.

        Returns:
            tuple: A tuple containing the clean logits and the reconstructed logits.
        """

        if verbose: 
            print(f"Prompt: {prompt.text}\nCaching activations...")

        if method == 'ablation':
            reconstruction = 'necessity'

        # Get logits and cache
        #with torch.no_grad():
        #    logits, cache = self.model.run_with_cache(prompt.tokens)

        hooks = []
        z_vectors = torch.zeros((1, self.model.cfg.n_layers, prompt.tokens.shape[-1], self.model.cfg.d_model), device=DEVICE) # [1 l p dm]

        # Iterate over the nodes
        for i, name in enumerate(node_names):
            
            node_name, components = name.split('.')
            node = self.get_node(node_name)

            for component_name in components:
                # Read editing position
                attr, offset = self.get_node_attr(node_name, component_name)
                full_attr = node['q' if component_name in 'qz' else 'kv']
                attr_pos = prompt.get_variable(f"{attr}_pos" if attr != "END" else attr) + offset

                for head in node['heads']:
                    l, h = map(int, head.split('.'))
                    hook_name = utils.get_act_name(component_name, l)

                    acts = cache[hook_name][:, attr_pos, h] # [1 dh]

                    """
                    Reconstruction
                    Z - Sufficency: o' = o + (f_h - z_h) @ W_O_h = f_h @ W_O_h
                    Z -  Necessity: o' = o + (z_bar_h - f_h) @ W_O_h
                    Z -   Ablation: o' = o + (z_bar_h - z_h) @ W_O_h
                    Supervised reconstructs f_h while SAE reconstructs f_h @ W_O_h
                    
                    QKV - Sufficency: a_h' = a_h + (f_h - a_h) = f_h
                    QKV - Necessity: a_h' = a_h + (a_bar_h - f_h)
                    QKV -  Ablation: a_h' = a_h + (a_bar_h - a_h)
                    Supervised reconstructs f_h while SAE reconstructs x_pre, from which f_h is extracted as f_h = x @ W_A_h 
                    (with A in {Q, K, V})
                    """

                    if component_name == "z":
                        W = self.model.W_O[l, h] # [dh dm]
                        #b = self.model.b_O[l, None] # [1 dm]
                        zW = torch.matmul(acts, W) # [1 dm]
                        if reconstruction == 'necessity':
                            z_bar = self.avg_activations[component_name][full_attr][None, l, h].to(DEVICE) # [1 dh]
                            z_barW = torch.matmul(z_bar, W) # [1 dm]
                        
                        if method == 'supervised':
                            f = self.get_supervised_feature(attribute='IO', name=prompt.get_variable('IO'), component=component_name, position=full_attr)
                            f += self.get_supervised_feature(attribute='S', name=prompt.get_variable('S'), component=component_name, position=full_attr)
                            f += self.get_supervised_feature(attribute='POS', name=prompt.get_variable('POS'), component=component_name, position=full_attr) # [l h dm]
                            
                            fW = torch.matmul(f[None, l, h].to(DEVICE), W) # [1 dm]
                            edit = fW - zW if reconstruction == 'sufficency' else z_barW - fW
                        elif method == 'sae':
                            sae = self.saes[hook_name] 
                            fW = sae(zW) # [1 dm]
                            edit = fW - zW if reconstruction == 'sufficency' else z_barW - fW
                        elif method == 'ablation':
                            edit = z_barW - zW
                        
                        z_vectors[:, l, attr_pos] += edit

                    else:
                        if method == 'ablation':
                            acts_bar = self.avg_activations[component_name][full_attr][None, l, h].to(DEVICE) # [1 dh]
                            edit = acts_bar - acts
                        else:
                            if method == 'supervised':
                                fW = self.get_supervised_feature(attribute='IO', name=prompt.get_variable('IO'), component=component_name, position=full_attr)
                                fW += self.get_supervised_feature(attribute='S', name=prompt.get_variable('S'), component=component_name, position=full_attr)
                                fW += self.get_supervised_feature(attribute='POS', name=prompt.get_variable('POS'), component=component_name, position=full_attr) # [l h dm]
                                fW = fW[None, l, h].to(DEVICE)

                            elif method == 'sae':
                                acts = cache[utils.get_act_name('resid_pre', l)][:, attr_pos] # [1 dm]
                                sae = self.saes[utils.get_act_name('resid_pre', l)]
                                with torch.no_grad():
                                    f = sae(acts) # [1 dm]
                                    f = self.model.blocks[l].ln1(f) # [1 dm]
                                    acts = self.model.blocks[l].ln1(acts) # [1 dm]
                                W = getattr(self.model, f'W_{component_name.upper()}')[l, h] # [dm dh]
                                #b = getattr(self.model, f'b_{component_name.upper()}')[l, h, None] # [1 dh]
                                fW = torch.matmul(f, W) # [1 dh]
                                acts = torch.matmul(acts, W) # [1 dh]
                            
                            if reconstruction == 'sufficency':
                                edit = fW - acts
                            elif reconstruction == 'necessity':
                                acts_bar = self.avg_activations[component_name][full_attr][None, l, h].to(DEVICE) # [1 dh]
                                edit = acts_bar - fW # [1 dh]
                
                        hook_fn = partial(editing_hook, pos=attr_pos, head=h, edit=edit)
                        hooks.append((hook_name, hook_fn))

                    if verbose:
                        print(f"Hooking L{l}H{h} {component_name} ({full_attr}) at position {attr_pos}")
   
        for l in range(self.model.cfg.n_layers):
            hook_name = utils.get_act_name('attn_out', l)
            for pos in range(prompt.tokens.shape[-1]):
                if z_vectors[:, l, pos].sum() != 0:
                    hooks.append((hook_name, partial(editing_hook, pos=pos, edit=z_vectors[:, l, pos])))

        with torch.no_grad():
            reconstructed_logits = self.model.run_with_hooks(prompt.tokens, fwd_hooks=hooks)

        return reconstructed_logits

    def get_activations(self, batch_size: int = 64):
        if not hasattr(self, 'task_df'):
            print("Task dataframe not found. Creating task dataframe...")
            self.create_task_df()

        if os.path.exists('tmp/activations.pt'):
            print("Loading activations...")
            self.activations = torch.load('tmp/activations.pt')
            self.avg_activations = {k: {a: v.mean(1) for a, v in self.activations[k].items()} for k in self.activations.keys()}
            return

        activations = {i: [] for i in ['q', 'k', 'v', 'z']}

        for b in tqdm(range(0, len(self.task_df), batch_size)):
            tokens = self.model.to_tokens(self.task_df.iloc[b:b+batch_size, 0])

            positions = self.task_df.iloc[b:b+batch_size][['IO_pos', 'S1_pos', 'S1+1_pos', 'S2_pos', 'END']].values
            
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)

            for key in activations.keys():
                key_acts = cache.stack_activation(key) # [l b pos h dm]
                activations[key].append(torch.cat([key_acts[:, None, i, pos] for i, pos in enumerate(positions)], dim=1).cpu())

            del cache

        self.activations = {}
        self.avg_activations = {}
        attributes = ['IO', 'S1', 'S1+1', 'S2', 'END']

        for key in activations.keys():
            key_acts = torch.cat(activations[key], 1).cpu() # [l N 5 h dh]
            self.activations[key] = {a: key_acts[:, :, i] for i, a in enumerate(attributes)} # [l N h dh]
            self.avg_activations[key] = {k: v.mean(1) for k, v in self.activations[key].items()} # [l h dh]

        print("Activations computed. Saving activations...")
        torch.save(self.activations, 'tmp/activations.pt')

    def compute_supervised_dictionary(self):

        if not hasattr(self, 'activations'):
            print("Activations not found. Computing activations...")
            self.get_activations()

        if os.path.exists('tmp/io_features.pt') and os.path.exists('tmp/s_features.pt') and os.path.exists('tmp/pos_features.pt'):
            print("Loading supervised dictionaries...")
            self.IO_features = torch.load('tmp/io_features.pt')
            self.S_features = torch.load('tmp/s_features.pt')
            self.POS_features = torch.load('tmp/pos_features.pt')
            return
    
        io_vec = {}
        s_vec = {}

        unique_io = self.task_df['IO'].unique()
        unique_s = self.task_df['S'].unique()
        
        print("Computing supervised dictionary...")
        centered_activations = {}
        for c in ['q', 'k', 'v', 'z']:
            c_acts = {}
            for attr in self.activations[c]:
                c_acts[attr] = self.activations[c][attr] - self.avg_activations[c][attr][:, None]
            centered_activations[c] = c_acts
    

        for name in tqdm(set(unique_io) | set(unique_s)):
            io_vec[name] = {}
            s_vec[name] = {}

            for c in ['q', 'k', 'v', 'z']:
                mask = torch.tensor(self.task_df['IO'] == name, dtype=torch.bool, device='cpu')
                if mask.any():
                    io_vec[name][c] = {attr: centered_activations[c][attr][:, mask].mean(1) for attr in self.activations[c]}
                
                mask = torch.tensor(self.task_df['S'] == name, dtype=torch.bool, device='cpu')
                if mask.any():
                    s_vec[name][c] = {attr: centered_activations[c][attr][:, mask].mean(1) for attr in self.activations[c]}

        # POS
        pos_vec = {'ABB': {}, 'BAB': {}}
        
        for c in ['q', 'k', 'v', 'z']:
            for pos in pos_vec.keys():
                mask = torch.tensor(self.task_df['POS'] == pos, dtype=torch.bool, device='cpu')
                if mask.any():
                    pos_vec[pos][c] = {attr: centered_activations[c][attr][:, mask].mean(1) for attr in self.activations[c]}
        
        self.IO_features = io_vec
        self.S_features = s_vec
        self.POS_features = pos_vec

        # Save dictionaries
        print("Dictionaries computed. Saving supervised dictionaries...")
        torch.save(io_vec, 'tmp/io_features.pt')
        torch.save(s_vec, 'tmp/s_features.pt')
        torch.save(pos_vec, 'tmp/pos_features.pt')
    
    def get_supervised_feature(self, attribute, component, position, name=None):
        if attribute == 'IO':
            return self.IO_features[name][component][position]
        elif attribute == 'S':
            return self.S_features[name][component][position]
        elif attribute == 'POS':
            return self.POS_features[name][component][position] # [l h dh]


# Feature search functions
def optimization_fs(sae, clean_acts, corr_acts, pos, head, N):
    W_dec = sae.W_dec.clone().detach()

    # Step 1) Find the top N features by sorting loss gradients
    f_in = torch.ones(W_dec.shape[0], device=W_dec.device, dtype=torch.float) / W_dec.shape[0]
    f_in.requires_grad = True

    transformed_acts = clean_acts + torch.matmul(f_in, W_dec)
    loss = torch.norm(transformed_acts - corr_acts)
    loss.backward()

    gradient_importance = f_in.grad.abs()
    best_features_ids = torch.argsort(gradient_importance, descending=True)[:N]
    best_features = W_dec[best_features_ids] # N x dm

    # Step 2) Optimize the weights of the top N features
    alphas = torch.randn(N, device=W_dec.device, dtype=torch.float) / N
    alphas.requires_grad = True
    optimizer = torch.optim.SGD([alphas], lr=0.02)

    for i in range(500):
        optimizer.zero_grad()

        transformed_acts = clean_acts + torch.matmul(alphas, best_features)
        
        loss = torch.norm(transformed_acts - corr_acts)
        loss.backward()
        optimizer.step()

    optimized_alphas = alphas.detach()

    return torch.matmul(optimized_alphas, best_features)

def greedy_fs(sae, clean_acts, corr_acts, pos, head, N):
    with torch.no_grad():
        _, sae_cache = sae.run_with_cache(clean_acts)
        _, corr_cache = sae.run_with_cache(corr_acts)

    sae_clean_acts = sae_cache['hook_sae_acts_post'].squeeze() # [dsae]
    sae_corr_acts = corr_cache['hook_sae_acts_post'].squeeze() # [dsae]

    W_dec = sae.W_dec.clone().detach() # [dsae dm]

    clean_acts_ = clean_acts.clone().detach() # [1 dm]
    corr_acts_ = corr_acts.clone().detach() # [1 dm]

    best_out_feature = []
    best_in_feature = []

    for i in range(N):        
        transformed_acts = clean_acts_ - sae_clean_acts[:, None] * W_dec # [dsae dm]
        loss = torch.norm(transformed_acts - corr_acts_, dim=1) # [dsae]
        best_out_feature.append(torch.argmin(loss).item())
        clean_acts_ -= sae_clean_acts[best_out_feature[-1]] * W_dec[best_out_feature[-1]]

        transformed_acts = corr_acts_ - sae_corr_acts[:, None] * W_dec # [dsae dm]
        loss = torch.norm(clean_acts_  - transformed_acts, dim=1) # [dsae]
        best_in_feature.append(torch.argmin(loss).item())
        corr_acts_ -= sae_corr_acts[best_in_feature[-1]] * W_dec[best_in_feature[-1]]

    feature_vector = clean_acts_ - clean_acts + corr_acts - corr_acts_
    return feature_vector