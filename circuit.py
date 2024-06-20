# Base class for circuits
import re
from functools import partial
import random
import torch
import pandas as pd
from transformer_lens import utils
from sae_lens import SAE
from tqdm import tqdm
from hooks import zero_abl_hook, patching_hook, feature_patching_hook, feature_qkv_patching_hook, feature_z_patching_hook

class TransformerCircuit:
    def __init__(self, model, task):
        self.model = model
        self.task = task
        self.saes = {}

    def get_node(self, node_name):
        for node in self.task['nodes']:
            if node['name'] == node_name:
                return node
        return None

    def get_variable(self, variable_name):
        for variable in self.task['variables']:
            if variable['name'] == variable_name:
                return variable
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

        for layer in tqdm(range(self.model.cfg.n_layers)):
            sae, cfg_dict, _ = SAE.from_pretrained(
                sae_id,
                f'blocks.{layer}.hook_{component}',
                device=device
            )
            self.saes[sae.cfg.hook_name] = sae

# Class for IOI circuit
class IOICircuit(TransformerCircuit):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.names = self.get_variable('IO')['values']
        self.components = ['q', 'k', 'v', 'z']

    def create_task_df(self):
        task_df = {
            'prompt': [],
            'IO': [],
            'S1': [],
            'S2': [],
            'Pos': [],
            'IO_pos': [],
            'S1_pos': [],
            'S2_pos': []
        }

        for i, prompt in enumerate(self.task['prompts']):
            task_df['prompt'].append(prompt['prompt'])
            task_df['IO'].append(prompt['variables']['IO'])
            task_df['S1'].append(prompt['variables']['S1'])
            task_df['S2'].append(prompt['variables']['S2'])

            pos = prompt['variables']['Pos']
            pos = 0 if pos == "ABB" else 1
            task_df['Pos'].append(pos)

            io_pos = self.get_variable('IO')['position'][pos]
            s1_pos = self.get_variable('S1')['position'][pos]
            s2_pos = self.get_variable('S2')['position'][pos]

            task_df['IO_pos'].append(io_pos)
            task_df['S1_pos'].append(s1_pos)
            task_df['S2_pos'].append(s2_pos)

        return pd.DataFrame(task_df)

    def run_with_patch(
            self, 
            prompt: dict, 
            node_names: str, 
            attribute: str, 
            method: str ='zero',
            patches: list=None,
            verbose: bool=False,
            new_attr: str=None,
            N: int=4
            ):
        '''
        Function to run the circuit with a patch applied to the specified nodes.
        Takes in:
        - prompt: the prompt instance to be used
        - node_names: the names of the nodes to be patched
        - attribute: the attribute of the task to be patched
        - method: the method to be used for patching (one of 'zero', 'corr', 'feature')
        - patches: the patches to be applied to the nodes if using 'feature' method

        Returns the difference of answer logits between the clean and patched prompt.
        '''
        # Checks
        assert method in ['zero', 'corr', 'feature', 'sae'], "Method must be either 'zero', 'corr', 'feature'"
        assert attribute in ['IO', 'S', 'Pos'], "Attribute must be either 'IO', 'S', 'Pos'"
        for node_name in node_names:
            node, components = node_name.split('.')
            assert self.get_node(node) is not None, f"Node {node_name} not found in the task"
            for c in components:
                assert c in self.components, f"Component {c} not found in the task"
        
        # Extracting variables from the prompt
        io = ' '+prompt['variables']['IO']
        s = ' '+prompt['variables']['S1']
        pos = prompt['variables']['Pos']
        pos = 0 if pos == "ABB" else 1

        clean_prompt = prompt['prompt']
        clean_tokens = self.model.to_tokens(clean_prompt)
        if verbose: print(f"Clean prompt: {clean_prompt}")

        if method in ['corr', 'sae']:
            # Creation of the corrupted prompt based on the attribute
            if attribute == 'IO':
                # Pick a random new name to put in IO
                new_io = ' ' + random.choice(list(set(self.names) - {io[1:], s[1:]})) if new_attr is None else ' '+new_attr
                new_attr = new_io[1:]
                corr_prompt = clean_prompt.replace(io, new_io)
            elif attribute == 'S':
                # Pick a random new name to put in S
                new_s = ' ' + random.choice(list(set(self.names) - {io[1:], s[1:]})) if new_attr is None else ' '+new_attr
                new_attr = new_s[1:]
                corr_prompt = clean_prompt.replace(s, new_s)
            elif attribute == 'Pos':
                # Generate a new prompt with the opposite template
                new_template = self.task['templates'][1-pos]
                corr_prompt = new_template.format(IO=io[1:], S1=s[1:], S2=s[1:])

            corr_tokens = self.model.to_tokens(corr_prompt)
            if verbose: print(f"Corrupted prompt: {corr_prompt}")
            
            with torch.no_grad():
                _, corr_cache = self.model.run_with_cache(corr_tokens)

        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        
        # Hooking the model for patching
        hooks = []
        z_vectors = [[] for l in range(self.model.cfg.n_layers)]

        for i, name in enumerate(node_names):
            # Split the node name into components (e.g. IH.qk -> IH, qk)
            node_name, components = name.split('.')
            node = self.get_node(node_name)

            # Add hooks to selected component of the node
            for component_name in components:
                if component_name in ['q', 'z']:
                    # Read the variable name and offset (e.g. IH.q -> S2+0 | PTH.z -> S1+1)
                    var, offset = self.read_variable(node['q'])
                else:
                    var, offset = self.read_variable(node['kv'])

                # Retrieve the token position of the variable
                var_pos = self.get_variable(var)['position'][pos] + offset

                # Add hooks to the component of each node head
                for head in node['heads']:
                    l, h = head.split('.')
                    l, h = int(l), int(h)
                    hook_name = utils.get_act_name(component_name, l)
                    
                    if method == 'zero':
                        hook_fn = partial(zero_abl_hook, pos=var_pos, head=h)
                    elif method == 'corr':
                        hook_fn = partial(patching_hook, pos=var_pos, head=h, corr=corr_cache[hook_name])
                    elif method == 'feature':
                        f_in = patches[i][0][component_name][l]
                        f_out = patches[i][1][component_name][l]
                        hook_fn = partial(feature_patching_hook, pos=var_pos, head=h, f_in=f_in, f_out=f_out)
                    elif method == 'sae':
                        if component_name == "z":
                            sae = self.saes[hook_name]
                            clean_acts = torch.matmul(clean_cache[hook_name][:, var_pos, h], self.model.W_O[l, h]).detach()
                            corr_acts = torch.matmul(corr_cache[hook_name][:, var_pos, h], self.model.W_O[l, h]).detach()
                        else:
                            hook_name = utils.get_act_name('resid_pre', l)
                            sae = self.saes[hook_name]
                            clean_acts = clean_cache[hook_name][:, var_pos]
                            corr_acts = corr_cache[hook_name][:, var_pos]
                        
                        feature_vector = greedy_fs(sae, clean_acts, corr_acts, var_pos, h, N)

                        if component_name == "z":
                            z_vectors[l].append((var_pos, feature_vector))
                        else:
                            hook_name = utils.get_act_name(component_name, l)
                            rs_pre = clean_cache[utils.get_act_name('resid_pre', l)][:, var_pos]
                            if component_name == "q":
                                M = self.model.W_Q[l, h]
                            elif component_name == "k":
                                M = self.model.W_K[l, h]
                            elif component_name == "v":
                                M = self.model.W_V[l, h]
                            edit = torch.matmul(rs_pre + feature_vector[None], M)
                            hook_fn = partial(feature_qkv_patching_hook, pos=var_pos, head=h, edit=edit)

                    if method != 'sae':
                        hooks.append((hook_name, hook_fn))
                    if verbose: print(f"Hooking L{l}H{h} {component_name} at position {var_pos}")

        # Add z hooks
        if method == 'sae':            
            for l in range(self.model.cfg.n_layers):
                pos_dict = {}
                hook_name = utils.get_act_name('attn_out', l)
                if len(z_vectors[l]) > 0:
                    for pos, feature_vector in z_vectors[l]:
                        if pos in pos_dict:
                            pos_dict[pos] += feature_vector
                        else:
                            pos_dict[pos] = feature_vector
                    
                    for pos, feature_vector in pos_dict.items():
                        hook_fn = partial(feature_z_patching_hook, pos=pos, f_in=feature_vector)
                        hooks.append((hook_name, hook_fn))

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        return new_attr, clean_logits, patched_logits


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
    return feature_vector.squeeze()