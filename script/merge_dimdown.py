import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file
import json
from tqdm import tqdm
from safetensors.torch import save_file
import shutil
import numpy as np
import argparse

# model_dir = "../Yi-1.5/AI-ModelScope/gemma-2b"
# adaptor_path = "output/240704_003834_gemma-2b-lamini0.5-bs128-len256-dd1536-lora20eff300/adapter_model.safetensors"
# adaptor_config = "output/240704_003834_gemma-2b-lamini0.5-bs128-len256-dd1536-lora20eff300/adapter_config.json"

# model_dir = "/data/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# adaptor_path = "output/240703_110219_llama2-7b-sft-dimdown2868-lamini0.5-bs128-len256-lora20eff200/adapter_model.safetensors"
# adaptor_config = "output/240703_110219_llama2-7b-sft-dimdown2868-lamini0.5-bs128-len256-lora20eff200/adapter_config.json"

parser = argparse.ArgumentParser(description='Process arguments for your script.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Directory where the main model is stored')
parser.add_argument('--adaptor_path', type=str, required=True,
                    help='Path to the adapter model')
# parser.add_argument('--adaptor_config', type=str, required=True,
#                     help='Path to the adapter configuration file')

args = parser.parse_args()
model_dir = args.model_dir
adaptor_path = os.path.join(args.adaptor_path, "adapter_model.safetensors")
adaptor_config = os.path.join(args.adaptor_path, "adapter_config.json")

output_dir = f"merged_outputs/dimdown/{adaptor_path.split('/')[-2]}"

with open(os.path.join(model_dir, "config.json"),'r') as f:
    model_config = json.load(f)

with open(adaptor_config) as f:
    config = json.load(f)
lora_alpha = config["lora_alpha"]
lora_rank = config["r"]

tensors_adpt = {}
with safe_open(adaptor_path, framework="pt", device="cuda") as f:
   for key in f.keys():
       tensors_adpt[key.replace("base_model.model.","")] = f.get_tensor(key)
# print(tensors_adpt.keys())

tensors_model = {}
st_name_keys = {}
for ff in sorted(os.listdir(model_dir)):
    if ff.endswith(".safetensors"):
        print(ff)
        if st_name_keys.get(ff, None) is None:
            st_name_keys[ff] = []
        with safe_open(os.path.join(model_dir, ff), framework="pt", device="cuda") as f:
            metadata = f.metadata()
            for key in f.keys():
                st_name_keys[ff].append(key)
                tensors_model[key] = f.get_tensor(key)

# merge lora
for k,v in tensors_adpt.items():
    if "lora_A.weight" in k:
        lora_a = tensors_adpt[k]
        lora_b = tensors_adpt[k.replace("lora_A.weight","lora_B.weight")]
        lora_merge = lora_b @ lora_a * lora_alpha / lora_a.size(0)
        # import ipdb;ipdb.set_trace()
        # merge to base model
        tensors_model[k.replace("lora_A.","")] = lora_merge + tensors_model[k.replace("lora_A.","")]
        print(f"Merge: {k}  +  {k.replace('lora_A','lora_B')}  =>  {k.replace('lora_A.','')}")

# delete lora weights after merge
lora_keys = [k for k in tensors_adpt.keys() if "lora" in k]
for k in lora_keys:
    del tensors_adpt[k]

# extract dimdown mask
mask_key = [k for k in tensors_adpt.keys() if "dimdown_mask.mask" in k]
assert len(mask_key)==1
mask_key = mask_key[0]
mask_value = tensors_adpt.pop(mask_key)
temperature = 1e6
bias = 0.0
scalar = 1.0/(1.0 + torch.exp((-temperature * mask_value).clamp(max=10))) + bias
scalar = scalar.round()
remain_idx = torch.where(scalar.flatten()>0)[0]


last_layer_idx = 0
for k,v in tensors_adpt.items():
    if "layers." in k:
        layer_idx = int(k.split(".")[2])
        if layer_idx > last_layer_idx:
            last_layer_idx = layer_idx
        
for k,v in tqdm(tensors_adpt.items()):
    if "dimdown_inout.0" in k:
        inout0 = v
        inout1 = tensors_adpt[k.replace("dimdown_inout.0","dimdown_inout.1")]
        inout = inout1 @ inout0
        inout += torch.eye(inout.shape[0]).to(inout)
        if k.startswith("model.dimdown0"):
            value = tensors_model['model.embed_tokens.weight']
            tensors_model['model.embed_tokens.weight'] = (value @ inout.to(value).T)[:,remain_idx]
            target_key = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.input_layernorm.weight",
            ]
        elif "dimdown0.dimdown_inout" in k:
            layer_idx = (k.split(".")[2])
            prefix = k.split(".")[:3]
            prefix = ".".join(prefix)
            postfix_list = [
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.up_proj.weight",
                "mlp.gate_proj.weight",
            ]
            target_key = [prefix + '.' + postfix for postfix in postfix_list]
            # import ipdb;ipdb.set_trace()
        elif "dimdown1.dimdown_inout" in k:
            layer_idx = int(k.split(".")[2])+1
            prefix = k.split(".")[:2]
            prefix = ".".join(prefix)
            if layer_idx > last_layer_idx:
                target_key = [
                    prefix+"."+f"{layer_idx-1}.mlp.down_proj.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                ]
            else:
                postfix_list = [
                    f"{layer_idx-1}.mlp.down_proj.weight",
                    f"{layer_idx}.input_layernorm.weight",
                    f"{layer_idx}.self_attn.q_proj.weight",
                    f"{layer_idx}.self_attn.k_proj.weight",
                    f"{layer_idx}.self_attn.v_proj.weight",
                ]
                target_key = [prefix + '.' + postfix for postfix in postfix_list]
        else:
            raise NotImplementedError
        for ii in range(len(target_key)):
            value = tensors_model[target_key[ii]]

            if "norm" in target_key[ii]:
                value = value * np.sqrt(value.shape[0] / len(remain_idx))
                # import ipdb;ipdb.set_trace()

            if len(value.shape)==1:
                value = value[remain_idx]
            elif len(value.shape)==2:
                if any(kk in target_key[ii] for kk in ["down_proj", "o_proj"]):
                    value = ( inout.to(value) @ value)[remain_idx,:]
                else:
                    value = value[:,remain_idx]
            else:
                raise NotImplementedError
            tensors_model[target_key[ii]] = value
        for key in target_key:
            print(f"Finish =>  {key}")

for k,v in tensors_model.items():
    print(f"{k}  =>  {v.shape}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# copy all json files in model_dir to output_dir
for file in os.listdir(model_dir):
    if file.endswith(".json"):
        shutil.copy(os.path.join(model_dir, file), output_dir)
        print(f"Copy {file} to {output_dir}")
shutil.copy(os.path.join(model_dir, "tokenizer.model"), output_dir)
print(f"Copy tokenizer.model to {output_dir}")

model_config["hidden_size"] = len(remain_idx)
with open(os.path.join(output_dir, "config.json"), 'w') as fp:
    json.dump(model_config, fp, indent=4)
for k,v in tqdm(st_name_keys.items()):
    file_path = os.path.join(output_dir, k)
    tensors_to_save = {}
    for layer_name in v:
        tensors_to_save[layer_name] = tensors_model[layer_name].cpu()
    save_file(tensors_to_save, os.path.join(output_dir, k),metadata=metadata)
    print(f"Save {k} to {output_dir}")
# import ipdb;ipdb.set_trace()

