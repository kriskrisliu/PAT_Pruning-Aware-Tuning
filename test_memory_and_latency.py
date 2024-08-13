import torch
import json
from transformers import AutoTokenizer, AutoConfig, AddedToken
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import argparse

external_cfg = {
    "ft_mode": 'none',
    "global_step": 10000,
    "dimdown_dim": 1111,
    "trainable_mask": True,
    "identity_loss": True,
}
with open("global_external_config.json","w") as fp:
    json.dump(external_cfg, fp)


parser = argparse.ArgumentParser(description="Arguments for running the model with specific configurations.")
# Add arguments
parser.add_argument("--model_name_or_path", type=str, default="merged_outputs/dimdown/240703_110219_llama2-7b-sft-dimdown2868-lamini0.5-bs128-len256-lora20eff200", help="Path to the model directory.")
# parser.add_argument("--template_name", type=str, default="gemma", help="Name of the template to use.")
parser.add_argument("--memory", action="store_true", help="")

args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    load_in_4bit=False,
    trust_remote_code=True,
    # low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=None,
    attn_implementation = "flash_attention_2"
)
if args.memory:
    import ipdb; ipdb.set_trace()