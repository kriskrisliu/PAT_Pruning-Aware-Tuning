## Introduction
This repo is the official implementation of `PAT: Pruning-Aware Tuning for Large Language Models`

## Acknowledgment
Modified from [FireFly](https://github.com/yangjianxin1/Firefly)

## Installation
```bash
conda create -n pat python=3.10
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
cd transformers-4.40.1
pip install -e .
cd ../peft
pip install -e .
cd ..
pip install -r requirements.txt
```

## Evaluation
```bash
ADAPTER=output/240605_230718_llama2-7b-sft-lora8-dimdown3072-lamini0.5-bs128-len256
FT_MODE=dimdown
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python chat.py --model_name_or_path meta-llama/Llama-2-7b-hf --adapter_name_or_path $ADAPTER --template_name llama2-base-alpaca  --ft_mode $FT_MODE --trainable_mask --identity_loss --chat debug-all
```


