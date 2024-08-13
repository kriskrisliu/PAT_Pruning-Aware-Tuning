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

## Datasets
We employ `Lamini-Instruction` for fine-tuning, which can be found [here](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) in HuggingFace. Additionally, we provide our 50% randomly sampled data in this [link](https://box.nju.edu.cn/f/76ae99a847d44fb08cfe/).

## Evaluation
```bash
ADAPTER=<path-to-adaptor>
FT_MODE=dimdown
GPU=0
CUDA_VISIBLE_DEVICES=$GPU python chat.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--adapter_name_or_path $ADAPTER \
--template_name llama2-base-alpaca  \
--ft_mode $FT_MODE \
--trainable_mask \
--identity_loss \
--chat debug-all
```

## Model Zoo
We can merge the HSMs after PAT by using `script/merge_dimdown.py`.
```bash
ADAPTER=<path-to-adaptor>
python script/merge_dimdown.py \
--model_dir meta-llama/Llama-2-7b-hf \
--adaptor_path $ADAPTER
```
Additionally, we provide some PAT results [here]().
- [x] Llama 2 7B
- [x] Llama 2 13B
- [x] Gemma 2B
- [x] Gemma 7B
- [x] Yi-1.5 34B