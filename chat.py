from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
import copy
import argparse

import sys
# sys.path.append("../../")
from component.utils import ModelUtils
from component.template import template_dict
import os
from datasets import load_dataset
from tqdm import tqdm
from prompt_template import *
from torch.utils.data import DataLoader, Subset
import numpy as np
import re
from safetensors import safe_open

import random
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)  # Set your desired seed value here


# Manually building the dict based on the provided function names and the user's requirement
FUNC_POOL = {
    "boolq":prompt_one_example_on_boolq,
    "cb":prompt_one_example_on_cb,
    "multirc":prompt_one_example_on_multirc,
    "wic":prompt_one_example_on_wic,
    "wsc":prompt_one_example_on_wsc,
    "rte":prompt_one_example_on_rte,
    "copa":prompt_one_example_on_copa,
    "WinoGrande":prompt_one_example_on_WinoGrande,
    "openbookqa":prompt_one_example_on_openbookqa,
    "anli":prompt_one_example_on_anli,
    "record":prompt_one_example_on_record,
    "hellaswag":prompt_one_example_on_hellaswag,
    "piqa":prompt_one_example_on_piqa,
    "mmlu":prompt_one_example_on_mmlu,
    "arce":prompt_one_example_on_arce,
    "commonsense_qa":prompt_one_example_on_commonsense_qa,
    "siqa":prompt_one_example_on_siqa,
    "boolean_expressions":prompt_one_example_on_bbh_boolean_expressions,
    "causal_judgement":prompt_one_example_on_bbh_causal_judgement,
    "date_understanding":prompt_one_example_on_bbh_date_understanding,
    "disambiguation_qa":prompt_one_example_on_bbh_disambiguation_qa,
    "dyck_languages":prompt_one_example_on_bbh_dyck_languages,
    "formal_fallacies":prompt_one_example_on_bbh_formal_fallacies,
    "geometric_shapes":prompt_one_example_on_bbh_geometric_shapes,
    "hyperbaton":prompt_one_example_on_bbh_hyperbaton,
    "logical_deduction_five_objects":prompt_one_example_on_bbh_logical_deduction_five_objects,
    "logical_deduction_seven_objects":prompt_one_example_on_bbh_logical_deduction_seven_objects,
    "logical_deduction_three_objects":prompt_one_example_on_bbh_logical_deduction_three_objects,
    "movie_recommendation":prompt_one_example_on_bbh_movie_recommendation,
    "multistep_arithmetic_two":prompt_one_example_on_bbh_multistep_arithmetic_two,
    "navigate":prompt_one_example_on_bbh_navigate,
    "object_counting":prompt_one_example_on_bbh_object_counting,
    "penguins_in_a_table":prompt_one_example_on_bbh_penguins_in_a_table,
    "reasoning_about_colored_objects":prompt_one_example_on_bbh_reasoning_about_colored_objects,
    "ruin_names":prompt_one_example_on_bbh_ruin_names,
    "salient_translation_error_detection":prompt_one_example_on_bbh_salient_translation_error_detection,
    "snarks":prompt_one_example_on_bbh_snarks,
    "sports_understanding":prompt_one_example_on_bbh_sports_understanding,
    "temporal_sequences":prompt_one_example_on_bbh_temporal_sequences,
    "tracking_shuffled_objects_five_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_five_objects,
    "tracking_shuffled_objects_seven_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_seven_objects,
    "tracking_shuffled_objects_three_objects":prompt_one_example_on_bbh_tracking_shuffled_objects_three_objects,
    "web_of_lies":prompt_one_example_on_bbh_web_of_lies,
    "word_sorting":prompt_one_example_on_bbh_word_sorting,
    
}
all_resultsp = {}

def first_capital_postprocess(text, candidate=None):
    if candidate is None:
        for t in text:
            if t.isupper():
                return t
    else:
        for t in text:
            if t.isupper() and t in candidate:
                return t
    return ''

def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids


def build_prompt(tokenizer, template, query, history, system=None, verbose=False, postfix=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                if verbose:
                    print(system_text)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            if verbose:
                print(message)
            # if role == 'user':
            #     message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            # else:
            #     message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
        if postfix is not None:
            tokens = tokenizer.encode(postfix, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if 'gemma' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})
        print("add special tokens")

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def main():
    # 使用合并后的模型进行推理
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Arguments for running the model with specific configurations.")

    # Add arguments
    parser.add_argument("--model_name_or_path", type=str, default="/data/home/liuyijiang/workspace/mmlab/LLaMA2-Accessory/checkpoints/gemma-2b-it", help="Path to the model directory.")
    parser.add_argument("--template_name", type=str, default="gemma", help="Name of the template to use.")
    parser.add_argument("--adapter_name_or_path", type=str, default=None, help="Path to the adapter or name of the adapter.")
    parser.add_argument("--datasets", default=None, nargs="+", metavar="DATASET", help="List of datasets")
    parser.add_argument("--chat", type=str, default=None, help="[debug-all]")
    parser.add_argument("--verbose", action="store_true", help="")
    parser.add_argument("--ft_mode", type=str, choices=["none", "dimdown"], help="")
    parser.add_argument("--trainable_mask", action="store_true", help="set mask trainable with sigmoid regularizer")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="")
    parser.add_argument("--identity_loss", default=False, action="store_true", help="add router mapper")
    parser.add_argument("--ppl", action="store_true", help="")

    # Parse the arguments
    args = parser.parse_args()

    external_cfg = {
        "ft_mode": args.ft_mode,
        "global_step": 0,
        "dimdown_dim": 0,
        "trainable_mask": args.trainable_mask,
        "identity_loss": args.identity_loss,
    }
    with open("global_external_config.json","w") as fp:
        json.dump(external_cfg, fp)

    template_name = args.template_name
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = args.adapter_name_or_path

    template = template_dict[template_name]
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = args.max_new_tokens
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()

    # for name, module in model.named_modules():
    #     if hasattr(module, "init_parameters"):
    #         module.init_parameters()
    #         print("module.init_parameters()",name)
    dimdown_tensors = {}
    if adapter_name_or_path is not None:
        with safe_open(os.path.join(adapter_name_or_path,"adapter_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "dimdown" in key:
                    logger.info(f"Load dimdown weights! {key}")
                    dimdown_tensors[key] = f.get_tensor(key)
        dimdown_loading_status = model.load_state_dict(dimdown_tensors, strict=False)
        print(dimdown_loading_status)
    for name, module in model.named_modules():
        if hasattr(module, "inference_only"):
            print(name, "has inference_only")
            setattr(module, "inference_only", True)
            module.step(0)

    if "bloom" in model_name_or_path:
        tokenizer = load_tokenizer(model_name_or_path)
    else:
        tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
        assert len(stop_token_id) == 1
        stop_token_id = stop_token_id[0]
    
    if 'internlm2' in model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    print(tokenizer.__dict__)

    model.to(torch.bfloat16)
    model.eval()

    if args.chat is not None:
        system = "You are a helpfull assistant."
        prompt = "who is the president of usa?"
        while 1:
            inference(prompt, model, tokenizer, template, 100, stop_token_id, "None", True, system)
            prompt = args.chat if args.chat.startswith("debug-") else input("> ")
            if prompt.strip() == "fromfile":
                with open("fromfile.json","r") as fp:
                    content = json.load(fp)
                    prompt = content["text"]
                    system = content.get("system",None)
            elif prompt.strip()=="debug":
                import ipdb;ipdb.set_trace()

            elif prompt.strip().startswith("debug-"):
                dataset = prompt.strip().split("debug-")[-1]
                dataset = dataset.split(",")
                args.chat = 'debug'
                root = "dataset_eval"
                results = {}
                for folder in sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root,f))]):
                    if "all" in dataset:
                        pass
                    elif any(d in folder for d in dataset):
                        pass
                    else:
                        continue
                    if os.path.isdir(os.path.join(root,folder)):
                        correct_list = []
                        for fn in os.listdir(os.path.join(root,folder)):
                            with open(os.path.join(root,folder,fn),'r') as fp:
                                content = json.load(fp)
                            max_new_tokens = 5
                            pbar = tqdm(content.items(), desc=f"{fn},ppl={args.ppl}")
                            golds = [val["gold"] for key,val in content.items()]
                            golds = list(set(golds))
                            golds_ = str(['A','B','C','D','E','F','G','H'][:len(golds)]).replace("[","").replace("]","")
                            for key,val in pbar:
                                prompt = f"Answer the question by replying the index of the correct choice, which should be one of {golds_}.\n" + val["origin_prompt"]
                                prompt = prompt + "\nThe index of the answer is"
                                # postfix = f"The index ({golds_}) of the answer is"
                                postfix = None
                                gold = val["gold"]
                                system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                                response = inference(prompt, model, tokenizer, template, max_new_tokens, 
                                                     stop_token_id, dataset="None", 
                                                     verbose=False, system=system, return_raw=True,
                                                     postfix=postfix,
                                                     ppl=args.ppl, golds=golds)
                                pred = first_capital_postprocess(response, candidate=golds)
                                correct = pred == gold
                                correct_list += [correct]
                                pbar.set_postfix({
                                    "Acc":round(np.mean(correct_list),5)
                                })
                                if pred=="":
                                    print(round(np.mean(correct_list),5), response)
                                else:
                                    pass
                                    # print(pred)
                        results[folder] = round(np.mean(correct_list),5)
                        print(results)
                        for key,val in results.items():
                            print(f"{val}")
                        print("#"*40)


def inference(query, model, tokenizer, template, max_new_tokens, stop_token_id, dataset, verbose, system=None, return_raw=False,
              postfix=None, ppl=False, golds=None):

    input_ids = build_prompt(tokenizer, template, query, [], system=system, verbose=verbose, postfix=postfix).to(model.device)
    model.generation_config.do_sample=False
    model.generation_config.num_beams=1
    if ppl:
        assert len(golds)<=4, "golds should be less than 4"
        with torch.no_grad():
            logits = model(input_ids, return_dict=True).logits[0,-1,:]
        # score = [logits[ii] for ii in [319, 350, 315, 360][:len(golds)]]
        # import ipdb;ipdb.set_trace()
        score = [logits[ii] for ii in tokenizer.encode(" A B C D")[-4:][:len(golds)]]
        response = ["A","B","C","D"][:len(golds)][torch.vstack(score).argmax()]
        return response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, 
            do_sample=False, num_beams=1,
            top_p=None, temperature=None, 
            eos_token_id=stop_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # import ipdb;ipdb.set_trace()
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.split(tokenizer.eos_token)[0]
    if verbose:
        print(response)
    return response

if __name__ == '__main__':
    main()

