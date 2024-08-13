import json
import re
import torch

def clean_format_for_list_tuple(s):
    if type(s)==list:
        s = s[0]
    elif type(s)==tuple:
        s = s[0]
    else:
        assert (type(s)==str or type(s)==int or type(s)==torch.Tensor), f"{type(s)}"
    return s

def prompt_one_example_on_boolq(batch, raw=False, ppl=False, ppl_label=None, few_shot=False):
    question = batch['question'][0]
    passage = batch['passage'][0]
    label = batch['label'][0]
    idx = batch["idx"][0].item()
    if raw:
        prompt = f"""Passage: {passage}
Question: {question}
Choices:
A. False
B. True
Answer:
"""
        gt = ["A","B"][label.item()]
        return prompt, gt, idx
    elif few_shot:
        prompt = f"""Passage: United States and the International Criminal Court -- The United States should have the chance to observe and assess the functioning of the court, over time, before choosing to become subject to its jurisdiction. Given these concerns, I will not, and do not recommend that my successor, submit the treaty to the Senate for advice and consent until our fundamental concerns are satisfied.
Question: does the icc has jurisdiction in the united states
Choices:
A. False
B. True
Answer:
A

Passage: {passage}
Question: {question}
Choices:
A. False
B. True
Answer:
"""
        gt = ["A","B"][label.item()]
        return prompt, gt, idx
    elif ppl:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
### Instruction:
## Task description
You are answering a true/false question.
You'll be presented with a "passage", accompanied by a "question or statement".
Your objective is to give the "answer" of the "question or statement" based on the information from the "passage".
The answer should be either "true" or "false".

## Task
### Passage:
{passage}
### Question:
{question}
### Answer (true or false):
{ppl_label}
"""
        gt = "true" if label.item()==1 else "false"
        return prompt, gt, idx
    else:
        prompt = f"""## Task description
You are answering a true/false question.
You'll be presented with a "passage", accompanied by a "question or statement".
Your objective is to give the "answer" of the "question or statement" based on the information from the "passage".
The answer should be either "true" or "false".
Before finalizing your answer, thoroughly analyze the information. 

## Task
### Passage:
{passage}
### Question or Statement:
{question}
### Analysis:
Let's think step-by-step."""

    gt = "true" if label.item()==1 else "false"
    return prompt, gt, idx

def prompt_one_example_on_cb(batch, raw=False, ppl=False, ppl_label=None):
    premise = batch['premise'][0]
    hypothesis = batch['hypothesis'][0]
    label = batch['label'][0]
    idx = batch["idx"][0].item()
    if raw:
        prompt = f"""Premise: {premise}
Hypothesis: {hypothesis}
Choices:
A. entailment
B. contradiction
C. neutral
Answer:
"""
        gt = ["A","B","C"][label]
        return prompt, gt, idx
    elif ppl:
        prompt = f"""## Task description
You are tasked with an analytical assessment.
You'll be presented with a "premise", accompanied by a "hypothesis".
Your objective is to answer the "entailment status", which will be either "entailment", "neutral", or "contradiction".

## Task
### premise:
{premise}
### hypothesis:
{hypothesis}
### entailment status ("entailment", "neutral", or "contradiction"):
{ppl_label}
"""
        if label.item()==1:
            gt = "contradiction"
        elif label.item()==0:
            gt = "entailment"
        elif label.item()==2:
            gt = "neutral"
        else:
            import ipdb;ipdb.set_trace()
        return prompt, gt, idx
    else:
        prompt = f"""## Task description
You are tasked with an analytical assessment.
You'll be presented with a "premise", accompanied by a "hypothesis".
Your objective is to answer the "entailment status", which will be either "entailment", "neutral", or "contradiction".
Before finalizing your answer, thoroughly analyze the information. 

## Task
### premise:
{premise}
### hypothesis:
{hypothesis}
### Analysis:
Let's think step-by-step."""

    if label.item()==1:
        gt = "contradiction"
    elif label.item()==0:
        gt = "entailment"
    elif label.item()==2:
        gt = "neutral"
    else:
        import ipdb;ipdb.set_trace()
    # gt = "contradiction" if label.item()==1 else "entailment"
    return prompt, gt, idx

def prompt_one_example_on_multirc(batch, raw=False, ppl=False, ppl_label=None):
    # import ipdb;ipdb.set_trace()
    paragraph = batch['paragraph'][0]
    question = batch['question'][0]
    answer = batch['answer'][0]
    label = batch['label'][0]
    idx = batch["idx"]["answer"][0].item()
    if raw:
        prompt = f"""Paragraph: {paragraph}
Question: {question}
Candidate answers: {answer}
Choices:
A. False
B. True
Answer:
"""
        gt = ["A","B"][label]
        return prompt, gt, idx
    elif ppl:
        prompt = f"""## Task description
You are assigned an analytical assessment.
You will receive a "paragraph" followed by a "question".
Additionally, an "answer" will be provided.
Your objective is to determine whether the provided "answer" is correct.
Respond with either "true" (if the answer is correct) or "false" (if the answer is incorrect).

## Task
### paragraph:
{paragraph}
### question:
{question}
### answer:
{answer}
### Is the answer correct?:
{ppl_label}
"""
        gt = "true" if label.item()==1 else "false"
        return prompt, gt, idx
    else:
        prompt = f"""## Task description
You are assigned an analytical assessment.
You will receive a "paragraph" followed by a "question".
Additionally, an "answer" will be provided.
Your objective is to determine whether the provided "answer" is correct.
Respond with either "true" (if the answer is correct) or "false" (if the answer is incorrect).
Before finalizing your response, thoroughly analyze the information.

## Task
### paragraph:
{paragraph}
### question:
{question}
### answer:
{answer}
### Analysis:
Let's think step-by-step."""

    gt = "true" if label.item()==1 else "false"
    return prompt, gt, idx

def prompt_one_example_on_wic(batch, raw=False,ppl=False, ppl_label=None):
    # import ipdb;ipdb.set_trace()
    sentence1 = batch['sentence1'][0]
    sentence2 = batch['sentence2'][0]
    word = batch['word'][0]
    label = batch['label'][0]
    idx = batch["idx"][0].item()
    if raw:
        prompt = f"""Sentence 1: {sentence1}
Sentence 2: {sentence2}
Is '{word}' in the above two sentenses the same?
Choices:
A. No
B. Yes
Answer:
"""
        gt = ["A","B"][label]
        return prompt, gt, idx
    elif ppl:
        prompt = f"""## Task description
Your task is to compare the meanings of a specific word as used in two distinct sentences. Here's a step-by-step breakdown:

1. You will receive two sentences: "sentence1" and "sentence2".
2. Within these sentences, a specific "word" will be highlighted.
3. Your objective is to give an answer labeled "Is the meaning consistent?", which will be either "yes" or "no".

## Task
### sentence1:
{sentence1}
### sentence2:
{sentence2}
### word:
{word}
### Is the meaning consistent?:
{ppl_label}
"""
        gt = "yes" if label.item()==1 else "no"
        return prompt, gt, idx
    else:
        prompt = f"""## Task description
Your task is to compare the meanings of a specific word as used in two distinct sentences. Here's a step-by-step breakdown:

1. You will receive two sentences: "sentence1" and "sentence2".
2. Within these sentences, a specific "word" will be highlighted.
3. Your objective is to give an answer labeled "Is the meaning consistent?", which will be either "yes" or "no".
4. Before finalizing your answer, thoroughly analyze the information.

## Task
### sentence1:
{sentence1}
### sentence2:
{sentence2}
### word:
{word}
### Is the meaning of the word consistent in two sentences?:
Let's think step-by-step.
"""

    gt = "yes" if label.item()==1 else "no"
    return prompt, gt, idx

def prompt_one_example_on_wsc(batch, raw=False,ppl=False, ppl_label=None):
    text = batch['text'][0]
    span1_text = batch['span1_text'][0]
    span2_text = batch['span2_text'][0]

    label = batch["label"][0]
    if raw:
        answer = ["A","B"][label]
        instruction = f"""{text}
Do '{span1_text}' and '{span2_text}' refer to the same entity in the above sentence?
Choices:
A. No
B. Yes
Answer:
"""
        return instruction, answer, -1
    else:
        raise NotImplementedError

def prompt_one_example_on_rte(batch, raw=False,ppl=False, ppl_label=None):
    premise = batch['premise'][0]
    hypothesis = batch['hypothesis'][0]
    # idx = batch["idx"][0]
    label = batch["label"][0]

    if raw:
        answer = ["A","B"][label]
        instruction = f"""Premise: {premise}
Hypothesis: {hypothesis}
Choices:
A. entailment
B. not entailment
Answer:
"""
        return instruction, answer, -1
    else:
        raise NotImplementedError

def prompt_one_example_on_copa(batch, raw=False,ppl=False, ppl_label=None):
    premise = batch['premise'][0]
    choice1 = batch['choice1'][0]
    choice2 = batch['choice2'][0]
    question = batch["question"][0]
    idx = batch["idx"][0].item()
    label = batch["label"][0]

    if raw:
        answer = ["A","B"][label]
        instruction = f"""Premise: {premise}
Question: What's the {question} for this?
Alternative 1: {choice1}
Alternative 2: {choice2}
Choices:
A. Alternative 1
B. Alternative 2
Answer:
"""
        return instruction, answer, idx
    else:
        raise NotImplementedError


def prompt_one_example_on_WinoGrande(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    example = batch
    sentence = example["sentence"][0]
    option1 = example["option1"][0]
    option2 = example["option2"][0]
    answer = example["answer"][0]

    choices_text = ""
    for label, text in zip(["A","B"], [option1,option2]):
        _ = f"{label}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Context: {sentence}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ['A',"B"][int(answer)-1]
    return instruction, output, -1

def prompt_one_example_on_openbookqa(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    example = batch
    question_stem = example["question_stem"][0]
    choices = example["choices"]
    answerKey = example["answerKey"][0]

    choices_text = ""
    for la, text in zip(choices["label"], choices["text"]):
        _ = f"{la[0]}. {text[0]}"
        choices_text += f"{_}\n" 

    instruction = f"""Question: {question_stem}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = answerKey[0]
    return instruction, output, -1

def prompt_one_example_on_anli(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    premise = example["premise"][0]
    hypothesis = example["hypothesis"][0]
    label = example["label"][0].item()

    choices_text = ""
    for la, text in zip(["A","B","C"], ["entailment", "neutral", "contradiction"]):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Premise: {premise}
Hypothesis: {hypothesis}
Question: What is the relation between the premise and hypothesis?
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ["A","B","C"][int(label)]
    return instruction, output, -1


def prompt_one_example_on_record(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    passage = example["passage"][0]
    query = example["query"][0]
    entities = example["entities"][0]
    entity_spans = example["entity_spans"]
    answers = example["answers"][0]

    choices_text = ""
    for text in entities:
        _ = f"{text}"
        choices_text += f"{_}\n" 

    instruction = f"""Passage: {passage}
Result: {query}
Question: What entity does \"@placeholder\" refer to in the result? 
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ""
    for an in answers:
        output += f"{an}\n"
    output = output.rstrip()
    return instruction, output, -1


def prompt_one_example_on_hellaswag(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    activity_label = example["activity_label"]
    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"][0]

    choices_text = ""
    for la, text in zip(["A","B","C","D"], endings):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Activity: {activity_label}
Context: {ctx}
Question: Which ending makes the most sense?
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ["A","B","C","D"][int(label)]
    return instruction, output, -1


def prompt_one_example_on_piqa(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    goal = example["goal"][0]
    sol1 = example["sol1"][0]
    sol2 = example["sol2"][0]
    label = example["label"][0].item()

    choices_text = ""
    for la, text in zip(["A","B"], [sol1, sol2]):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Goal: {goal}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ["A","B","C","D"][int(label)]
    return instruction, output, -1

def prompt_one_example_on_mmlu(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    question = batch["question"]
    choices = batch["choices"]
    answer = batch["answer"]
    # import ipdb;ipdb.set_trace()
    question = clean_format_for_list_tuple(question)
    answer = clean_format_for_list_tuple(answer)
    assert len(choices)==4
    prompt_template = """Question: {question}
Choices:
A. {A}
B. {B}
C. {C}
D. {D}
Answer:
"""
    instruction = prompt_template.format_map(
        {
            'question':question,
            'A':clean_format_for_list_tuple(choices[0]),
            'B':clean_format_for_list_tuple(choices[1]),
            'C':clean_format_for_list_tuple(choices[2]),
            'D':clean_format_for_list_tuple(choices[3]),
        }
    )
    gt = ["A","B",'C','D'][answer]
    output = f"{gt}"
    return instruction, output, -1

def prompt_one_example_on_arce(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    print(batch)
    example = batch
    question = example["question"]
    choices = example["choices"]
    answerKey = example["answerKey"]

    choices_text = ""
    for label, text in zip(choices["label"], choices["text"]):
        _ = f"{label}. {text}"
        choices_text += f"{_}\n" 
    instruction = f"""Question: {question}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = f"{answerKey}"
    return instruction, output, -1


def prompt_one_example_on_commonsense_qa(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    question = example["question"][0]
    question_concept = example["question_concept"][0]
    choices = example["choices"]
    answerKey = example["answerKey"][0]

    choices_text = ""
    for label, text in zip(choices["label"], choices["text"]):
        _ = f"{label[0]}. {text[0]}"
        choices_text += f"{_}\n" 

    instruction = f"""Concept: {question_concept}
Question: {question}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = answerKey
    return instruction, output, -1


def prompt_one_example_on_siqa(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # print(batch)
    example = batch
    context = example["context"][0]
    question = example["question"][0]
    answerA = example["answerA"][0]
    answerB = example["answerB"][0]
    answerC = example["answerC"][0]
    label = example["label"][0]

    choices_text = ""
    for la, text in zip(["A","B","C"], [answerA, answerB, answerC]):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Context: {context}
Question: {question}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = ["A","B","C","D"][int(label)-1]
    return instruction, output, -1


def prompt_one_example_on_bbh_boolean_expressions(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    choices_text = ""
    for la, text in zip(["A","B"], ["True","False"]):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""Question: {input}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = "A" if target=="True" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_causal_judgement(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    choices_text = ""
    for la, text in zip(["A","B"], ["Yes","No"]):
        _ = f"{la}. {text}"
        choices_text += f"{_}\n" 

    instruction = f"""{input}
Choices:
{choices_text.rstrip()}
Answer:
"""
    output = "A" if target=="Yes" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_date_understanding(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)

    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_disambiguation_qa(batch, raw=False,ppl=False, ppl_label=None):
    return prompt_one_example_on_bbh_date_understanding(batch, raw=raw,ppl=ppl, ppl_label=ppl_label)

def prompt_one_example_on_bbh_dyck_languages(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    instruction = f"""{input}
Answer:
"""
    output = target
    return instruction, output, -1

def prompt_one_example_on_bbh_formal_fallacies(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    instruction = f"""{input}
Choices:
A. valid
B. invalid
Answer:
"""
    output = "A" if target=="valid" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_geometric_shapes(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]

    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_hyperbaton(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_logical_deduction_five_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_logical_deduction_seven_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_logical_deduction_three_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_movie_recommendation(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_multistep_arithmetic_two(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    instruction = f"""Solve multi-step arithmetic problems.

Q: ((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =
Answer:
123

Q: ((-9 * 7 * 7 * -9) + (4 * -9 - 8 - -4)) =
Answer:
3929

Q: ((-3 + 5 * 8 * -4) - (9 - 8 * -7 + -9)) =
Answer:
-219

Q: {input}
Answer:
"""
    output = target
    return instruction, output, -1

def prompt_one_example_on_bbh_navigate(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    instruction = f"""{input}
Choices:
A. Yes
B. No
Answer:
"""
    output = "A" if target=="Yes" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_object_counting(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    
    instruction = f"""Questions that involve enumerating objects and asking the model to count them.

Question: I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?
Answer: 6

Question: I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?
Answer: 11

Question: I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?
Answer: 8

Question: {input}
Answer:
"""
    output = target
    return instruction, output, -1

def prompt_one_example_on_bbh_penguins_in_a_table(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_reasoning_about_colored_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_ruin_names(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_salient_translation_error_detection(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_snarks(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_sports_understanding(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Choices:
A. Yes
B. No
Answer:
"""
    output = "A" if target=="yes" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_temporal_sequences(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_tracking_shuffled_objects_five_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_tracking_shuffled_objects_seven_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_tracking_shuffled_objects_three_objects(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Answer:
"""
    output = target.replace("(","")
    output = output.replace(")","")
    return instruction, output, -1

def prompt_one_example_on_bbh_web_of_lies(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""{input}
Choices:
A. Yes
B. No
Answer:
"""
    output = "A" if target=="Yes" else "B"
    return instruction, output, -1

def prompt_one_example_on_bbh_word_sorting(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["input"][0]
    target = example["target"][0]
    pattern = r'\(([a-zA-Z])\) '
    replacement = r'\1. '
    input = re.sub(pattern, replacement, input)
    
    instruction = f"""Sort a list of words.

Question: Sort the following words alphabetically: List: oven costume counterpart
Answer: costume counterpart oven

Question: Sort the following words alphabetically: List: hypochlorite ponderosa phone credulity
Answer: credulity hypochlorite phone ponderosa

Question: Sort the following words alphabetically: List: newt arson parthia seismography mugho aspect census
Answer: arson aspect census mugho newt parthia seismography.

Question: {input}
Answer:
"""
    output = target
    return instruction, output, -1

def prompt_one_example_on_imdb(batch, raw=False,ppl=False, ppl_label=None):
    if not raw:
        raise NotImplementedError
    # import ipdb;ipdb.set_trace()
    example = batch
    input = example["text"][0]
    target = example["label"][0]
    
    instruction = f"""Based on the following review, make the binary sentiment classification.
Review: {input}
Choices:
A. Negative
B. Positive
Answer:
"""
    output = "A" if target.item()==0 else "B"
    return instruction, output, -1