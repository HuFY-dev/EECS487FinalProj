from transformers import (
    LlamaTokenizerFast,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from datasets import load_dataset, load_dataset, load_metric, Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge import Rouge
import pandas as pd
import re
from argparse import ArgumentParser
import wandb
from huggingface_hub import login as login_hf
import os
import json

parser = ArgumentParser()
parser.add_argument('--debug', action="store_true")
parser.add_argument('--control', action="store_true")
parser.add_argument('--model_path', default="meta-llama/Llama-2-7b-hf", required=False)
parser.add_argument('--batch_size', type=int, default=8, required=False)
parser.add_argument('--num_shots', type=int, default=2, choices=[1,2], required=False)
args = parser.parse_args()

DEBUG = args.debug
CONTROL = args.control
BATCH_SIZE = args.batch_size
NUM_SHOTS = args.num_shots

# Load LLaMA-2 with 4 bit quantization
llama_path = args.model_path
local_only = args.model_path != "meta-llama/Llama-2-7b-hf"
cache_path = os.path.join(os.getenv('TRANSFORMERS_CACHE'), "LLaMA-2-hf")
if not local_only:
    if not os.path.exists(cache_path):
        login_hf()
        _ = AutoModelForCausalLM.from_pretrained(llama_path, device_map="auto", load_in_8bit=True)
        del _
    llama_path = cache_path
llama_tokenizer = LlamaTokenizerFast.from_pretrained(
    llama_path,
    local_files_only=local_only,
)
llama_config = AutoConfig.from_pretrained(llama_path)
with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_config(llama_config)
empty_model.tie_weights()
bnb_quantization_config = BnbQuantizationConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4",
)
llama_model = load_and_quantize_model(
    empty_model, 
    weights_location=llama_path, 
    bnb_quantization_config=bnb_quantization_config, 
    device_map = "auto",
)
llama_tokenizer.padding_side = 'left'
llama_tokenizer.pad_token = llama_tokenizer.unk_token

# Load SQuAD2.0 dataset
dataset = load_dataset("squad_v2")
val_dataset = dataset['validation']

# Group validation by context
df = pd.DataFrame(val_dataset)
grouped_df = df.groupby("context").agg(list)
grouped_df.reset_index(inplace=True)
val_dataset_grouped = Dataset.from_pandas(grouped_df)

if DEBUG:
    val_dataset_grouped = val_dataset_grouped.select(range(25))
    
# Load In-context examples
with open("icl_examples.json", 'r') as file:
    examples = json.load(file)
icl_examples = [e["context"] + e["summary"] + e["quiz"] + "\n" for e in examples]
control_examples = [e["context"] + e["quiz"] + "\n" for e in examples]

# Prepare input to the model
def generate_batched_prompt(batch):
    text = [data["context"] for data in batch]
    if CONTROL:
        icl_set = control_examples[:NUM_SHOTS]
    else:
        icl_set = icl_examples[:NUM_SHOTS]
    icl_example = "".join(icl_set)
    prompts = [icl_example + "Context:\n" + t + "\n" for t in text]
    inputs = llama_tokenizer(
        prompts,
        padding=True, 
        return_tensors="pt"
    )
    return {
        "text": text,
        "prompts": prompts,
        "questions": [data["question"] for data in batch], # list[list[str]] | list[str]
        **inputs
    }

# Evaluation helper
rouge = load_metric("rouge")
def compute_metrics(pred: list[str], labels: list[str]):
    rouge_output = rouge.compute(
        predictions=pred, references=labels, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

val_grouped_dataloader = DataLoader(val_dataset_grouped, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batched_prompt)

wandb.init(
    project="EECS487FinalProj",
    config={
        "batch_size": BATCH_SIZE,
        "num_shots": NUM_SHOTS,
    }
)
table = wandb.Table(
    columns=[
        "pred",
        "best_match",
        "rouge2_score",
        "ground_truth",
        "prompt",
        "context",
        "generation",
    ]
)

# Evaluation
for dp in tqdm(val_grouped_dataloader):
    llama_model.eval()
    if DEBUG:
        print("Generating summary...")
        print(f'Number of input tokens: {len(dp["input_ids"][0])}')
        print("Prompt:")
        print(dp["prompts"][0] + "\n")
    # Generate token ids from prompt
    generated = llama_model.generate(
        input_ids=dp["input_ids"].to(llama_model.device), 
        attention_mask=dp["attention_mask"].to(llama_model.device),
        max_new_tokens=128,
    )
    # Decode token ids into tokens
    results = llama_tokenizer.batch_decode(generated[:, len(dp["input_ids"][0]):])
    if DEBUG:
        print("Generated:")
        print(results[0] + "\n")
    
    # Extract generated quiz via RegEx
    questions = [re.search("Question:\n(.+)", r).groups()[0] if re.search("Question:\n(.+)", r) is not None else None for r in results]
    if DEBUG:
        print("Generated question:")
        print(questions[0])
        print("Ground truth:")
        print(dp["questions"][0])
    
    # Log results
    for pred, trues, prompt, context, generation in zip(questions, dp["questions"], dp["prompts"], dp["text"],results):
        # Find best match from all possible ground truth quizzes
        if pred is None:
            continue
        best_score = {
            "rouge2_precision": -1,
            "rouge2_recall": -1,
            "rouge2_fmeasure": -1,
        }
        match = ""
        for true in trues:
            score = compute_metrics([pred], [true])
            if score["rouge2_fmeasure"] > best_score["rouge2_fmeasure"]:
                best_score = score
                match = true
        if DEBUG:
            print("Best score:")
            print(best_score)
        # Log to W&B table
        table.add_data(
            pred,
            match,
            best_score,
            trues,
            prompt,
            context,
            generation,
        )

# Upload data
wandb.log({"generated": table})
        