from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DistilBertModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig
from datasets import load_dataset, load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from logging import getLogger
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from rouge import Rouge
from rouge import FilesRouge
from argparse import ArgumentParser

LEN_TRAIN = 20000
LEN_VAL = 25

parser = ArgumentParser()
parser.add_argument('--debug', action="store_true")
parser.add_argument('--model_name', default="allenai/led-base-16384", required=False)
parser.add_argument('--epochs', type=int, default=2, required=False)
parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--max_input_length', type=int, default=8192, required=False)
parser.add_argument('--max_output_length', type=int, default=512, required=False)
parser.add_argument('--num_beams', default=2, type=int, required=False)
parser.add_argument('--max_length', default=512, type=int, required=False)
parser.add_argument('--min_length', default=100, type=int, required=False)
parser.add_argument('--length_penalty', default=2.0, type=float, required=False)
parser.add_argument('--no_repeat_ngram_size', default=3, type=int, required=False)
parser.add_argument('--output_dir', default="./", required=False)
parser.add_argument('--logging_steps', default=5, type=int, required=False)
parser.add_argument('--eval_steps', default=10, type=int, required=False)
parser.add_argument('--save_steps', default=10, type=int, required=False)
parser.add_argument('--save_total_limit', default=2, type=int, required=False)
parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False)
args = parser.parse_args()
    
batch_size = args.batch_size

dataset = load_dataset("cnn_dailymail", '3.0.0')
train_dataset = dataset['train']
val_dataset = dataset['validation']

if args.debug:
    train_dataset = train_dataset.select(range(LEN_TRAIN))
    val_dataset = val_dataset.select(range(LEN_VAL))
    
# model_name = "t5-small"
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
led = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    gradient_checkpointing=True, 
    use_cache=False,
).to('cuda')

# set generate hyperparameters
led.config.num_beams = args.num_beams
led.config.max_length = args.max_length
led.config.min_length = args.min_length
led.config.length_penalty = args.length_penalty
led.config.early_stopping = True
led.config.no_repeat_ngram_size = args.no_repeat_ngram_size

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    output_dir=args.output_dir,
    logging_steps=args.logging_steps,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.epochs,
)

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=args.max_input_length,
    )
    outputs = tokenizer(
        batch["highlights"],
        padding="max_length",
        truncation=True,
        max_length=args.max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "highlights"],
)

val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["article", "highlights"],
)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

rouge = load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()