import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from itertools import islice

local_path = 'IGPT/data/bruno/new_processed_data.jsonl'
dataset = load_dataset('json', data_files=local_path, streaming=True)
print(dataset)

hf_token = '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'

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)

print("Model and tokenizer loaded successfully!")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    return tokenizer(examples["input"],
                     text_target=examples["output"],
                     truncation=True,
                     padding='max_length',
                     max_length=256)

tokenized_dataset = dataset['train'].map(tokenize_function, batched=True)

train_subset = list(islice(iter(tokenized_dataset), 10000))
train_subset = Dataset.from_list(train_subset)
eval_subset = list(islice(iter(tokenized_dataset), 2000))
eval_subset = Dataset.from_list(eval_subset)

print(train_subset[0])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("IGPT/fine_tune/saved_model_lora")
tokenizer.save_pretrained("IGPT/fine_tune/saved_model_lora")

input_text = "Hello, how are you doing?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

device = model.device
input_ids = input_ids.to(device)

output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
