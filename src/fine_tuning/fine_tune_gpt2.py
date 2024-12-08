import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
from accelerate import init_empty_weights
from transformers import LlamaForCausalLM


data_folder = "data/emily"
data_file = "data/emily/discord_messages.txt"
os.environ["WANDB_DISABLED"] = "true"
model_name = "gpt2-medium"
output_dir = f"./fine_tune/emily/{model_name}"
batch_size = 1
epochs = 3

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def load_and_tokenize_data(data_folder, tokenizer):
    inputs = []
    
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_path == data_file:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                conversation = f.read().strip()
                input_text = ""
                
                for pair in conversation.split("\n\n"):
                    if pair.strip():
                        formatted_text = pair
                        input_text += formatted_text

                if input_text:
                    inputs.append(input_text)

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    return tokenized_inputs

tokenized_data = load_and_tokenize_data(data_folder, tokenizer)

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

dataset = CustomTextDataset(tokenized_data)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="no",
    fp16=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(output_dir, safe_serialization=False)
tokenizer.save_pretrained(output_dir)

print("Model fine-tuning complete and saved to:", output_dir)