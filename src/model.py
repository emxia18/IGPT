import os
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

# Set up paths and model parameters
data_folder = "data/bruno/processed_data_folder"
os.environ["WANDB_DISABLED"] = "true"
model_name = "gpt2"  # You can use "gpt2-medium" or other variants for better results
output_dir = f"./fine_tune/{model_name}"
batch_size = 2
epochs = 3

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Custom function to prepare data for language modeling
def load_and_tokenize_data(data_folder, tokenizer):
    inputs = []
    
    # Read all files in the data folder
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            conversation = f.read().strip()
            input_text = ""
            
            # Split on two newlines to get individual prompt-response pairs
            for pair in conversation.split("\n\n"):
                if pair.strip():  # Ensure we skip any blank segments
                    split_index = pair.find("Bruno Dumont:")
                    if split_index == -1:
                        prompt = pair
                        response = ""
                    else:
                        prompt = pair[:split_index]
                        response = pair[split_index:]
                    formatted_text = f"{prompt}\n{response}\n\n"  # Maintain the structure
                    input_text += formatted_text

            # Encode and add to the list of inputs
            if input_text:
                inputs.append(input_text)
    
    # Tokenize all inputs in a single batch
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    return tokenized_inputs

# Tokenize the dataset
tokenized_data = load_and_tokenize_data(data_folder, tokenizer)

# Prepare a dataset class to handle the data for Trainer
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

# Create the dataset
dataset = CustomTextDataset(tokenized_data)

# Set up training arguments
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
)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_dir, max_shard_size="2GB") 
tokenizer.save_pretrained(output_dir)

print("Model fine-tuning complete and saved to:", output_dir)