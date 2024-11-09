import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom Trainer to compute the loss manually
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss manually by using the model outputs and labels.
        """
        labels = inputs.get("labels")  # Get labels from inputs
        outputs = model(**inputs)  # Forward pass
        logits = outputs.get("logits")  # Get logits from the outputs

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute the loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding tokens
        )

        return (loss, outputs) if return_outputs else loss


def prepare_data(data_path, tokenizer, max_length=512):
    """
    Prepares and tokenizes the dataset for fine-tuning.
    """
    # Load the dataset from the JSONL file
    dataset = load_dataset("json", data_files={"train": data_path})
    limited_dataset = dataset["train"].select(range(10000))
    split_dataset = limited_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # Define the tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Tokenize and add labels for training
    tokenized_dataset = split_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})
    return tokenized_dataset


def fine_tune_model(model_name, tokenized_dataset, output_dir):
    """
    Fine-tunes a pre-trained causal language model on the given dataset.
    """
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if new tokens are added

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_total_limit=2,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        report_to="none",
    )

    # Use the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,  # Pass the tokenizer for padding and decoding
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()
    print(f"Fine-tuning complete. Model saved at: {output_dir}")


if __name__ == "__main__":
    # Specify the paths and model
    data_path = "data/emily/cleaned_discord_messages.jsonl"  # Path to the input dataset
    output_dir = "./fine_tune/emily"  # Directory to save the fine-tuned model
    model_name = "gpt2"  # Pre-trained model to fine-tune

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Step 1: Prepare the dataset
    print("Preparing data...")
    tokenized_dataset = prepare_data(data_path, tokenizer)

    # Step 2: Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_model(model_name, tokenized_dataset, output_dir)
