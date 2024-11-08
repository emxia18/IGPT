from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_reply(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Example usage:
input_text = "Name of sender: Can you meet me at 5?\n"
reply = generate_reply(input_text)
print("Reply:", reply)