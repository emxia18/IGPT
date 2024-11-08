from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_path = "./fine_tune/emily/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained('./fine_tune/emily/gpt2-medium', from_tf=False)

def generate_reply(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Example usage:
input_text = [
    'tell me about yourself',
    'where are you from',
    'what are your passions',
    'who are your friends'
]

for text in input_text:
    reply = generate_reply(text)
    print("Reply:", reply)