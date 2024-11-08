from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_path = "./fine_tune/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_reply(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Example usage:
input_text = [
    'Bruno Dumonts friend is having a conversation with him. Respond on his behalf.',
    'whens ur pwr 2',
    'im thinking of doing 229 and 171 and this robotics class called 123, at least thats the plan rn',
    'idk yet, im still deciding \n but i dont rly wanna do 229 and 111 \n like tgt \n and im p sure on 229 \n wbu',
    'i was also considering taking cs224w',
    'oh fuck yah \n bro i dont wanna overload tho ykwom',
]

for text in input_text:
    reply = generate_reply(text)
    print("Reply:", reply)