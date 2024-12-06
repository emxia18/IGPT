import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import joblib
import numpy as np

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
llm = GPT2LMHeadModel.from_pretrained(model_name)

message_classifier = joblib.load("message_classifier_model.joblib")

def get_style_probability(text, classifier_model):

    prob = classifier_model.predict_proba([text])[0][0]
    return prob

def generate_response(prompt, max_length=50):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response = prompt

    for _ in range(max_length):
        outputs = llm(input_ids)
        logits = outputs.logits[:, -1, :]

        llm_probs = torch.softmax(logits, dim=-1).detach().numpy().squeeze()
        
        combined_probs = []

        for idx, word_prob in enumerate(llm_probs):
            word = tokenizer.convert_ids_to_tokens([idx])[0]
            temp_response = response + word.replace("Ġ", " ")
 
            style_prob = get_style_probability(temp_response, message_classifier)
            
            combined_prob = word_prob * style_prob
            combined_probs.append(combined_prob)

        combined_probs = np.array(combined_probs) / sum(combined_probs)

        next_token_id = np.random.choice(range(len(combined_probs)), p=combined_probs)
        next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]

        if next_token == tokenizer.eos_token:
            break

        response += next_token.replace("Ġ", " ")
        print(response)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
    
    return response

prompt_text = "Hey, how's it going?"
response = generate_response(prompt_text)
print("Generated response:", response)
