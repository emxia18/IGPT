from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib
import numpy as np
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mosaicml/mpt-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache")


message_classifier = joblib.load("message_classifier_model.joblib")

naive_bayes_weight = 0.3
temperature = 0.7

def get_style_probability(text, classifier_model):
    prob = classifier_model.predict_proba([text])[0][0]
    return prob

def clean_text(text):
    text = re.sub(r'Ċ|Ġ', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

import torch
import numpy as np

def generate_response(prompt, max_length=50, use_bayes=True, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response = prompt

    for _ in range(max_length):
        # Get LLM's predictions
        outputs = llm(input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        llm_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()

        combined_probs = []
        eos_token_id = tokenizer.eos_token_id

        for idx, llm_prob in enumerate(llm_probs):
            if idx == eos_token_id:
                combined_probs.append(llm_prob)
            else:
                word = tokenizer.convert_ids_to_tokens([idx])[0]
                
                # Calculate Naive Bayes probability
                if use_bayes:
                    temp_response = response + word.replace("Ġ", " ")
                    style_prob = get_style_probability(temp_response, message_classifier)
                    # Adjust LLM probability using Naive Bayes output
                    combined_prob = llm_prob * ((1 - naive_bayes_weight) + naive_bayes_weight * style_prob)
                else:
                    combined_prob = llm_prob
                
                combined_probs.append(combined_prob)

        # Normalize combined probabilities
        combined_probs = np.array(combined_probs)
        combined_probs /= combined_probs.sum()

        # Select next token based on weighted probabilities
        next_token_id = np.random.choice(range(len(combined_probs)), p=combined_probs)

        # Check for end of sentence token
        if next_token_id == eos_token_id:
            break

        # Append token to the response
        next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]
        if next_token.strip() in ["Ċ", "Ġ", ""]:
            continue

        response += next_token.replace("Ġ", " ")
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
        print(response)
    response = clean_text(response)
    return response

prompt_text = "Hey, how's it going?"

response_with_bayes = generate_response(prompt_text, use_bayes=True)
print("Response with Naive Bayes:", response_with_bayes)

response_without_bayes = generate_response(prompt_text, use_bayes=False)
print("Response without Naive Bayes:", response_without_bayes)
