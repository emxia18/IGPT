import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import joblib
import numpy as np

# Load the pre-trained LLM model and tokenizer (e.g., GPT-2)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
llm = GPT2LMHeadModel.from_pretrained(model_name)

# Load your Naive Bayes model
message_classifier = joblib.load("message_classifier_model.joblib")

def get_style_probability(text, classifier_model):
    """
    Use the Naive Bayes model to predict the probability that the text so far is in Bruno's style.
    """
    # Predict probability of being Bruno's style
    prob = classifier_model.predict_proba([text])[0][0]  # Bruno's probability
    return prob

def generate_response(prompt, max_length=50):
    """
    Generate a response based on combined LLM and Naive Bayes probabilities.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response = prompt

    for _ in range(max_length):
        # Get LLM logits for next token
        outputs = llm(input_ids)
        logits = outputs.logits[:, -1, :]  # Last token's logits

        # Get LLM probabilities and convert them to a numpy array
        llm_probs = torch.softmax(logits, dim=-1).detach().numpy().squeeze()
        
        # Initialize combined_probs list
        combined_probs = []

        # Go through each possible next token and calculate combined probability
        for idx, word_prob in enumerate(llm_probs):
            word = tokenizer.convert_ids_to_tokens([idx])[0]
            temp_response = response + word.replace("Ġ", " ")  # Adding space if needed
            
            # Calculate the Naive Bayes "style" probability for the expanded response
            style_prob = get_style_probability(temp_response, message_classifier)
            
            # Combine LLM and style probabilities (you can tweak the weighting here)
            combined_prob = word_prob * style_prob
            combined_probs.append(combined_prob)

        # Normalize combined probabilities
        combined_probs = np.array(combined_probs) / sum(combined_probs)

        # Sample the next token based on combined probabilities
        next_token_id = np.random.choice(range(len(combined_probs)), p=combined_probs)
        next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]

        # Stop generation if end of sentence
        if next_token == tokenizer.eos_token:
            break

        # Add token to response
        response += next_token.replace("Ġ", " ")
        print(response)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
    
    return response

# Example usage
prompt_text = "Hey, how's it going?"
response = generate_response(prompt_text)
print("Generated response:", response)
