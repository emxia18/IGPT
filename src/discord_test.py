from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_fine_tuned_model(model_path, tokenizer_name):
    """
    Load the fine-tuned model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def test_model_as_emily(model, tokenizer, prompt, max_length=50, num_return_sequences=1):
    """
    Generate text based on the given prompt using the fine-tuned model, responding as Emily.
    """
    emily_prompt = "Emily: " + prompt
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    responses = generator(emily_prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return [response["generated_text"] for response in responses]

if __name__ == "__main__":
    # Specify the path to the fine-tuned model and tokenizer
    fine_tuned_model_path = "./fine_tune/emily/checkpoint-6507"  # Path to your fine-tuned model
    tokenizer_name = "gpt2"  # Same tokenizer used during fine-tuning

    # Load the model and tokenizer
    print("Loading fine-tuned model...")
    model, tokenizer = load_fine_tuned_model(fine_tuned_model_path, tokenizer_name)

    # Interactively test the model as Emily
    print("Testing the model as if it were Emily. Type 'exit' to quit.")
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == "exit":
            print("Exiting the testing script.")
            break

        # Generate predictions as Emily
        print("\nEmily's response(s):")
        response = test_model_as_emily(model, tokenizer, prompt, max_length=50, num_return_sequences=1)
        print(response)
