from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu()

def calculate_embeddings(string_list):
    for string in string_list:
        embedding1 = get_bert_embedding(string[0])
        embedding2 = get_bert_embedding(string[1])

        embedding1_np = embedding1.numpy()
        embedding2_np = embedding2.numpy()

        similarity = cosine_similarity(embedding1_np, embedding2_np)
        print("Cosine Similarity:", similarity[0][0])

bot_list = [
    ["im doing p good, wbu?", "i'm doing well"],
    ["yes im so down, whats a good time for u, i can do whenever", "i would love to but i can only make it at 5:30 instead of 6:00"],
    ["yes yes where do u wanna meet", "i think that sounds like a great idea"],
    ["honestly just sleep and rest", "i like to just chill and do nothing"]
]

gpt_list = [
    ["im doing p good, wbu?", "pretty chill, hbu?"],
    ["yes im so down, whats a good time for u, i can do whenever", "omg yes, where r u thinking?"],
    ["yes yes where do u wanna meet", "maybe later, it sounds nice tho"],
    ["honestly just sleep and rest", "stay in bed and watch random stuff lol"]
]

print("=" * 20, "bot", "=" * 20)
calculate_embeddings(bot_list)
print("=" * 20, "gpt", "=" * 20)
calculate_embeddings(gpt_list)

