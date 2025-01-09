import torch
import torch.nn as nn
from transformers import RagSequenceForGeneration, RagTokenizer

def get_rag_model(model_name="facebook/rag-token-base"):
    tokenizer = RagTokenizer.from_pretrained(model_name)
    model = RagSequenceForGeneration.from_pretrained(model_name)
    return model, tokenizer

def get_bert_model(model_name="bert-base-uncased"):
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_gpt2_model(model_name="gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def freeze_model(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze

def unfreeze_model(model):
    freeze_model(model, freeze=False)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model_class, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    return model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)