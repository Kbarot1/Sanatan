import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        sequence_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

def get_bert_model(model_name='bert-base-uncased', num_labels=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertModel.from_pretrained(model_name).config
    config.num_labels = num_labels
    model = BertForQuestionAnswering(config)
    return model, tokenizer