import torch.nn as nn
from transformers import BertModel
import torch


class BertTokenClassifier(nn.Module):
    """
    BERT + a 2-way token classifier head.
    Outputs a pair of logits at each token position.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        # Classification head: 2 output classes (0=correct, 1=incorrect)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1) Get BERT’s contextual embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # (B, T, H)

        # 2) Project to 2 logits per token
        logits = self.classifier(sequence_output)    # (B, T, 2)
        return logits

class BertTokenClassifier_LSTM(nn.Module):
    """
    BERT + BiLSTM + 2-way token classifier head.
    Outputs a pair of logits at each token position.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        # Load pretrained BERT (frozen)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,  # Hidden size per direction
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head (2*hidden_size because bidirectional)
        nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), 
                      nn.Linear(hidden_size, 2))
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1) Get BERT's contextual embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True 
        )
        sequence_output = outputs.last_hidden_state  # (B, T, H)
        
        # initial embeddings of words
        orig_embeds = outputs.hidden_states[0] # token embeddings, positional embeddings, and segment embeddings

        # input_embeds = self.bert.embeddings(
        #     input_ids=input_ids,
        #     token_type_ids=token_type_ids
        # )  # (B, T, H)
    
        cat_sequence_ouput = torch.cat((sequence_output, orig_embeds), 2)

        # 2) Process with BiLSTM
        lstm_output, _ = self.bilstm(cat_sequence_ouput)  # (B, T, 2*H)
        
        # 3) Project to 2 logits per token
        logits = self.classifier(lstm_output)  # (B, T, 2)
        return logits
    
