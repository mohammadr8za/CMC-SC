import torch.nn as nn
from transformers import BertModel
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import ast
import pandas as pd


class TokenDetectionDataset(Dataset):
    """
    Returns for each example:
      - input_ids      (LongTensor, [T])
      - attention_mask (LongTensor, [T])
      - token_type_ids (LongTensor, [T])
      - labels         (LongTensor, [T])  0 or 1
    """
    def __init__(self, texts, labels, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_seq = self.labels[idx]

        # 1) Tokenize (no special handling of labels here)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = enc['input_ids'].squeeze(0)        # (T)
        attn_mask = enc['attention_mask'].squeeze(0)   # (T)
        type_ids  = enc.get('token_type_ids',
                            torch.zeros_like(input_ids)).squeeze(0)

        # 2) Build label tensor, pad/truncate to max_length
        lbl = torch.zeros(self.max_length, dtype=torch.long)
        valid_len = min(len(label_seq), self.max_length)
        lbl[:valid_len] = torch.tensor(label_seq[:valid_len], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': type_ids,
            'labels': lbl
        }


def load_texts_and_labels(tsv_path: str, text_col: str = "transcription",
                          label_col: str = "alignment_list"):
    """
    Reads a TSV with columns [audio_name, target_text, transcription, wer, cer, alignment_list]
    and returns two lists:
      - texts: List[str] of the transcription strings
      - labels: List[List[int]] of the same length, each a list of 0/1 per token

    Rows where transcription is '<MISSING>' will yield:
      texts[i] = ''        (empty string)
      labels[i] = []       (empty list)

    Args:
      tsv_path:   path to the TSV file
      text_col:   name of the column holding the ASR output
      label_col:  name of the column holding the alignment_list (string repr of Python list)

    Returns:
      texts, labels
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")

    texts = []
    labels = []

    for _, row in df.iterrows():
        hyp = row[text_col].strip()
        lbl_str = row[label_col].strip()

        if hyp.upper() == "<MISSING>" or lbl_str.upper() == "NA":
            # No valid transcription → empty example
            texts.append("")
            labels.append([])
        else:
            # Keep the raw transcription string
            texts.append(hyp)
            # Parse the label list, e.g. "[0,1,0,0]" → [0,1,0,0]
            try:
                lbl_list = ast.literal_eval(lbl_str)
                if not isinstance(lbl_list, list):
                    raise ValueError
            except Exception:
                raise ValueError(f"Invalid alignment_list at row {_}: {lbl_str}")
            labels.append(lbl_list)

    return texts, labels
