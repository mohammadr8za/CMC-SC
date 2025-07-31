# ─── Dataset Definition ─────────────────────────────────────────────────────────
from torch.utils.data import Dataset
import pandas as pd
import torch
import librosa
from transformers import BertModel, BertTokenizer, Wav2Vec2FeatureExtractor, WavLMModel
import os
import re
import string

# Regex to remove all punctuation
_PUNCT_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")

def normalize_text(text: str) -> str:
    """
    Lowercase, strip, remove punctuation, collapse whitespace.
    """
    text = (text or "").lower().strip()
    text = _PUNCT_REGEX.sub("", text)        # drop punctuation
    text = re.sub(r"\s+", " ", text)         # collapse multiple spaces
    return text

class CorrectionDataset(Dataset):
    """
    Dataset from TSV with columns:
      audio_name, target_text, transcription, wer, cer, alignment_list
    Prepares inputs for MultiModalCorrectionModel:
      - input_ids, attention_mask, token_type_ids
      - speech_feats (raw waveform tensors)
      - labels (0/1/-100 for detection)
      - orig_ids, target_ids for evaluation
    """
    def __init__(self, tsv_path, tokenizer, max_length=128,
                 audio_root_path='[Audio_Clips_Root_Path]'):
        self.df = pd.read_csv(tsv_path, sep='\t', dtype=str).fillna("")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.audio_root_path = audio_root_path
        # wav_reader: function(path)-> waveform tensor
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Paths & text
        path   = os.path.join(self.audio_root_path, row['audio_name'])
        ref    = normalize_text(row['target_text'])
        hyp    = normalize_text(row['transcription'])
        labels = row['alignment_list']  # string "[0,1,0,...]"

        # 1) Text tokenization
        enc = self.tokenizer(
            hyp,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        token_type_ids = enc.get('token_type_ids', torch.zeros_like(input_ids)).squeeze(0)

        # 2) Detection labels
        raw = eval(labels)
        label_tensor = torch.full((self.max_length,), -100, dtype=torch.long)
        N = min(len(raw), self.max_length-2)
        label_tensor[1:1+N] = torch.tensor(raw[:N], dtype=torch.long)

        # 3) orig vs target ids for evaluation
        orig_enc = self.tokenizer(ref,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        orig_ids   = orig_enc['input_ids'].squeeze(0)
        target_ids = enc['input_ids'].squeeze(0)  # corrected target == transcription?

        # 4) Speech waveform
        wav, _ = librosa.load(path, sr=16000)
        # speech_feats returned as list; handled in model

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels':         label_tensor,
            'orig_ids':       orig_ids,
            'target_ids':     target_ids,
            'speech_feats':   wav
        }

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sample_tsv = '[YOUR_INPUT_TRAIN_DATA]' # Default is TSV

    dataset = CorrectionDataset(tsv_path=sample_tsv,
                                tokenizer=tokenizer)
    
    print(dataset[0])
