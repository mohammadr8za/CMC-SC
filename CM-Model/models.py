import torch.nn as nn
from transformers import BertModel
import torch
from transformers import BertModel, BertTokenizer, Wav2Vec2FeatureExtractor, WavLMModel
from wavlm.WavLM import WavLM, WavLMConfig
import librosa


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
    

class MultiModalCorrectionModel(nn.Module):
    """
    end-to-end error corrector using custom WavLM checkpoint:
      - Masks detected tokens
      - Pools speech embeddings from wavlm_tokens
      - Runs multi-modal BERT
      - Soft-fuses and decodes
    """
    def __init__(
        self,
        detector_ckpt: str = '[Detection_Module_Pretrained_Checkpoint]',
        bert_model_name: str = 'bert-base-uncased',
        wav_model_ckpt: str = 'wavlm/checkpoint/WavLM-Base.pt',
        max_speech_len: int = 50,
        tau: float = 0.5,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        # Text BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Detector (frozen)
        detector     = BertTokenClassifier_LSTM().to(device=device)
        detector.load_state_dict(torch.load(detector_ckpt))
        self.detector = detector
        # for p in self.detector.parameters(): p.requires_grad = False

        
        # wavLM model (frozen)
        # load the pre-trained checkpoints
        wav_lm_checkpoint = torch.load(wav_model_ckpt)
        self.cfg = WavLMConfig(wav_lm_checkpoint['cfg'])
        wav_model = WavLM(self.cfg)
        wav_model.load_state_dict(wav_lm_checkpoint['model'])
        self.wavlm = wav_model
        for p in self.wavlm.parameters(): p.requires_grad = False
        
        # Speech pooling + projection
        hidden_size = self.bert.config.hidden_size
        wav_hidden  = wav_lm_checkpoint['cfg']['encoder_embed_dim']
        self.wavpool = nn.AdaptiveAvgPool1d(max_speech_len)
        self.wavproj = nn.Linear(wav_hidden, hidden_size)
        
        # Threshold & vocab head
        self.tau = tau
        self.vocab_head = nn.Linear(hidden_size, self.bert.config.vocab_size)

    def wavlm_tokens(self, wav):
        wav_input_16khz = torch.from_numpy(wav).unsqueeze(0)

        if self.cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
        rep = self.wavlm.extract_features(wav_input_16khz)[0]
        return rep
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        waveforms: torch.Tensor  # shape (B, L) numpy or tensor
    ):
        B, T = input_ids.size()
        device = input_ids.device
        # Detection
        with torch.no_grad():
            det_logits = self.detector(input_ids, attention_mask, token_type_ids)
        alpha = torch.softmax(det_logits, dim=-1)[...,1]
        # Mask text embeddings
        E_orig = self.bert.embeddings.word_embeddings(input_ids)
        mask_id = self.tokenizer.mask_token_id
        mask_emb = self.bert.embeddings.word_embeddings(
            torch.full((B,1), mask_id, dtype=torch.long, device=device)
        ).expand(B, T, -1)
        specials = (input_ids==self.tokenizer.cls_token_id)|(input_ids==self.tokenizer.sep_token_id)
        to_mask = (alpha>self.tau)&~specials
        E_masked = torch.where(to_mask.unsqueeze(-1), mask_emb, E_orig)
        # WavLM tokens
        # assume waveforms is list of numpy arrays
        reps = []
        for w in waveforms: 
            rep = self.wavlm_tokens(w)  # (1, S, H_wav)
            reps.append(rep.squeeze(0))
        S = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)  # (B, S_max, H_wav)
        # Pool + project
        S = S.transpose(1,2)
        S_pool = self.wavpool(S)
        S_proj = self.wavproj(S_pool.transpose(1,2))
        # Multi-modal BERT
        all_embeds = torch.cat([E_masked, S_proj.to(device)], dim=1)
        speech_mask = torch.ones(B, S_proj.size(1), device=device, dtype=torch.long)
        full_mask = torch.cat([attention_mask, speech_mask], dim=1)
        seg_ids = torch.cat([token_type_ids, speech_mask], dim=1)
        outputs = self.bert(
            inputs_embeds=all_embeds,
            attention_mask=full_mask,
            token_type_ids=seg_ids,
            return_dict=True
        )
        C = outputs.last_hidden_state[:,:T,:]
        # Soft fusion
        E_fused = (1-alpha.unsqueeze(-1))*E_orig + alpha.unsqueeze(-1)*C
        # Decode
        logits = self.vocab_head(E_fused)
        return logits, alpha


if __name__ == "__main__":
    pass
