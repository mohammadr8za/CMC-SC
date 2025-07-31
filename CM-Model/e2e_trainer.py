import torch
import torch.nn as nn
import os
from evaluate import evaluate_correction
from models import MultiModalCorrectionModel
from correction_dataset import CorrectionDataset
from transformers import BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader

# HYPERP-ARAMERTERS
NUM_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints'
PLOT_DIR = 'plots'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def train_model(
    model, train_ds, dev_ds,
    tokenizer, optimizer, scheduler,
    device, num_epochs=5, det_threshold=0.5
):
    
    best_f1 = 0.0
    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        total_loss = 0
        for batch in train_loader:
            # prepare batch
            for k in ['input_ids','attention_mask','token_type_ids']:
                batch[k] = batch[k].to(device)
            speech = batch['speech_feats']
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits, alpha = model(
                batch['input_ids'], batch['attention_mask'],
                batch['token_type_ids'], speech
            )
            # detection loss
            det_logits = torch.stack([alpha,1-alpha],dim=-1).permute(0,2,1)
            loss_det = nn.CrossEntropyLoss(ignore_index=-100)(
                det_logits, labels
            )

            # correction loss
            target = batch['target_ids'].to(device)
            corr_logits = logits.permute(0,2,1)
            loss_corr = nn.CrossEntropyLoss(ignore_index=-100)(
                corr_logits, target
            )
            loss = loss_det + loss_corr
            # loss = loss_corr
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Dev evaluation
        eval_res = evaluate_correction(
            model, dev_ds, tokenizer, device, det_threshold
        )
        prec, rec, f1 = eval_res['detection']
        print(f"Epoch {epoch} | TrainLoss: {total_loss:.2f} | Dev F1: {f1:.4f} | WER: {eval_res['wer']} | CER: {eval_res['cer']}")
        os.makedirs('checkpoints', exist_ok=True)
        if f1>best_f1:
            best_f1=f1
            torch.save(model.state_dict(),'checkpoints/best_corr.pt')
        if scheduler: scheduler.step(f1)

# ───────────────────────────────────────────────────────────────────

def main():
    tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cm_model  = MultiModalCorrectionModel.to(DEVICE)
    optimizer = AdamW(cm_model.parameters(), lr=LR)

    train_tsv = '[YOUR_INPUT_TRAIN_DATA]'
    dev_tsv   = '[YOUR_INPUT_DEV_DATA]'
    
    train_ds  = CorrectionDataset(train_tsv=train_tsv, tokenizer=tokenizer, max_length=MAX_LEN)
    dev_ds  = CorrectionDataset(train_tsv=dev_tsv, tokenizer=tokenizer, max_length=MAX_LEN)

    train_model(model=cm_model,
                train_ds=train_ds,
                dev_ds=dev_ds,
                optimizer=optimizer,
                tokenizer=tokenizer,
                num_epochs=NUM_EPOCHS,
                det_threshold=0.5,
                device=DEVICE)



if __name__ == "__main__":
    main()
