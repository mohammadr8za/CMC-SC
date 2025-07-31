import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import BertTokenClassifier, BertTokenClassifier_LSTM
from custom_datasets import TokenDetectionDataset, load_texts_and_labels


# === Configuration ===
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-5
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints'
PLOT_DIR = 'plots'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Data Preparation ===
# Assume texts and labels loaded from data_utils
texts_train, labels_train = load_texts_and_labels("/home/eri/Documents/peyghan/ASR-EC/datasets/audio-dataset/en/whisper-outputs/asr_output_train_WER0.2060_CER0.0861.tsv")
texts_dev, labels_dev = load_texts_and_labels("/home/eri/Documents/peyghan/ASR-EC/datasets/audio-dataset/en/whisper-outputs/asr_output_dev_WER0.2405_CER0.1050.tsv")

train_dataset = TokenDetectionDataset(texts_train, labels_train, max_length=MAX_LEN)
dev_dataset   = TokenDetectionDataset(texts_dev,   labels_dev,   max_length=MAX_LEN)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader    = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)


# --- Compute class weights for imbalance ---
total_tokens = sum([len(labels) for labels in labels_train])
pos_count = sum([sum(labels) for labels in labels_train]) + 1e-6
neg_count    = total_tokens - pos_count
# weight[class] — higher weight for the minority class
class_weight = torch.tensor([pos_count/total_tokens, neg_count/total_tokens])
# e.g. if pos<<neg then weight[1] > weight[0]
class_weight = class_weight.to(DEVICE)

# === Model, Optimizer, Loss ===
model     = BertTokenClassifier_LSTM().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

# CrossEntropyLoss with class weights (optional)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
# loss_fn = torch.nn.CrossEntropyLoss()

# === Tracking Metrics ===
history = {
    'train_loss': [], 'train_acc': [],
    'dev_loss': [],   'dev_acc': []
}
best_dev_acc = 0.0



# === Training and Evaluation Functions ===
def train_one_epoch(loader):
    model.train()
    running_loss = running_correct = running_tokens = 0
    for batch in tqdm(loader, desc='Training'):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        labels         = batch['labels'].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)  # (B,T,2)
        mask = attention_mask.bool()
        valid_logits = logits[mask]
        valid_labels = labels[mask]

        loss = loss_fn(valid_logits, valid_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * valid_labels.size(0)
        preds = valid_logits.argmax(dim=-1)
        running_correct += (preds == valid_labels).sum().item()
        running_tokens  += valid_labels.size(0)

    epoch_loss = running_loss / running_tokens
    epoch_acc  = running_correct / running_tokens
    return epoch_loss, epoch_acc


def evaluate(loader):
    model.eval()
    running_loss = running_correct = running_tokens = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask, token_type_ids)
            mask = attention_mask.bool()
            valid_logits = logits[mask]
            valid_labels = labels[mask]

            loss = loss_fn(valid_logits, valid_labels)
            running_loss += loss.item() * valid_labels.size(0)
            preds = valid_logits.argmax(dim=-1)
            running_correct += (preds == valid_labels).sum().item()
            running_tokens  += valid_labels.size(0)

    epoch_loss = running_loss / running_tokens
    epoch_acc  = running_correct / running_tokens
    return epoch_loss, epoch_acc



# === Main Training Loop ===
for epoch in range(1, NUM_EPOCHS+1):
    train_loss, train_acc = train_one_epoch(train_loader)
    dev_loss,   dev_acc   = evaluate(dev_loader)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['dev_loss'].append(dev_loss)
    history['dev_acc'].append(dev_acc)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
          f" | Dev Loss={dev_loss:.4f}, Dev Acc={dev_acc:.4f}")

    # Save best model
    # global best_dev_acc
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_model.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved new best model to {ckpt_path}")

# === Plotting ===
os.makedirs(PLOT_DIR, exist_ok=True)

epochs = list(range(1, NUM_EPOCHS+1))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, history['train_loss'], label='Train')
plt.plot(epochs, history['dev_loss'], label='Dev')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history['train_acc'], label='Train')
plt.plot(epochs, history['dev_acc'], label='Dev')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(PLOT_DIR, 'results.png'))
