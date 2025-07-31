import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from models import BertTokenClassifier, BertTokenClassifier_LSTM
from custom_datasets import load_texts_and_labels, TokenDetectionDataset
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, dataset, thresholds=None, plot_confusion=True, device=DEVICE):
    """
    Evaluate a 2-logit BERT token classifier.

    Args:
      model       : your trained BertTokenClassifier
      dataset     : TokenDetectionDataset (with labels 0/1)
      thresholds  : list of prob thresholds for P(label=1) to sweep
      plot_confusion: whether to plot the best confusion matrix

    Returns:
      results: list of (threshold, precision, recall, f1, cm)
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.71, 0.05)

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    best_f1 = -1.0
    best_thresh = None
    best_cm = None
    best_report = None
    results = []

    with torch.no_grad():
        # Precompute all logits & labels for speed
        all_probs = []
        all_labels = []
        all_masks  = []

        for batch in tqdm(loader, desc="Gathering logits"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)  # (1, T, 2)
            probs  = torch.softmax(logits, dim=-1)                     # (1, T, 2)

            all_probs.append(probs.squeeze(0).cpu())           # (T,2)
            all_labels.append(labels.squeeze(0).cpu())         # (T,)
            all_masks.append(attention_mask.squeeze(0).cpu())  # (T,)

        # Stack across dataset
        all_probs  = torch.cat(all_probs, dim=0)  # (N_all_tokens, 2)
        all_labels = torch.cat(all_labels, dim=0) # (N_all_tokens,)
        all_masks  = torch.cat(all_masks, dim=0)  # (N_all_tokens,)

        # Keep only non-padding
        valid_idx = all_masks.bool()
        all_probs  = all_probs[valid_idx]
        all_labels = all_labels[valid_idx]

        # Extract P(label=1)
        p1 = all_probs[:,1]  # (N_valid,)

        # Sweep thresholds
        for thr in thresholds:
            preds = (p1 > thr).long().numpy()
            labs  = all_labels.numpy()

            cm = confusion_matrix(labs, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                labs, preds, average='binary'
            )
            results.append((thr, prec, rec, f1, cm))

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thr
                best_cm = cm
                best_report = classification_report(
                    labs, preds, target_names=["Correct (0)", "Incorrect (1)"]
                )

    # Print best
    print(f"\n📌 Best threshold: {best_thresh:.2f} (F1 = {best_f1:.4f})\n")
    print("Confusion Matrix:")
    print(best_cm)
    print("\nClassification Report:")
    print(best_report)

    if plot_confusion:
        plot_confusion_matrix(best_cm, ["Correct (0)", "Incorrect (1)"])

    return results

def plot_confusion_matrix(cm, classes):
    """
    cm: 2x2 confusion matrix
    classes: list of class names in order [pred0, pred1]
    """
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    # plt.savefig('')


if __name__ == "__main__":
    MAX_LEN = 128
    CHECKPOINT_DIR = 'checkpoints_040307'

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_model.pt')
    texts_test, labels_test = load_texts_and_labels("/home/eri/Documents/peyghan/ASR-EC/datasets/audio-dataset/en/whisper-outputs/filtered_asr_output_test_WER0.2852_CER0.1593.tsv")
    
    test_dataset   = TokenDetectionDataset(texts_test,   labels_test,   max_length=MAX_LEN)
    model     = BertTokenClassifier_LSTM().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path))
    results = evaluate_model(model, test_dataset, thresholds=[0.3,0.4,0.5,0.6], plot_confusion=True)
