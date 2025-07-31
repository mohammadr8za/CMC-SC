import torch
import torch.nn as nn
from transformers import BertTokenizer



# ─── Evaluation Utilities ─────────────────────────────────────────────────────────────────

def evaluate_correction(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer: BertTokenizer,
    device: torch.device,
    det_threshold: float,
    eval_batch_size: int = 1
):
    from torch.utils.data import DataLoader
    from sklearn.metrics import precision_recall_fscore_support

    loader = DataLoader(dataset, batch_size=eval_batch_size)
    all_true, all_pred = [], []
    origs, cors, refs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for k in ['input_ids','attention_mask','token_type_ids']:
                batch[k] = batch[k].to(device)
            speech = batch['speech_feats'].to(device)
            labels = batch['labels'].to(device)
            logits, alpha = model(
                batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], speech
            )
            pred_det = (alpha>det_threshold).long()
            mask = labels!= -100
            all_true += labels[mask].view(-1).cpu().tolist()
            all_pred += pred_det[mask].view(-1).cpu().tolist()
            orig_ids = batch['orig_ids']; tgt_ids = batch['target_ids']
            corr_ids = logits.argmax(-1).cpu()
            for o, c, r in zip(orig_ids, corr_ids, tgt_ids):
                origs.append(tokenizer.decode(o, skip_special_tokens=True))
                cors.append(tokenizer.decode(c, skip_special_tokens=True))
                refs.append(tokenizer.decode(r, skip_special_tokens=True))
    prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary')
    from evaluate_asr_with_bert import compute_alignment_and_wer, compute_cer
    wb, wa, cb, ca = [], [], [], []
    for o,c,r in zip(origs,cors,refs):
        wb.append(compute_alignment_and_wer(r.split(),o.split())[0])
        wa.append(compute_alignment_and_wer(r.split(),c.split())[0])
        cb.append(compute_cer(r,o)); ca.append(compute_cer(r,c))
    return {
        'detection': (prec,rec,f1),
        'wer': (sum(wb)/len(wb), sum(wa)/len(wa)),
        'cer': (sum(cb)/len(cb), sum(ca)/len(ca))
    }