import os
import argparse
import re
import string
import pandas as pd
from difflib import SequenceMatcher
from transformers import BertTokenizer

# ─── Text Normalization ───────────────────────────────────────────────────────────

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

# ─── Alignment & Error‐Rate Computation ────────────────────────────────────────────

def compute_alignment_and_wer(ref_tokens, hyp_tokens):
    """
    Align two lists of BERT tokens and compute WER.
    Returns:
      wer: float
      labels: List[int] over hyp_tokens (0=correct,1=sub/ins)
    """
    N_ref = len(ref_tokens)
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens)

    subs = ins = dels = 0
    labels = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            labels.extend([0] * (j2 - j1))
        elif tag == 'replace':
            subs += max(i2 - i1, j2 - j1)
            labels.extend([1] * (j2 - j1))
        elif tag == 'insert':
            ins += (j2 - j1)
            labels.extend([1] * (j2 - j1))
        elif tag == 'delete':
            dels += (i2 - i1)
        else:
            raise ValueError(f"Unknown tag {tag}")

    wer = (subs + ins + dels) / N_ref if N_ref > 0 else 0.0
    return wer, labels

def compute_cer(ref: str, hyp: str):
    """
    Compute character error rate between two normalized strings.
    """
    r_chars = list(ref)
    h_chars = list(hyp)
    N_ref = len(r_chars)
    matcher = SequenceMatcher(None, r_chars, h_chars)

    subs = ins = dels = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            subs += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            ins += (j2 - j1)
        elif tag == 'delete':
            dels += (i2 - i1)
        # 'equal' adds no errors
    cer = (subs + ins + dels) / N_ref if N_ref > 0 else 0.0
    return cer

# ─── Main Evaluation Function ─────────────────────────────────────────────────────

def evaluate_tsv(input_tsv, output_dir=None):
    # Load inference results
    df = pd.read_csv(input_tsv, sep="\t", dtype=str)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    results = []
    total_err_tokens = 0.0
    total_ref_tokens = 0
    total_err_chars  = 0.0
    total_ref_chars  = 0

    for _, row in df.iterrows():
        audio = row['audio_name']
        raw_ref = row.get('target_text', "")    or ""
        raw_hyp = row.get('transcription', "")  or ""

        # Normalize both
        ref = normalize_text(raw_ref)
        hyp = normalize_text(raw_hyp)

        if hyp == '<missing>':
            wer = 'ND'
            cer = 'ND'
            labels = 'NA'
        else:
            # Token-level WER & alignment
            ref_tokens = tokenizer.tokenize(ref)
            hyp_tokens = tokenizer.tokenize(hyp)
            wer_val, label_list = compute_alignment_and_wer(ref_tokens, hyp_tokens)

            wer = f"{wer_val:.4f}"
            total_err_tokens += wer_val * len(ref_tokens)
            total_ref_tokens += len(ref_tokens)

            # Character-level CER
            cer_val = compute_cer(ref, hyp)
            cer = f"{cer_val:.4f}"
            total_err_chars += cer_val * len(ref)
            total_ref_chars += len(ref)

            # Format label list
            labels = "[" + ",".join(str(l) for l in label_list) + "]"

        results.append({
            'audio_name':     audio,
            'target_text':    raw_ref,
            'transcription':  raw_hyp,
            'wer':            wer,
            'cer':            cer,
            'alignment_list': labels
        })

    # Overall metrics
    overall_wer = (total_err_tokens / total_ref_tokens) if total_ref_tokens > 0 else 0.0
    overall_cer = (total_err_chars  / total_ref_chars)  if total_ref_chars  > 0 else 0.0
    print(f"Overall WER = {overall_wer:.4f}")
    print(f"Overall CER = {overall_cer:.4f}")

    # Save output TSV
    base, ext = os.path.splitext(os.path.basename(input_tsv))
    out_name = f"{base}_WER{overall_wer:.4f}_CER{overall_cer:.4f}{ext}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_name = os.path.join(output_dir, out_name)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_name, sep="\t", index=False)
    print(f"Wrote detailed results to {out_name}")

# ─── CLI Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ASR TSV with BERT-token alignment, normalization, WER & CER"
    )
    parser.add_argument("--input_tsv", help="TSV with columns audio_name,target_text,transcription")
    parser.add_argument("--output_dir", help="Directory to save output TSV", default=None)
    args = parser.parse_args()

    evaluate_tsv(args.input_tsv, args.output_dir)
