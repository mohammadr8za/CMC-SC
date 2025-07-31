# CMC-SC: Cross-Modal Contextualized ASR Spelling Correction

**ICNLSP 2025**

---

## Short Description

Code for **CMC-SC: Cross-Modal Contextualized ASR Spelling Correction via BERT and WavLM Using a Soft-Fusion Framework** accepted at ICNLSP 2025.

---

## Repository Contents

- `data/` - Scripts and instructions for downloading and preprocessing datasets.
- `models/` - Pretrained model checkpoints.
- `src/` - Source code for training, evaluation, and inference:
  - `train.py` - Model training script.
  - `evaluate.py` - Evaluation and metrics computation.
  - `infer.py` - Inference pipeline.
- `examples/` - Sample commands and example input/output.
- `requirements.txt` - Python dependencies.
- `LICENSE` - License information.
- `CITATION.cff` - Citation details for the paper.

---

## Requirements

- Python 3.8 or higher
- PyTorch >=1.10
- transformers
- datasets
- soundfile
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. Download the ASR transcripts and audio files following the instructions in `data/README.md`.
2. Run the preprocessing script:
   ```bash
   python src/data_preprocess.py --input_dir data/raw --output_dir data/processed
   ```

---

## Training

Train the CMC-SC model:

```bash
python src/train.py \
  --config configs/train.yaml \
  --data_dir data/processed \
  --output_dir models/CMC-SC
```


---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{peyghan2025cmc-sc,
  title={CMC-SC: Cross-Modal Contextualized ASR Spelling Correction via BERT and WavLM Using a Soft-Fusion Framework},
  author={Peyghan, Mohammad Reza and Amini, Sajjad and Ghaemmaghami, Shahrokh},
  booktitle={ICNLSP},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

