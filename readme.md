# CMC-SC: Cross-Modal Contextualized ASR Spelling Correction via BERT and WavLM using a Soft Fusion Framework

**ICNLSP 2025**

---

## Short Description

Code for **CMC-SC: Cross-Modal Contextualized ASR Spelling Correction via BERT and WavLM Using a Soft-Fusion Framework** accepted at ICNLSP 2025.

Models' checkpoints (Detection and Spelling Correction) will be added to this repository soon. 
---

## Repository Contents

- `Requirements/` - required packages to run the code in this repository.
- `Data-Processing/` - Source code for data pre-processing (to be used for both detection and CMC-SC modules.
- `CM-Model/` - Source code for training, and evaluation of the CM Spelling Correction module
- `Detection/` - Source code for training, and evaluation of the detection module.
- `CITATION.cff` - Citation details for the paper.

---

## Requirements

- Python 3.10 or higher

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. Download the Common Voice (En) Dataset and follow the instructions in `data/README.md`.
2. Run the preprocessing script:
   ```bash

   ```

---

## Detection

Train the CMC-SC model:

```bash

```

## CMC-SC

Train the CMC-SC model:

```bash

```


---

## Citation

If you use this code, please cite our paper:

```bibtex

```

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

