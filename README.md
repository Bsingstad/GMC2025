# Cha-Cha-Chagas: Auxiliary Pretraining and Fine-Tuning Across Heterogeneous Datasets for ECG-Based Chagas Disease Detection

This repository contains the **Python implementation** for our entry in the **George B. Moody PhysioNet Challenge 2025**, titled *"Auxiliary Pretraining and Fine-Tuning Across Heterogeneous Datasets for ECG-Based Chagas Disease Detection"*.

Our submission was developed by the **Cha-Cha-Chagas** team and builds upon the official [PhysioNet Challenge 2025 Python example repository](https://github.com/physionetchallenges/python-example-2025), expanding it with deep learning–based architectures, auxiliary pretraining, and fine-tuning strategies for improved ECG-based Chagas disease detection.

---

## 🧠 Overview

Chagas disease (American trypanosomiasis) is a parasitic infection caused by *Trypanosoma cruzi*, soemtimes leading to **chronic Chagas cardiomyopathy (CCC)**. Detection from ECG signals remains challenging due to the scarcity of high-quality labeled data.

Our approach investigates whether **auxiliary pretraining on weakly labeled ECG data** (from the large CODE-15% dataset) can improve downstream Chagas detection when **fine-tuned on datasets with stronger labels** such as SaMi-Trop (serologically confirmed positives) and PTB-XL (assumed negatives).

Despite the hypothesis, experiments revealed that this pretraining strategy **did not outperform conventional supervised training**, underscoring the importance of dataset balance, label reliability, and domain similarity in multi-dataset ECG modeling.

---

## 📁 Repository Structure

* `train_model.py` — Wrapper for training.
* `run_model.py` — Wrapper for inference.
* `evaluate_model.py` — Used for local validation with the official PhysioNet evaluation code.
* `team_code.py` — **Main implementation** containing:

  * Deep neural network architecture (`Net1D`).
  * Dataset handling (`ECGDataset`).
  * Auxiliary pretraining and fine-tuning logic.
  * Model saving/loading utilities.

As instructed by the Organizers, we did **not** modify the official PhysioNet scripts (`train_model.py`, `run_model.py`, `helper_code.py`). Only `team_code.py`.

---

## 🧩 Method Summary

### Data Sources

We used the following datasets:

* **CODE-15%**: Large Brazilian ECG dataset with weak self-reported Chagas labels.
* **SaMi-Trop**: Serologically verified Chagas cohort (positive cases only).
* **PTB-XL**: German ECG dataset (assumed Chagas-negative controls).

### Data Processing

* Resampling to **400 Hz**.
* Fixed window of **7 seconds** (shortest common duration).
* All **12 standard ECG leads** used.
* Resampling and zero-padding handled dynamically during dataset loading.
* Signals and metadata read using **WFDB** package.

### Model Architecture

We implemented a **1D convolutional neural network (Net1D)** inspired by the *InceptionTime* architecture:

* Residual connections for stable training.
* Variable receptive fields for multi-scale temporal feature extraction.
* Integrated **Squeeze-and-Excitation** blocks.
* Final linear classifier for binary Chagas prediction.

### Training Strategy

#### 1. Auxiliary Pretraining

* Model pretrained on **CODE-15%**.
* Auxiliary tasks included demographic and rhythm-related ECG features.

#### 2. Fine-Tuning

* Model fine-tuned using **SaMi-Trop** (positive) and **PTB-XL** (negative) data.
* Used **balanced mini-batches** via a `WeightedRandomSampler`.
* Optimization: Adam (lr=1e-4), weight decay=1e-5.
* Loss: Binary cross-entropy (BCEWithLogitsLoss).
* Training for 5 epochs per phase, with early stopping based on validation AUROC.

---

## 🧪 Results Summary

| Data used for pretraining | Data used for fine-tune or conventional training | AUROC on internal validation data | AUPRC on internal validation data | Challenge score on hidden validation data | REDS-II (hidden test) | SaMi-Trop (hidden test) | ELSA-Brasil (hidden test) |
| ------------------------- | ------------------------------------------------ | --------------------------------- | --------------------------------- | ----------------------------------------- | ------- | --------- | ----------- |
| CODE-15%                  | SaMi-Trop and PTB-XL                             | 0.69                              | 0.22                              | 0.040                                     | –       | –         | –           |
| –                         | SaMi-Trop and PTB-XL                             | –                                 | –                                 | 0.066                                     | –       | –         | –           |
| –                         | CODE-15%                                         | –                                 | –                                 | 0.143                                     | –       | –         | –           |
| –                         | SaMi-Trop and CODE-15%                           | 0.81                              | 0.41                              | 0.316                                     | 0.296   | 0.247     | 0.136       |

**Key Finding:** Auxiliary pretraining using weak CODE-15% labels did **not** generalize better than direct supervised training on the development data nor the hidden validation. Therefore the best performing model, using conventional training on CODE-15% and SaMi-Trop, were applied on the final hidden test set (only one attempt). A possible cause of the unexpectedly poor results may be domain shift between datasets (e.g., Brazilian vs. German populations), which likely introduced dataset bias during fine-tuning.

---

## 🚀 Running the Code

### 1. Installation

Create a virtual environment or use Docker. Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training

```bash
python train_model.py -d training_data -m model -v
```

### 3. Inference

```bash
python run_model.py -d holdout_data -m model -o holdout_outputs -v
```

### 4. Evaluation

Clone and run the official evaluation repo:

```bash
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```

### 5. Docker (Recommended)

To ensure reproducibility:

```bash
docker build -t chagas_image .
docker run -it -v ~/example/training_data:/challenge/training_data \
               -v ~/example/model:/challenge/model \
               -v ~/example/holdout_data:/challenge/holdout_data \
               -v ~/example/holdout_outputs:/challenge/holdout_outputs chagas_image bash
```

---

## ⚙️ Model Loading and Saving

* **Save:** The model is stored as `model.keras.h5` inside the model folder.
* **Load:** Automatically reloaded via `load_model()` in `team_code.py`.

---

## 📊 Limitations and Future Work

* **Dataset Bias:** Differences in acquisition equipment, population, and temporal context introduce confounders.
* **Label Quality:** CODE-15% self-reported labels remain unreliable for fine-grained learning.
* **Future Directions:** Domain adaptation, adversarial debiasing, or contrastive pretraining could enhance generalization across diverse ECG datasets.

---

## 🧩 References

This implementation accompanies the paper:

> **Auxiliary Pretraining and Fine-Tuning Across Heterogeneous Datasets for ECG-Based Chagas Disease Detection**
> Bjørn-Jostein Singstad, Nikolai Olsen Eidheim, Amila Ruwan Guruge, Ola Marius Lysaker, Vimala Nunavath.
> University of South-Eastern Norway & Akershus University Hospital, 2025.
> [GitHub Repository](https://github.com/Bsingstad/GMC2025)

---

## 🧑‍💻 Authors

**Cha-Cha-Chagas Team:**

* Bjørn-Jostein Singstad
* Nikolai Olsen Eidheim
* Amila Ruwan Guruge
* Ola Marius Lysaker
* Vimala Nunavath

---

## Useful Links

* [PhysioNet Challenge 2025](https://physionetchallenges.org/2025/)
* [Evaluation Code Repository](https://github.com/physionetchallenges/evaluation-2025)
* [Official Discussion Forum](https://groups.google.com/forum/#!forum/physionet-challenges)
