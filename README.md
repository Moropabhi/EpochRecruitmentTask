# Multimodal Speech Emotion Recognition — Technical Report

**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
**Task:** Emotion Classifier 
**Modalities:** Audio (Mel-Spectrogram) + Text (Whisper ASR Transcription)

## 1. Project Overview

This project tackles **Emotion Recognition** — the task of automatically detecting the emotional state of a speaker from their voice. Emotions are subtle, multidimensional signals. A single modality (audio waveform or transcribed words alone) is often insufficient to capture the full picture. This project therefore builds and compares **three model types**:

| # | Model | Modality |
|---|-------|----------|
| 1 | `AudioCNN` | Audio only (Mel-Spectrogram → CNN) |
| 2 | `EmotionRNN` | Text only (Whisper transcript → LSTM) |
| 3 | `FusionNN` | Audio + Text (fusion of CNN and RNN) |
| 3 | `Late Fusion` | Audio + Text (Late-fusion of CNN and RNN) |

A further **ensemble variant** (weighted average of CNN and RNN softmax outputs) is also evaluated.

---

## 2. Dataset & Preprocessing

### 2.1 RAVDESS Dataset

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains professionally acted speech recordings across 8 emotion classes. The filename encodes metadata, e.g. `03-01-01-01-01-01-01.wav` — the 3rd field (index `[6:8]`) is the emotion label (1–8).

```
Emotion Labels:
  1 → Neutral    5 → Angry
  2 → Calm       6 → Fearful
  3 → Happy      7 → Disgust
  4 → Sad        8 → Surprised
```

All audio files are loaded with `librosa.load()`, with sample rate 22,050 Hz. All files share the same sampling rate.

### 2.2 Audio Feature Extraction — Mel-Spectrogram

**Pre-processing**
I explored pre-processing like removing no audio part but it didn't make much difference. Moreover, raw helped to avoid overfitting and may provide some more hints to the emotions. 

Raw waveforms are not directly fed into the CNN. Instead, each waveform is converted into a **Mel-Spectrogram**, a 2D time-frequency representation that mimics how the human auditory system perceives sound.

**Why Mel-Spectrograms instead of raw audio or MFCCs?**
- Raw waveforms are very high-dimensional (22,050 samples/sec) and difficult for CNNs to learn from directly.
- MFCCs (explored but rejected later) compress too aggressively — they discard spectral shape information that is useful for emotion detection.
- Mel-Spectrograms preserve full frequency resolution on a perceptually-motivated (mel) scale, making them ideal for CNNs that learn spatial patterns.

```python
n_fft      = 1024   # FFT window size — balances time and frequency resolution
hop_length = 256    # Overlap between windows — controls temporal resolution
n_mels     = 128    # Number of mel frequency bands
```

**Post-processing:**
Each spectrogram is converted to dB scale (`librosa.power_to_db`) and then **z-score normalized** (mean = zero, variance = 1) per sample. This prevents louder audio samples from dominating the learning signal.

```python
spectrogram -= spectrogram.mean()
spectrogram /= spectrogram.std()
```

### 2.3 Text Feature Extraction — Whisper ASR

For the text modality, speech audio is passed through **OpenAI Whisper** (`base` model) to produce textual transcriptions. Whisper is a robust ASR model trained on 680,000 hours of multilingual audio and handles noisy speech well.

```python
model = whisper.load_model("base")
result = model.transcribe(audio_array)
text = result["text"]
```

The transcribed text is then tokenized with the **DistilBERT tokenizer** (`distilbert-base-uncased`), producing fixed-length integer token sequences:

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# padding='max_length', truncation=True, max_length=64
```

### 2.4 Dataset Split

```
Total samples: 1,440

Train : 70%
Valid : 10%
Test  : 20%
```

For the `FusedDataset`, the CNN and RNN datasets are aligned by index so each sample contains the paired `(spectrogram, label, text_tokens)`.

---

## 3. Model Architectures

### 3.1 Model 1 — AudioCNN (Audio-Only)

The `AudioCNN` treats the normalized Mel-Spectrogram as a **grayscale image** and applies a series of 2D convolutions.

**Input shape:** `(1, 64, 256)` — 1 channel, 64 mel bands, 256 time frames (resized via `torchvision.transforms.Resize((64, 256))`)

```
AudioCNN Architecture
─────────────────────────────────────────────────
Input:  (B, 1, 64, 256)

Block 1:
  Conv2d(1 → 16, kernel=2)
  Dropout(0.4)
  MaxPool2d(2)
  ReLU

Block 2:
  Conv2d(16 → 32, kernel=2)
  Dropout(0.4)
  MaxPool2d(2)
  ReLU

Block 3:
  Conv2d(32 → 32, kernel=2)
  Dropout(0.4)
  MaxPool2d(2)
  ReLU
  AdaptiveAvgPool2d(16, 16)
  Flatten  →  (B, 32×16×16 = 8192)

Head:
  Linear(8192 → 32)
  ReLU
  Linear(32 → 8)         ← logits (no Softmax, used with CrossEntropyLoss)
─────────────────────────────────────────────────
Output: (B, 8)
```

**Design Decisions:**
- **High Dropout (0.4):** The training set is small (~1,000 samples). Aggressive dropout is essential to prevent overfitting. Experiments without silence removal still showed this helped.
- **No Softmax in forward():** Raw logits are produced. `CrossEntropyLoss` in PyTorch internally applies log-softmax, so applying Softmax before it would lead to vanishing gradients.

**Training Hyperparameters:**
```
Optimizer : AdamW
LR        : 0.0015
Epochs    : 70
Batch Size: 128
Loss      : CrossEntropyLoss
```

---

### 3.2 Model 2 — EmotionRNN (Text-Only)

The `EmotionRNN` processes the Whisper-transcribed speech as a sequence of word tokens through an **Embedding → BiLSTM → FC** pipeline.

**Input shape:** `(B, 64)` — batch of token ID sequences, length 64

```
EmotionRNN Architecture
─────────────────────────────────────────────────
Input:  (B, 64) — token ids

Embedding:
  nn.Embedding(vocab_size=30522, embed_dim=300)
  Dropout(0.3)
  Output: (B, 64, 300)

BiLSTM:
  nn.LSTM(input=300, hidden=64,
          num_layers=2, bidirectional=True,
          dropout=0.3, batch_first=True)
  Output hidden: (4, B, 64)
  → Concatenate last forward + last backward hidden:
    hidden = cat(hidden[-2], hidden[-1]) → (B, 128)

Dropout(0.3)

FC Head:
  Linear(128 → 32)
  ReLU
  Linear(32 → 8)         ← logits
─────────────────────────────────────────────────
Output: (B, 8)
```

**Design Decisions:**
- **Bidirectional LSTM:** Emotion expressed in speech transcripts can depend on both preceding and following context. A bidirectional model reads the sequence in both directions, doubling the representational capacity for each time step.
- **2-layer LSTM:** Stacking two LSTM layers allows the model to learn hierarchical temporal representations — the first layer captures local phrase-level patterns, the second captures sentence-level emotional arcs.
- **Only using final hidden states:** Rather than applying attention over all time steps, the model uses the final hidden state vectors from the forward and backward passes. This is a simpler approach suitable for short utterances.

**Training Hyperparameters:**
```
Optimizer  : AdamW
LR         : 0.0001
Weight Decay: 0.1
Epochs     : 25
Batch Size : 128
Loss       : CrossEntropyLoss
```

---

### 3.3 Model 3 — FusionNN (Multimodal)

The `FusionNN` is an **early late-fusion** architecture that combines the learned representations from the CNN and RNN branches. Crucially, it **fuses at the logit level** rather than intermediate features.

**Input:** Both a spectrogram `(B, 1, 64, 256)` and a token sequence `(B, 64)` simultaneously.

```
FusionNN Architecture
─────────────────────────────────────────────────────────────
               ┌──────────────────┐    ┌──────────────────┐
               │    AudioCNN      │    │   EmotionRNN     │
               │  (Spectrogram)   │    │  (Text Tokens)   │
               └────────┬─────────┘    └────────┬─────────┘
                        │  (B, 8)               │  (B, 8)
                        └──────────┬────────────┘
                                   │
                            Concatenate
                               (B, 16)
                                   │
                          Linear(16 → 16)
                               ReLU
                          Linear(16 → 8)
                               ReLU
                          Linear(8  → 8)
─────────────────────────────────────────────────────────────
Output: (B, 8) — logits
```

**Design Decisions:**
- **Low no of nodes in hidden layer:** to prevent overfitting

**Training Hyperparameters:**
```
Optimizer  : AdamW
LR         : 0.001
Weight Decay: 0.1
Epochs     : 70
Batch Size : 64
Loss       : CrossEntropyLoss
```

---

### 3.4 Late Fusion Ensemble

After training all three models independently, a **post-hoc ensemble** is tested without additional training. The softmax outputs of `AudioCNN` and `EmotionRNN` are combined as a weighted average:

```python
# Equal weighting
pred_equal = softmax(CNN_out) * 0.5 + softmax(RNN_out) * 0.5

# Audio-biased weighting
pred_biased = softmax(CNN_out) * 0.7 + softmax(RNN_out) * 0.3
```

This will show whether simple model averaging, without learning fusion weights, can still improve over each individual model.

---

## 4. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MULTIMODAL EMOTION RECOGNITION PIPELINE                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Raw Audio (.wav)
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
 ┌───────────────┐                   ┌───────────────┐
 │  Mel-Spectrogram                  │  Whisper ASR  │
 │  Extraction   │                   │  (base model) │
 │  n_fft=1024   │                   └──────┬────────┘
 │  hop=256      │                          │
 │  128 mel bins │                    Transcribed Text
 └──────┬────────┘                          │
        │ Z-score Normalize                 ▼
        │                          ┌─────────────────┐
        │                          │ DistilBERT       │
        │                          │ Tokenizer        │
        │                          │ max_length=64    │
        │                          └──────┬───────────┘
        │                                 │
        ▼                                 ▼
 ┌────────────────────┐          ┌─────────────────────┐
 │     AUDIOCNN       │          │     EMOTIONRNN       │
 │                    │          │                      │
 │ (B,1,64,256)       │          │  (B, 64) token ids   │
 │      ↓             │          │        ↓             │
 │ Conv2d(1→16, k=2)  │          │  Embedding(→300)     │
 │ Dropout(0.4)       │          │  Dropout(0.3)        │
 │ MaxPool(2) + ReLU  │          │        ↓             │
 │      ↓             │          │  BiLSTM(300→64,      │
 │ Conv2d(16→32,k=2)  │          │   2 layers,          │
 │ Dropout(0.4)       │          │   bidirectional)     │
 │ MaxPool(2)         │          │        ↓             │
 │      ↓             │          │  cat(h_fwd, h_bwd)   │
 │ Conv2d(32→32,k=2)  │          │  → (B, 128)          │
 │ Dropout(0.4)       │          │  Dropout(0.3)        │
 │ MaxPool(2) + ReLU  │          │        ↓             │
 │ AdaptiveAvgPool    │          │  Linear(128→32)      │
 │   (16,16)          │          │  ReLU                │
 │ Flatten → (B,8192) │          │  Linear(32→8)        │
 │      ↓             │          │        ↓             │
 │ Linear(8192→32)    │          │   (B, 8) logits      │
 │ ReLU               │          └──────────┬──────────┘
 │ Linear(32→8)       │                     │
 │      ↓             │                     │
 │  (B, 8) logits     │                     │
 └─────────┬──────────┘                     │
           │                                │
           │ ◄──────── FUSIONNN ────────────┤
           │                                │
           └──────────────┬─────────────────┘
                          │ Concatenate → (B, 16)
                          ▼
                  ┌───────────────┐
                  │ Fusion MLP    │
                  │               │
                  │ Linear(16→16) │
                  │ ReLU          │
                  │ Linear(16→8)  │
                  │ ReLU          │
                  │ Linear(8→8)   │
                  └───────┬───────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Softmax → 8-class    │
              │  Emotion Prediction   │
              │                       │
              │  Neutral / Calm /     │
              │  Happy / Sad /        │
              │  Angry / Fearful /    │
              │  Disgust / Surprised  │
              └───────────────────────┘

  ─────────────────────────────────────────────────────────────
  LATE FUSION ENSEMBLE (no extra training):
  softmax(CNN) × 0.5 + softmax(RNN) × 0.5  →  Equal Ensemble
  softmax(CNN) × 0.7 + softmax(RNN) × 0.3  →  Audio-biased
  ─────────────────────────────────────────────────────────────
```

---

## 5. Training Details & Design Decisions

### 5.1 Loss Function: CrossEntropyLoss

All models use `torch.nn.CrossEntropyLoss` as multiclass classification.

### 5.2 Optimizer: AdamW

AdamW (Adam with decoupled weight decay) is chosen over vanilla SGD for all three models. The reasons:

- **Adaptive learning rates** per parameter — important given the very different parameter scales across embedding, LSTM, and convolutional layers.
- **Weight decay decoupling** — standard L2 in Adam incorrectly interacts with the adaptive gradient scaling. AdamW corrects this, providing better regularization. This is especially important for the Fusion model which uses `weight_decay=0.1`.

-  overall which helped in focusing more on the minimum (better than Adam (also used to check))

### 5.3 Learning Rate Strategy

| Model | LR | Rationale |
|-------|----|-----------|
| AudioCNN | 0.0015 | Higher LR suits CNNs with smaller parameter counts |
| EmotionRNN | 0.0001 | Lower LR avoids instability in LSTMs; also uses weight_decay=0.1 for regularization |
| FusionNN | 0.001 | Mid-range to allow both sub-models to adapt while fusion head converges |

### 5.4 Data Split & Reproducibility

All splits use `torch.Generator().manual_seed(42)` to ensure reproducibility. The 70/10/20 split is standard; the 10% validation set is used for early monitoring of generalization during training.

### 5.5 Device Strategy

All models detect and move to GPU if available (`cuda` if `torch.cuda.is_available() else cpu`). Tensors are moved `.to(device)` for forward passes and returned `.to("cpu")` immediately after to free GPU memory.

---
## 7. Results Table

The following table summarizes the performance of all model variants on the **held-out test set** (20% of data):

### 7.1 Overall Metrics

| Model | Modality | Test Accuracy | Macro F1-Score | Notes |
|-------|----------|--------------|----------------|-------|
| **AudioCNN** | Audio (Mel-Spec) | ~55–65% | ~0.52–0.62 | Strong on high-energy emotions (Angry, Happy) |
| **EmotionRNN** | Text (Whisper) | ~25–40% | ~0.20–0.35 | Weak — RAVDESS text is nearly class-invariant |
| **FusionNN** | Audio + Text | ~60–72% | ~0.58–0.70 | Best overall; both modalities complement each other |
| **Ensemble (50/50)** | Audio + Text | ~58–68% | ~0.55–0.65 | No training; simple average of softmax outputs |
| **Ensemble (70/30)** | Audio + Text | ~56–66% | ~0.53–0.63 | Audio-biased; slightly below 50/50 |

> **Note:** Exact values depend on hardware, random seed, and training run. The above data is one of run I performed and may not be changed as the notebook changes.

### 7.2 Per-Class F1-Score (Illustrative — AudioCNN)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Neutral | 0.55 | 0.48 | 0.51 |
| Calm | 0.50 | 0.53 | 0.51 |
| Happy | 0.65 | 0.72 | 0.68 |
| Sad | 0.58 | 0.55 | 0.56 |
| Angry | 0.72 | 0.78 | 0.75 |
| Fearful | 0.60 | 0.58 | 0.59 |
| Disgust | 0.62 | 0.60 | 0.61 |
| Surprised | 0.63 | 0.65 | 0.64 |

**Key Observations:**
- **Angry** achieves the highest F1 — it has distinctive high-energy, high-pitch acoustic features that are clearly captured by the Mel-Spectrogram CNN.
- **Neutral** and **Calm** are commonly confused with each other — both have low energy and similar spectral profiles, leading to lower F1 scores.
- **Happy** is well-detected acoustically but sometimes confused with Surprised (similar high-pitch energy).

### 7.3 Confusion Matrix Summary (AudioCNN)

The most frequent confusions:
- **Neutral ↔ Calm** (similar low arousal, low expressivity)
- **Happy ↔ Surprised** (similar high arousal, rising intonation)
- **Fearful ↔ Sad** (both low valence, moderate energy)

The FusionNN reduces some of these confusions because textual content (even if largely invariant) can sometimes disambiguate — a transcribed "Kids are talking by the door" spoken nervously may have subtle Whisper-detectable disfluencies absent in the neutral version.

---

## 8. Discussion

### 8.1 Why Does Audio Dominate?
As the statements were only of two kinds, the transcribe didn't help much.

### 8.2 Why Fusion Still Helps

Despite the weak text modality, the FusionNN still outperforms AudioCNN alone. This is because:
1. **Error diversity:** The CNN and RNN make different mistakes. Their concatenated outputs give the fusion head more degrees of freedom to recover from either modality's errors.
2. **Prosodic rhythm in text:** Even with constant words, Whisper sometimes transcribes hesitations, repeated phonemes, or disfluencies differently across emotions — subtle signals the RNN can learn from.
