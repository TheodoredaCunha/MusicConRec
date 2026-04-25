# MusicConRec Implementation Guide

## Overview

MusicConRec is an adaptation of the **ConRec** paper (Dippel et al., 2021) for music. The key innovation is combining **audio reconstruction** with **contrastive learning** between audio and chord-beat representations.

### Paper: "Towards Fine-grained Visual Representations by Combining Contrastive Learning with Image Reconstruction and Attention-weighted Pooling"

**Original ConRec (Vision):**
```
Contrastive Loss: between augmented versions of SAME image
Reconstruction Loss: MSE pixel reconstruction
```

**MusicConRec (Audio):**
```
Contrastive Loss: between DIFFERENT modalities (audio vs. chord-beat)
Reconstruction Loss: Multi-scale STFT reconstruction
```

---

## Architecture Overview

### 1. Audio Branch (Representation Learning)
```
Audio Input (B, 1, T)
    ↓
EncodecModel.encode()  [frozen]
    ↓
Audio Codes (B, Q, T, 1024)
    ↓
Code Embedding Layer  
    ↓
Codes (B, T, feature_dim)
    ↓
AttentionWeightedPooling → h_audio (B, feature_dim)
    ↓
ProjectionHead → z_audio (B, proj_dim=128)
```

### 2. Chord-Beat Branch (Representation Learning)
```
Chord-Beat Input (B, T_chord, 13)
    ↓
Linear Projection → (B, T, feature_dim)
    ↓
PositionalEncoding
    ↓
TransformerEncoder (4 layers)
    ↓
AttentionWeightedPooling → h_chord (B, feature_dim)
    ↓
ProjectionHead → z_chord (B, proj_dim=128)
```

### 3. Reconstruction Branch (Audio Quality)
```
Audio Codes (B, Q, T, 1024) from encoder
    ↓
EncodecModel.decode()  [frozen]
    ↓
Reconstructed Audio (B, 1, T)
    ↓
[Compared to original audio using Multi-Scale STFT Loss]
```

---

## Loss Function

### Combined Loss:
```
L_total = λ_contrastive * L_ntxent + λ_recon * L_stft
```

### NT-Xent (Contrastive Loss)
- **Formula:** Normalized temperature-scaled cross entropy
- **Input:** z_audio (B, 128), z_chord (B, 128)
- **Operation:** Bidirectional (audio→chord + chord→audio)
- **Temperature:** 0.07 (from hyperparams)
- **Output:** Single scalar loss

### Multi-Scale STFT Loss (Reconstruction)
- **Scales:** 3 frequencies [1024, 512, 256]
- **Components:** 
  - Spectral Convergence Loss
  - Log Magnitude Loss
- **Input:** Original audio (B, T), Reconstructed audio (B, T)
- **Output:** Single scalar loss

### Hyperparameters
```json
{
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "ntxent_temperature": 0.07,
    "lambda_contrastive": 1.0,      ← Can tune for balance
    "lambda_recon": 0.05            ← Can tune for balance
}
```

---

## Key Design Decisions

### 1. **Using Pre-trained EncodecModel**
- **Why:** Provides compressed audio representation without training from scratch
- **Frozen:** Parameters are frozen to use as fixed encoder/decoder
- **Output:** 1024-dimensional codebook indices

### 2. **Attention-Weighted Pooling**
- **Why:** Captures fine-grained temporal patterns in music
- **Implementation:** Learned attention weights over time dimension
- **Output:** Fixed-size representation from variable-length sequences

### 3. **Chord-Beat Representation**
- **Dimension:** 13 features (likely: 12 pitch classes + 1 beat indicator)
- **Encoder:** Transformer (better for sequential structure)
- **Positional Encoding:** Captures temporal positions

### 4. **Multi-Scale STFT Loss**
- **Why:** Frequency-domain reconstruction captures perceptual quality
- **Better than:** MSE pixel loss (used in original ConRec)
- **Advantage:** Accounts for human auditory perception

---

## Training Procedure

### Data Loading
```python
# Audio + Chord-Beat pairs loaded from MusicBench dataset
# Audio: 24kHz, normalized to [-1, 1]
# Chord-Beat: Temporal alignment with audio
```

### Optimization
- **Optimizer:** Adam (lr=0.0001)
- **Gradient Clipping:** max_norm=1.0 (prevents exploding gradients)
- **Scheduler:** None (constant learning rate)

### Checkpointing Strategy
1. **Best Model:** Saved when validation loss improves
2. **Final Model:** Saved after all epochs complete
3. **Early Stopping:** Patience=5 epochs without improvement

### Logging
- TensorBoard logs to `./runs/{timestamp}/`
- Metrics tracked:
  - `Loss/Train_Total`
  - `Loss/Train_Contrastive`
  - `Loss/Train_Reconstruction`
  - `Loss/Val_Total`
  - `Loss/Val_Contrastive`
  - `Loss/Val_Reconstruction`

---

## Expected Behavior

### Convergence
- **Contrastive Loss:** Should decrease as embeddings become more aligned
- **Reconstruction Loss:** Should decrease as audio quality improves
- **Total Loss:** Balance depends on lambda weights

### When λ_contrastive is too high:
- Audio reconstruction may suffer
- Embeddings become very similar

### When λ_recon is too high:
- Reconstruction quality improves but embeddings lose discriminability
- Contrastive learning may plateau

---

## Recommended Tuning Steps

1. **Verify baseline** with current hyperparameters
2. **If reconstruction loss is too high:** Increase `lambda_recon` (e.g., 0.05 → 0.1)
3. **If contrastive plateau:** Decrease `lambda_recon` or increase `lambda_contrastive`
4. **Learning rate tuning:** Try [5e-5, 2e-4] if convergence issues
5. **Batch size:** Larger batches (32, 64) can improve contrastive learning

---

## Evaluation Protocol

### Downstream Tasks
1. **Music Classification:** Freeze encoder, train linear classifier
2. **Retrieval:** Embedding-based audio/chord similarity
3. **Reconstruction Quality:** PESQ, STOI metrics

### Representation Quality
- **Uniformity:** Embeddings should be spread out
- **Alignment:** Similar audio-chord pairs should have similar embeddings
- **Separation:** Different audio-chord pairs should be far apart

---

## Differences from Original ConRec

| Aspect | Original ConRec | MusicConRec |
|--------|-----------------|------------|
| Domain | Vision (Images) | Audio |
| Encoder | U-Net (trained) | EncodecModel (frozen) |
| Contrastive Pairs | Augmentations of same image | Different modalities |
| Reconstruction | MSE pixel loss | Multi-scale STFT |
| Pooling | Spatial attention | Temporal attention |
| Reconstruction Target | Raw pixels | Encoded audio codes |

---

## Files Overview

```
model/
  ├── model.py                    # Main MusicConRec model
  ├── attention_weighted_pooling.py
  ├── projection.py
  └── chordbeat_encoder.py
loss/
  ├── ntxent.py                   # Contrastive loss
  └── recon.py                    # Reconstruction loss
dataset.py                         # Data loading
train.py                           # Training loop
eval.py                            # Evaluation
```

---

## Troubleshooting

### Issue: NaN Loss
- **Cause:** Likely gradient explosion or audio with extreme values
- **Fix:** Verify audio normalization, check gradient clipping

### Issue: Reconstruction Loss Very High
- **Cause:** Audio codes poorly reconstructed or compression artifacts
- **Fix:** Check EncodecModel input format, verify audio sampling rate

### Issue: Contrastive Loss Not Decreasing
- **Cause:** Embeddings not well-aligned or learning rate too low
- **Fix:** Increase learning rate, check embedding dimensions

### Issue: Overfitting (Val loss >> Train loss)
- **Cause:** Model too large or lambda_recon too high
- **Fix:** Reduce model size, increase regularization, or rebalance lambdas

---

## Future Improvements

1. **Learnable Encodec Parameters:** Train full end-to-end instead of frozen
2. **Data Augmentation:** Apply to audio/chord pairs for robustness
3. **Scheduler:** Learning rate scheduling for better convergence
4. **Momentum Encoder:** Use momentum contrast for better batch efficiency
5. **Multi-GPU:** DataParallel for faster training
6. **Model Inference:** Add inference-only mode with best checkpoint

