# MusicConRec Training Checklist

Complete this checklist before starting training to ensure everything is properly configured.

## Dataset & Data Loading

- [ ] **Verify Data Paths**
  - [ ] `training_hyperparam.json` paths point to correct data directories
  - [ ] Training data file exists: `./data/MusicBench_train.json`
  - [ ] Validation data file exists: `./data/MusicBench_val.json`
  - [ ] Audio files are accessible at paths specified in JSON

- [ ] **Verify Data Format**
  - [ ] Audio files: 24 kHz sample rate (as per EncodecModel requirement)
  - [ ] Chord-beat format: 13-dimensional features per time step
  - [ ] Both aligned temporally (same duration)
  - [ ] Batch loading works without errors: `python -c "from dataset import MusicBenchDataset; ds = MusicBenchDataset('./data/MusicBench_train.json', 'C:/temp'); print(len(ds))"`

- [ ] **Data Validation**
  ```bash
  python -c "
  from dataset import MusicBenchDataset
  ds = MusicBenchDataset('./data/MusicBench_train.json', './data')
  audio, chord = ds[0]
  print(f'Audio shape: {audio.shape}')
  print(f'Chord shape: {chord.shape}')
  print(f'Audio range: [{audio.min():.3f}, {audio.max():.3f}]')
  "
  ```

## Model Architecture

- [ ] **Audio Branch**
  - [ ] EncodecModel downloads successfully (will cache locally)
  - [ ] Verify EncodecModel parameters are frozen
  - [ ] Code embedding dimension matches codebook size (1024)
  - [ ] Attention pooling output shape: (B, 128)

- [ ] **Chord-Beat Branch**
  - [ ] Chord encoder input dimension: 13
  - [ ] Transformer encoder configured properly
  - [ ] Positional encoding works
  - [ ] Output dimension: (B, 128)

- [ ] **Model Instantiation**
  ```bash
  python -c "
  import torch
  from model.model import MusicConRec
  model = MusicConRec()
  audio = torch.randn(2, 1, 8000)
  chord = torch.randn(2, 500, 13)
  output = model(audio, chord)
  print(f'✓ Model forward pass successful')
  print(f'  - z_audio shape: {output[\"z_audio\"].shape}')
  print(f'  - z_chord shape: {output[\"z_chord\"].shape}')
  print(f'  - x_recon shape: {output[\"x_recon\"].shape}')
  "
  ```

## Loss Functions

- [ ] **Contrastive Loss (NT-Xent)**
  - [ ] Embedding normalization working
  - [ ] Similarity matrix computation correct
  - [ ] Bidirectional loss computation
  - [ ] Temperature parameter reasonable (0.07)
  
  ```bash
  python -c "
  import torch
  from loss.ntxent import nt_xent
  z_audio = torch.randn(16, 128)
  z_chord = torch.randn(16, 128)
  loss = nt_xent(z_audio, z_chord)
  print(f'✓ NT-Xent loss: {loss.item():.4f}')
  assert not torch.isnan(loss), 'Loss is NaN!'
  "
  ```

- [ ] **Reconstruction Loss (Multi-Scale STFT)**
  - [ ] STFT computation for 3 scales
  - [ ] Spectral convergence calculation
  - [ ] Log magnitude loss calculation
  
  ```bash
  python -c "
  import torch
  from loss.recon import multi_scale_stft_loss
  audio = torch.randn(4, 8000)
  recon = audio + 0.01 * torch.randn_like(audio)
  loss = multi_scale_stft_loss(audio, recon)
  print(f'✓ STFT loss: {loss.item():.4f}')
  assert not torch.isnan(loss), 'Loss is NaN!'
  "
  ```

## Hyperparameters

- [ ] **Review `training_hyperparam.json`**
  ```json
  {
    "epochs": 30,                    ← Reasonable for limited resources
    "batch_size": 16,                ← Fits in GPU memory?
    "learning_rate": 0.0001,         ← Standard for Adam
    "ntxent_temperature": 0.07,      ← Standard for contrastive
    "lambda_contrastive": 1.0,       ← Check balance
    "lambda_recon": 0.05,            ← Audio reconstruction weight
    "traindata_dir": "...",
    "valdata_dir": "..."
  }
  ```

- [ ] **GPU Memory Check**
  - [ ] Batch size doesn't exceed available VRAM (typical: 16 fits in 8GB)
  - [ ] Test with `torch.cuda.memory_allocated()` before training

- [ ] **Lambda Weights Balance**
  - [ ] Ratio `lambda_contrastive : lambda_recon` = 1.0 : 0.05 = 20:1
  - [ ] This prioritizes contrastive learning over reconstruction
  - [ ] Adjust if needed: If reconstruction matters more, increase `lambda_recon`

## Output Directories

- [ ] **Create Output Directories**
  ```bash
  mkdir -p outputs
  mkdir -p runs
  ```

- [ ] **Verify Write Permissions**
  - [ ] Can write to `./outputs/` (model checkpoints)
  - [ ] Can write to `./runs/` (TensorBoard logs)

- [ ] **TensorBoard Setup**
  ```bash
  tensorboard --logdir ./runs
  ```
  Then navigate to http://localhost:6006

## Pre-Training Tests

- [ ] **Single Batch Training**
  ```bash
  python -c "
  import torch
  from train import train
  from dataset import MusicBenchDataset, collate_fn
  from torch.utils.data import DataLoader
  
  # Quick test with 1 batch
  ds = MusicBenchDataset('./data/MusicBench_train.json', './data')
  loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
  audio, chord = next(iter(loader))
  
  from model.model import MusicConRec
  from loss.ntxent import nt_xent
  from loss.recon import multi_scale_stft_loss
  
  model = MusicConRec()
  output = model(audio, chord)
  
  loss_c = nt_xent(output['z_audio'], output['z_chord'])
  loss_r = multi_scale_stft_loss(audio.squeeze(1), output['x_recon'].squeeze(1))
  loss_total = loss_c + 0.05 * loss_r
  
  print(f'✓ Single batch test successful')
  print(f'  Contrastive loss: {loss_c.item():.4f}')
  print(f'  Reconstruction loss: {loss_r.item():.4f}')
  print(f'  Total loss: {loss_total.item():.4f}')
  "
  ```

- [ ] **Backward Pass Test**
  ```bash
  python -c "
  import torch
  from model.model import MusicConRec
  
  model = MusicConRec()
  audio = torch.randn(2, 1, 8000, requires_grad=False)
  chord = torch.randn(2, 500, 13, requires_grad=False)
  
  output = model(audio, chord)
  loss = output['z_audio'].sum() + output['z_chord'].sum()
  loss.backward()
  
  # Check gradients
  grad_count = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
  total_params = sum(1 for p in model.parameters() if p.requires_grad)
  
  print(f'✓ Backward pass successful')
  print(f'  {grad_count}/{total_params} parameters have gradients')
  "
  ```

## GPU & Device Setup

- [ ] **Check GPU Availability**
  ```bash
  python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
  ```

- [ ] **Check CUDA Version**
  ```bash
  python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"
  ```

- [ ] **Memory Benchmark**
  ```bash
  python -c "
  import torch
  torch.cuda.reset_peak_memory_stats()
  
  from model.model import MusicConRec
  model = MusicConRec().cuda()
  
  audio = torch.randn(16, 1, 24000).cuda()  # 1 second at 24kHz
  chord = torch.randn(16, 1000, 13).cuda()
  
  output = model(audio, chord)
  
  peak_mem = torch.cuda.max_memory_allocated() / 1e9
  print(f'Peak GPU memory: {peak_mem:.2f} GB')
  print(f'✓ Batch size 16 with 1-second audio is feasible' if peak_mem < 8 else '✗ May cause OOM')
  "
  ```

## Dependencies

- [ ] **Check Required Packages**
  ```bash
  pip list | grep -E "torch|torchaudio|transformers|numpy"
  ```

- [ ] **Verify Versions** (minimum)
  - [ ] torch >= 2.0.0
  - [ ] torchaudio >= 2.0.0
  - [ ] transformers >= 4.30.0

## Before Starting Training

1. [ ] All checkpoints pass above
2. [ ] TensorBoard is monitoring directory: `tensorboard --logdir ./runs &`
3. [ ] Best model checkpoint path is clear: `./outputs/best_model.pth`
4. [ ] Training log will be saved with timestamp
5. [ ] Early stopping configured with patience=5
6. [ ] Gradient clipping enabled (max_norm=1.0)

## During Training

- [ ] **Monitor TensorBoard**
  - [ ] Open http://localhost:6006
  - [ ] Watch both loss curves (should decrease)
  - [ ] Check contrastive vs. reconstruction balance

- [ ] **Check Console Output**
  - [ ] Loss values printed each epoch
  - [ ] "✓ Best model saved" message when validation improves
  - [ ] No NaN or Inf values appearing

- [ ] **GPU Monitoring** (optional but recommended)
  ```bash
  nvidia-smi -l 1  # Update every 1 second
  ```

## After Training

- [ ] **Verify Checkpoints Created**
  - [ ] `./outputs/best_model.pth` exists
  - [ ] `./outputs/final_model.pth` exists
  - [ ] File sizes > 50MB (model has ~5M parameters)

- [ ] **Load Best Model**
  ```bash
  python -c "from inference import load_best_model; model = load_best_model(); print('✓ Best model loaded successfully')"
  ```

- [ ] **Test Inference**
  ```bash
  python -c "
  from inference import load_best_model
  import torch
  
  model = load_best_model()
  audio = torch.randn(1, 1, 8000)
  embedding = model.get_audio_embedding(audio)
  print(f'✓ Inference works. Embedding shape: {embedding.shape}')
  "
  ```

---

## Troubleshooting

If any step fails, check:

1. **NaN Loss:** Audio normalization, gradient clipping, learning rate
2. **Memory Error:** Reduce batch size or audio length
3. **Data Loading:** Verify JSON format, file paths, audio quality
4. **Model Error:** Check EncodecModel download, dimension mismatches
5. **Gradient Issues:** Verify loss computation, requires_grad settings

