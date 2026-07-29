import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, EncodecConfig
from model.attention_weighted_pooling import AttentionWeightedPooling
from model.projection import ProjectionHead
from model.chordbeat_encoder import ChordBeatEncoder


class MusicConRec(nn.Module):
    def __init__(self, codebook_size=1024, feature_dim=128, proj_dim=128,
                 queue_size=4096, momentum=0.99):
        super().__init__()

        # === AUDIO SIDE (query / online encoders) ===
        self.encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")

        # Do not freeze EncodecModel parameters — allow fine-tuning
        for param in self.encodec.parameters():
            param.requires_grad = True

        self.code_embedding = nn.Embedding(codebook_size, feature_dim)
        self.audio_pool = AttentionWeightedPooling(feature_dim)
        self.audio_proj = ProjectionHead(feature_dim, out_dim=proj_dim)

        # === CHORD SIDE (query / online encoder) ===
        self.chord_encoder = ChordBeatEncoder(
            input_dim=13,
            d_model=feature_dim
        )

        # === MOMENTUM (key) ENCODERS ===
        # Exact architectural copies of every module in the audio and chord
        # query paths, initialized with the same weights. These are never
        # touched by backprop — only by the EMA update in
        # update_momentum_encoders(). This is what makes it MoCo rather than
        # a plain memory-bank/SimCLR setup: the key embeddings come from a
        # slowly-drifting encoder, so the queue's negatives stay consistent
        # with each other across training steps instead of the queue holding
        # embeddings that were produced by many different fast-moving encoder
        # states.
        self.encodec_k = copy.deepcopy(self.encodec)
        self.code_embedding_k = copy.deepcopy(self.code_embedding)
        self.audio_pool_k = copy.deepcopy(self.audio_pool)
        self.audio_proj_k = copy.deepcopy(self.audio_proj)
        self.chord_encoder_k = copy.deepcopy(self.chord_encoder)

        for module in (self.encodec_k, self.code_embedding_k, self.audio_pool_k,
                       self.audio_proj_k, self.chord_encoder_k):
            for param in module.parameters():
                param.requires_grad = False

        # === QUEUES ===
        # Random unit-norm init (rather than zeros) so early-training negatives
        # aren't degenerate zero vectors, which contribute ~0 to every logit
        # until the queue has cycled through once.
        self.register_buffer("queue_audio", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_chord", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = momentum

    # =========================
    # SHARED AUDIO-BRANCH ENCODING LOGIC
    # =========================
    # Factored out so the query path (forward) and key path (forward_momentum)
    # run identical logic through their respective (online vs. momentum)
    # module instances instead of duplicating/drifting copies of this code.
    def _encode_audio(self, audio, encodec, code_embedding, audio_pool, audio_proj):

        for module in encodec.modules():
            if isinstance(module, nn.LSTM):
                module.flatten_parameters()

        encoder_outputs = encodec.encode(audio)
        audio_codes = encoder_outputs['audio_codes'].long()
        audio_scales = encoder_outputs['audio_scales']

        codes = audio_codes.squeeze(0).permute(0, 2, 1)
        codes = code_embedding(codes)   # (B, T, Q, D)
        codes = codes.mean(dim=2)       # (B, T, D)

        h = audio_pool(codes)           # (B, D)
        z = audio_proj(h)               # (B, proj_dim)
        return z, h, audio_codes, audio_scales

    def forward(self, audio, chord):
        """
        audio: (B, 1, T)
        chord: (B, T_chord, 13)
        """
        # =========================
        # QUERY ENCODERS (grad-tracked)
        # =========================
        z_audio, h_audio, audio_codes, audio_scales = self._encode_audio(
            audio, self.encodec, self.code_embedding, self.audio_pool, self.audio_proj
        )
        z_audio = F.normalize(z_audio, dim=-1)

        # =========================
        # RECONSTRUCTION (always via the online encoder — unrelated to MoCo)
        # =========================
        x_recon = self.encodec.decode(audio_codes, audio_scales)['audio_values']
        x_recon = torch.tanh(x_recon).clamp(-1.0, 1.0)

        # =========================
        # CHORD BRANCH (query)
        # =========================
        z_chord, h_chord = self.chord_encoder(chord)
        z_chord = F.normalize(z_chord, dim=-1)

        return {
            "x_recon": x_recon,
            "z_audio": z_audio,
            "z_chord": z_chord,
            "h_audio": h_audio,
            "h_chord": h_chord
        }

    @torch.no_grad()
    def forward_momentum(self, audio, chord):
        """
        Key-side forward pass through the momentum encoders. Call
        update_momentum_encoders() immediately before this, every training
        step, so the key encoders reflect the latest EMA of the query
        encoders. No gradients ever flow through this path.
        """
        k_audio, _, _, _ = self._encode_audio(
            audio, self.encodec_k, self.code_embedding_k, self.audio_pool_k, self.audio_proj_k
        )
        k_audio = F.normalize(k_audio, dim=-1)

        k_chord, _ = self.chord_encoder_k(chord)
        k_chord = F.normalize(k_chord, dim=-1)

        return k_audio, k_chord

    @torch.no_grad()
    def update_momentum_encoders(self):
        """EMA-update every key module from its corresponding query module.
        Call once per training step, before forward_momentum()."""
        self.momentum_update(self.encodec, self.encodec_k)
        self.momentum_update(self.code_embedding, self.code_embedding_k)
        self.momentum_update(self.audio_pool, self.audio_pool_k)
        self.momentum_update(self.audio_proj, self.audio_proj_k)
        self.momentum_update(self.chord_encoder, self.chord_encoder_k)

    def momentum_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data = self.momentum * target_param.data + (1.0 - self.momentum) * online_param.data

    def update_moco_queue(self, k_audio, k_chord):
        """Enqueue a batch of KEY embeddings (from forward_momentum, already
        detached/normalized) and dequeue the oldest entries."""
        batch_size = k_audio.size(0)
        queue_size = self.queue_audio.size(0)
        ptr = int(self.queue_ptr.item())

        if batch_size > queue_size:
            raise ValueError("Batch size exceeds MoCo queue size")

        if ptr + batch_size <= queue_size:
            self.queue_audio[ptr:ptr + batch_size] = k_audio.detach()
            self.queue_chord[ptr:ptr + batch_size] = k_chord.detach()
            self.queue_ptr[0] = (ptr + batch_size) % queue_size
        else:
            remaining = queue_size - ptr
            self.queue_audio[ptr:queue_size] = k_audio[:remaining].detach()
            self.queue_chord[ptr:queue_size] = k_chord[:remaining].detach()
            self.queue_audio[:batch_size - remaining] = k_audio[remaining:].detach()
            self.queue_chord[:batch_size - remaining] = k_chord[remaining:].detach()
            self.queue_ptr[0] = (ptr + batch_size) % queue_size