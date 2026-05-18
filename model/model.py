import torch
import torch.nn as nn
from transformers import EncodecModel, EncodecConfig
from model.attention_weighted_pooling import AttentionWeightedPooling
from model.projection import ProjectionHead
from model.chordbeat_encoder import ChordBeatEncoder

class MusicConRec(nn.Module):
    def __init__(self, codebook_size=1024, feature_dim=128, proj_dim=128):
        super().__init__()

        # === AUDIO SIDE ===
        self.encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")
        
        # Freeze EncodecModel parameters (we use it as fixed encoder/decoder)
        for param in self.encodec.parameters():
            param.requires_grad = False

        self.code_embedding = nn.Embedding(codebook_size, feature_dim)

        self.audio_pool = AttentionWeightedPooling(feature_dim)
        self.audio_proj = ProjectionHead(feature_dim, out_dim=proj_dim)

        # === CHORD SIDE ===
        self.chord_encoder = ChordBeatEncoder(
            input_dim=13,
            d_model=feature_dim
        )

    def forward(self, audio, chord):
        """
        audio: (B, 1, T)
        chord: (B, T_chord, 13)
        """


        # =========================
        # ENCODE
        # =========================

        encoder_outputs = self.encodec.encode(audio)  

        audio_codes = encoder_outputs['audio_codes'].long()
        audio_scales = encoder_outputs['audio_scales']


        codes = audio_codes.squeeze(0).permute(0, 2, 1)
        codes = self.code_embedding(codes) # (B, T, Q, D) 
        codes = codes.mean(dim=2) # (B, T, D)
        
        # =========================
        # POOL + PROJECT
        # =========================
        h_audio = self.audio_pool(codes)     # (B, D)
        z_audio = self.audio_proj(h_audio)   # (B, proj_dim)

        # =========================
        # RECONSTRUCTION
        # =========================
        x_recon = self.encodec.decode(audio_codes, audio_scales)['audio_values']
        x_recon = torch.tanh(x_recon).clamp(-1.0, 1.0)

        # =========================
        # CHORD BRANCH
        # =========================
        z_chord, h_chord = self.chord_encoder(chord)

        return {
            "x_recon": x_recon,
            "z_audio": z_audio,
            "z_chord": z_chord,
            "h_audio": h_audio,
            "h_chord": h_chord
        }