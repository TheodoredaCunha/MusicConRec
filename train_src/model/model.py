import torch
import torch.nn as nn
from transformers import EncodecModel
from model.attention_weighted_pooling import AttentionWeightedPooling
from model.projection import ProjectionHead
from model.chordbeat_encoder import ChordBeatEncoder


class MusicConRec(nn.Module):
    def __init__(self, codebook_size=1024, feature_dim=128, proj_dim=128):
        super().__init__()

        self.encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")
        for param in self.encodec.parameters():
            param.requires_grad = False

        self.code_embedding = nn.Embedding(codebook_size, feature_dim)

        self.audio_pool = AttentionWeightedPooling(feature_dim)
        self.audio_proj = ProjectionHead(feature_dim, out_dim=proj_dim)

        self.chord_encoder = ChordBeatEncoder(
            input_dim=13,
            d_model=feature_dim
        )

    def forward(self, audio, chord):
        encoder_outputs = self.encodec.encode(audio)

        audio_codes = encoder_outputs['audio_codes'].long()
        audio_scales = encoder_outputs['audio_scales']

        codes = audio_codes.squeeze(0).permute(0, 2, 1)
        codes = self.code_embedding(codes)
        codes = codes.mean(dim=2)

        h_audio = self.audio_pool(codes)
        z_audio = self.audio_proj(h_audio)

        x_recon = self.encodec.decode(audio_codes, audio_scales)['audio_values']

        z_chord, h_chord = self.chord_encoder(chord)

        return {
            "x_recon": x_recon,
            "z_audio": z_audio,
            "z_chord": z_chord,
            "h_audio": h_audio,
            "h_chord": h_chord
        }
