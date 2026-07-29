import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel

from model.attention_weighted_pooling import AttentionWeightedPooling
from model.projection import ProjectionHead
from model.chordbeat_encoder import ChordBeatEncoder


class MusicConRec(nn.Module):

    def __init__(
        self,
        codebook_size=1024,
        feature_dim=128,
        proj_dim=128,
        queue_size=4096,
        momentum=0.999,
        train_encodec=False,
    ):
        super().__init__()

        self.momentum = momentum

        # =========================
        # AUDIO ENCODER
        # =========================

        self.encodec = EncodecModel.from_pretrained(
            "facebook/encodec_24khz"
        )

        if not train_encodec:
            for p in self.encodec.parameters():
                p.requires_grad = False


        self.code_embedding = nn.Embedding(
            codebook_size,
            feature_dim
        )

        self.audio_pool = AttentionWeightedPooling(
            feature_dim
        )

        self.audio_proj = ProjectionHead(
            feature_dim,
            out_dim=proj_dim
        )


        # =========================
        # CHORD ENCODER
        # =========================

        self.chord_encoder = ChordBeatEncoder(
            input_dim=13,
            d_model=feature_dim
        )


        # =========================
        # MOMENTUM COPIES
        # =========================

        self.encodec_k = copy.deepcopy(self.encodec)
        self.code_embedding_k = copy.deepcopy(self.code_embedding)
        self.audio_pool_k = copy.deepcopy(self.audio_pool)
        self.audio_proj_k = copy.deepcopy(self.audio_proj)
        self.chord_encoder_k = copy.deepcopy(self.chord_encoder)


        for module in [
            self.encodec_k,
            self.code_embedding_k,
            self.audio_pool_k,
            self.audio_proj_k,
            self.chord_encoder_k
        ]:
            for p in module.parameters():
                p.requires_grad = False


        # =========================
        # QUEUE
        # =========================

        self.register_buffer(
            "queue_audio",
            torch.zeros(queue_size, proj_dim)
        )

        self.register_buffer(
            "queue_chord",
            torch.zeros(queue_size, proj_dim)
        )

        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )

        self.register_buffer(
            "queue_initialized",
            torch.tensor(False)
        )


    def flatten_lstm(self):

        for module in self.modules():

            if isinstance(module, nn.LSTM):
                module.flatten_parameters()


    def _encode_audio(
        self,
        audio,
        encodec,
        code_embedding,
        audio_pool,
        audio_proj
    ):

        self.flatten_lstm()

        outputs = encodec.encode(audio)

        codes = outputs["audio_codes"].long()
        scales = outputs["audio_scales"]

        codes = codes.squeeze(0).permute(0,2,1)

        codes = code_embedding(codes)

        codes = codes.mean(dim=2)

        h = audio_pool(codes)

        z = audio_proj(h)

        return z, h, codes, scales



    def forward(self, audio, chord):

        z_audio, h_audio, audio_codes, scales = self._encode_audio(
            audio,
            self.encodec,
            self.code_embedding,
            self.audio_pool,
            self.audio_proj
        )

        z_audio = F.normalize(z_audio, dim=-1)


        x_recon = self.encodec.decode(
            audio_codes,
            scales
        )["audio_values"]


        z_chord, h_chord = self.chord_encoder(chord)

        z_chord = F.normalize(
            z_chord,
            dim=-1
        )


        return {
            "x_recon": x_recon,
            "z_audio": z_audio,
            "z_chord": z_chord,
            "h_audio": h_audio,
            "h_chord": h_chord
        }



    @torch.no_grad()
    def forward_momentum(self, audio, chord):

        k_audio,_,_,_ = self._encode_audio(
            audio,
            self.encodec_k,
            self.code_embedding_k,
            self.audio_pool_k,
            self.audio_proj_k
        )

        k_audio = F.normalize(
            k_audio,
            dim=-1
        )

        k_chord,_ = self.chord_encoder_k(chord)

        k_chord = F.normalize(
            k_chord,
            dim=-1
        )

        return k_audio,k_chord



    @torch.no_grad()
    def update_momentum_encoders(self):

        pairs = [
            (self.encodec,self.encodec_k),
            (self.code_embedding,self.code_embedding_k),
            (self.audio_pool,self.audio_pool_k),
            (self.audio_proj,self.audio_proj_k),
            (self.chord_encoder,self.chord_encoder_k)
        ]

        for online,target in pairs:

            for p,q in zip(
                online.parameters(),
                target.parameters()
            ):

                q.data.mul_(self.momentum)
                q.data.add_(
                    (1-self.momentum)*p.data
                )



    @torch.no_grad()
    def update_moco_queue(
        self,
        k_audio,
        k_chord
    ):

        k_audio = k_audio.detach()
        k_chord = k_chord.detach()


        if not self.queue_initialized:

            self.queue_audio[:len(k_audio)] = k_audio
            self.queue_chord[:len(k_chord)] = k_chord
            self.queue_initialized.fill_(True)

            return


        ptr=int(self.queue_ptr)

        batch=len(k_audio)

        self.queue_audio[
            ptr:ptr+batch
        ] = k_audio


        self.queue_chord[
            ptr:ptr+batch
        ] = k_chord


        self.queue_ptr[0]=(ptr+batch)%self.queue_audio.size(0)