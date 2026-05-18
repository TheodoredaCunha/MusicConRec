import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import os
import json

from transformers import EncodecModel, AutoProcessor

from preprocessing.audio_preprocess import resample
from preprocessing.chord_beat import chord_beat

class MusicBenchDataset(Dataset):
    def __init__(self, dataset_dir, audio_dir):
        self.data_dir = audio_dir
        self.dataset_dir = dataset_dir
        # self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

        with open(self.dataset_dir, "r") as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]

        # =========================
        # AUDIO
        # =========================
        audio_path = os.path.join(self.data_dir, item["location"])
        waveform, sr = torchaudio.load(audio_path, normalize=False)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        # ensure shape (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)


        

        # =========================
        # CHORD-BEAT
        # =========================
        chords = item["chords"]
        chord_times = item["chords_time"]
        beats = item["beats"][1]
        beat_times = item["beats"][0]

        chord_beat_representation = chord_beat(chords, beats, chord_times, beat_times)

        return waveform, chord_beat_representation


def collate_fn(batch):
    waveforms, chord_beats = zip(*batch)

    # find max length
    max_len = max(w.shape[-1] for w in waveforms)

    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len - w.shape[-1]
        w = F.pad(w, (0, pad_len))  # pad time dimension
        padded_waveforms.append(w)

    waveforms = torch.stack(padded_waveforms)  # (B, 1, T)

    chord_beats = torch.nn.utils.rnn.pad_sequence(
        chord_beats,
        batch_first=True
    )

    return waveforms, chord_beats