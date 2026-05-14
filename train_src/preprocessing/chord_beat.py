import torch

PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "6":    [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
}


def _chord_to_multihot(chord_str: str) -> torch.Tensor:
    vec = torch.zeros(12)
    if not chord_str:
        return vec

    if "/" in chord_str:
        chord_part, bass_part = chord_str.split("/")
        bass_pc = PITCH_CLASS.get(bass_part)
    else:
        chord_part = chord_str
        bass_pc = None

    root = (
        chord_part[:2]
        if chord_part[:2] in PITCH_CLASS
        else chord_part[0]
    )

    quality = chord_part[len(root):] or "maj"
    root_pc = PITCH_CLASS.get(root)
    if root_pc is None:
        return vec

    if quality == "m":
        quality = "min"
    if quality == "m6":
        quality = "min6"

    intervals = CHORD_TEMPLATES.get(quality, CHORD_TEMPLATES["maj"])
    for interval in intervals:
        vec[(root_pc + interval) % 12] = 1.0

    if bass_pc is not None:
        vec[bass_pc] = 1.0

    return vec


def _encode_chords_multihot(chords):
    if not chords:
        return torch.zeros(0, 12)
    return torch.stack([_chord_to_multihot(c) for c in chords])


def _encode_beats(beats):
    if not beats:
        return torch.zeros(0, 1)

    time_sig = max(beats)
    if time_sig == 0:
        return torch.zeros(len(beats), 1)

    beat_positions = [(b) / time_sig for b in beats]
    return torch.tensor(beat_positions, dtype=torch.float32).unsqueeze(1)


def chord_beat(chords, beats, chord_times, beat_times, num_chunks=50, duration=10):      
    chords = _encode_chords_multihot(chords)
    beats = _encode_beats(beats)
    chunk_size = duration / num_chunks

    grid = torch.zeros((num_chunks, 13))
    current_chord = torch.zeros(12)
    chord_ptr = 0
    beat_ptr = 0

    for i in range(num_chunks):
        window_start = i * chunk_size
        window_end = (i + 1) * chunk_size

        while chord_ptr < len(chord_times) and chord_times[chord_ptr] <= window_start:
            current_chord = torch.as_tensor(chords[chord_ptr], dtype=torch.float32).clone().detach()
            chord_ptr += 1

        current_beat = 0.0
        while beat_ptr < len(beat_times) and beat_times[beat_ptr] < window_end:
            if beat_times[beat_ptr] >= window_start:
                current_beat = float(beats[beat_ptr])
            beat_ptr += 1

        grid[i, 0:12] = current_chord
        grid[i, 12] = current_beat

    return grid
