"""Pitch-only next-token LSTM on MAESTRO. Semester project version.

Predicts the next MIDI pitch from the previous SEQ_LEN pitches. No step, no
duration -- see train_ddp_v3.py if you want the full multi-head model.

    python lstm_pitch.py --smoke              # synthetic self-check, no download
    python lstm_pitch.py --epochs 10          # all 10 MAESTRO years, train, report, generate
    python lstm_pitch.py --num-years 1        # 2018 only -- fast, overfits
"""

import argparse
import copy
import math
import pathlib
import time
import zipfile

import numpy as np
import pretty_midi as pm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.hub import download_url_to_file

SEQ_LEN = 32
NUM_PITCHES = 128
DATA_DIR = pathlib.Path('data')
ALL_YEARS = ('2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018')


# ==========================================
# 1. DATASET DOWNLOADING & PREPROCESSING
# ==========================================

def download_maestro(dest_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    """Downloads and unzips the MAESTRO MIDI dataset if not already present."""
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_target = dest_dir / 'maestro-v3.0.0-midi.zip'
    extracted = dest_dir / 'maestro-v3.0.0'

    if not extracted.exists():
        if not zip_target.exists():
            print("Downloading MAESTRO...")
            download_url_to_file(url, str(zip_target), progress=True)
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

    return extracted


def midi_to_pitches(midi_path) -> np.ndarray:
    """Parses one MIDI file into a chronological array of MIDI pitch numbers."""
    midi = pm.PrettyMIDI(str(midi_path))
    if not midi.instruments:
        return np.empty(0, dtype=np.int64)
    # sort by (start, pitch) so simultaneous chord notes get a deterministic order
    notes = sorted(midi.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    return np.array([n.pitch for n in notes], dtype=np.int64)


def load_pitches(years=('2018',), dest_dir: pathlib.Path = DATA_DIR) -> list:
    """Returns one pitch array per song, caching the parsed result on disk."""
    cache = dest_dir / f"pitches_only_{'_'.join(years)}.npz"
    if cache.exists():
        with np.load(cache) as loaded:
            return [loaded[k] for k in loaded.files]

    root = download_maestro(dest_dir)
    midi_files = sorted(f for year in years for f in (root / year).glob('*.midi'))
    if not midi_files:
        raise FileNotFoundError(f"No .midi files under {root} for years {years}")

    print(f"Parsing {len(midi_files)} MIDI files...")
    songs = [p for p in map(midi_to_pitches, midi_files) if len(p) > SEQ_LEN]
    np.savez_compressed(cache, *songs)
    return songs


# ==========================================
# 2. PYTORCH DATASET & SPLITTING
# ==========================================

class NextPitchDataset(data.Dataset):
    """Sliding window over each song: SEQ_LEN pitches in, the same window shifted by one out."""

    def __init__(self, songs, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.songs = [torch.as_tensor(s, dtype=torch.long) for s in songs if len(s) > seq_len]
        # ponytail: flat numpy index instead of a list of tuples -- MAESTRO has
        # millions of windows and Python tuples cost ~60 bytes each.
        counts = [len(s) - seq_len for s in self.songs]
        self.song_idx = np.repeat(np.arange(len(self.songs)), counts)
        self.starts = np.concatenate([np.arange(c) for c in counts]) if counts else np.empty(0, int)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        song = self.songs[self.song_idx[idx]]
        start = self.starts[idx]
        # every position is a training example, not just the last one
        return song[start:start + self.seq_len], song[start + 1:start + 1 + self.seq_len]


def split_songs(songs, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Splits whole songs (never windows) into train/val/test.

    Splitting by window would leak overlapping context across the splits.
    """
    order = np.random.default_rng(seed).permutation(len(songs))
    train_end = int(len(order) * train_ratio)
    val_end = int(len(order) * (train_ratio + val_ratio))
    pick = lambda ids: [songs[i] for i in ids]
    return pick(order[:train_end]), pick(order[train_end:val_end]), pick(order[val_end:])


# ==========================================
# 3. MODEL
# ==========================================

class PitchLSTM(nn.Module):
    """Embedding -> 2-layer LSTM -> logits over the 128 MIDI pitches."""

    def __init__(self, embed_dim=64, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(NUM_PITCHES, embed_dim)
        # ponytail: nn.LSTM applies `dropout` between layers, so it only does
        # anything at num_layers >= 2. Drop back to 128/1 if this overfits or crawls.
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, NUM_PITCHES)

    def forward(self, pitch_seq):
        out, _ = self.lstm(self.embed(pitch_seq))
        return self.head(out)  # (batch, seq_len, NUM_PITCHES)


# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (mean loss, top-1 accuracy) for the next note after the window.

    Scored at the last position only -- that is the prediction generation uses,
    and it keeps the numbers comparable to the single-target version.
    """
    model.eval()
    total_loss, correct, oct_correct, seen = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y[:, -1].to(device)
        logits = model(x)[:, -1]
        pred = logits.argmax(1)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (pred == y).sum().item()
        # octave = pitch // 12. High octave acc with low pitch acc means the
        # pitch class (pitch % 12) is where the remaining error lives.
        oct_correct += (pred // 12 == y // 12).sum().item()
        seen += len(y)
    return total_loss / max(seen, 1), correct / max(seen, 1), oct_correct / max(seen, 1)


def train(model, train_loader, val_loader, epochs, device, lr=1e-3, weight_decay=1e-5, patience=5):
    """Trains up to `epochs`, stopping once val loss stalls and restoring the best weights."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, stale = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, NUM_PITCHES), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            seen += y.numel()

        train_secs = time.perf_counter() - epoch_start
        val_loss, val_acc, val_oct = evaluate(model, val_loader, criterion, device)
        val_secs = time.perf_counter() - epoch_start - train_secs
        print(f"Epoch {epoch:3d}/{epochs} | TRAIN loss {total_loss / seen:.4f} "
              f"pitch {correct / seen:.2%} | VAL loss {val_loss:.4f} ppl {math.exp(val_loss):.1f} "
              f"pitch {val_acc:.2%} oct {val_oct:.2%} | {train_secs:.0f}s train {val_secs:.0f}s val")

        # ponytail: plain early stopping on val loss. Raise `patience` if the
        # curve is noisy; add dropout/weight decay only if it still overfits.
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop: no val improvement for {patience} epochs "
                      f"(best {best_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return criterion


# ==========================================
# 5. GENERATION
# ==========================================

@torch.no_grad()
def generate(model, seed_pitches, device, num_notes=64, temperature=1.0, out='generated.mid'):
    """Samples `num_notes` pitches and writes them as a fixed-rhythm MIDI file."""
    model.eval()
    window = list(seed_pitches[-SEQ_LEN:])
    generated = []
    for _ in range(num_notes):
        x = torch.tensor([window[-SEQ_LEN:]], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[0, -1] / temperature, dim=-1)
        pitch = int(torch.multinomial(probs, 1))
        window.append(pitch)
        generated.append(pitch)

    # ponytail: fixed 0.25s per note -- this model predicts pitch only.
    # Add a duration head (see train_ddp_v3.py) if the rhythm needs to be real.
    midi = pm.PrettyMIDI()
    piano = pm.Instrument(program=0)
    for i, pitch in enumerate(generated):
        piano.notes.append(pm.Note(velocity=100, pitch=pitch, start=i * 0.25, end=i * 0.25 + 0.25))
    midi.instruments.append(piano)
    midi.write(out)
    print(f"Wrote {num_notes} generated notes to {out}")
    return generated


# ==========================================
# 6. SMOKE TEST
# ==========================================

def smoke_test(device):
    """Trains on a repeating scale. The model must learn it near-perfectly."""
    scale = np.tile([60, 62, 64, 65, 67, 69, 71, 72], 300)
    loader = data.DataLoader(NextPitchDataset([scale]), batch_size=128, shuffle=True)
    model = PitchLSTM().to(device)
    criterion = train(model, loader, loader, epochs=5, device=device)
    _, acc, _ = evaluate(model, loader, criterion, device)
    assert acc > 0.9, f"smoke test failed: accuracy {acc:.2%} on a repeating scale"
    print(f"Smoke test passed: {acc:.2%}")


# ==========================================
# 7. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-years', type=int, default=len(ALL_YEARS),
                        help=f'how many MAESTRO years to train on, most recent first '
                             f'(1-{len(ALL_YEARS)}, default all)')
    parser.add_argument('--temp', type=float, default=1.0, help='generation temperature')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true', help='run the synthetic self-check and exit')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.smoke:
        smoke_test(device)
        return

    torch.manual_seed(42)
    if not 1 <= args.num_years <= len(ALL_YEARS):
        parser.error(f"--num-years must be between 1 and {len(ALL_YEARS)}")
    years = ALL_YEARS[-args.num_years:]
    print(f"Years -> {', '.join(years)}")
    train_songs, val_songs, test_songs = split_songs(load_pitches(years))
    print(f"Songs -> train {len(train_songs)}, val {len(val_songs)}, test {len(test_songs)}")

    loader = lambda songs, shuffle: data.DataLoader(
        NextPitchDataset(songs), batch_size=args.batch_size, shuffle=shuffle)
    train_loader, val_loader = loader(train_songs, True), loader(val_songs, False)

    model = PitchLSTM().to(device)
    criterion = train(model, train_loader, val_loader, args.epochs, device)

    # ponytail: the test loader is built here, after training -- on purpose.
    # Nothing above this line can read the test split.
    test_loss, test_acc, test_oct = evaluate(model, loader(test_songs, False), criterion, device)
    print(f"\n=== FINAL TEST (held out) ===\nloss {test_loss:.4f} | "
          f"perplexity {math.exp(test_loss):.1f} | pitch acc {test_acc:.2%} | "
          f"octave acc {test_oct:.2%}")

    generate(model, test_songs[0][:SEQ_LEN], device, temperature=args.temp)


if __name__ == '__main__':
    main()
