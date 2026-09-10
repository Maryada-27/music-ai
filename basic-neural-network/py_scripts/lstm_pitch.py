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

    def __init__(self, songs, seq_len=SEQ_LEN, stride=1):
        self.seq_len = seq_len
        self.songs = [torch.as_tensor(s, dtype=torch.long) for s in songs if len(s) > seq_len]
        # ponytail: flat numpy index instead of a list of tuples -- MAESTRO has
        # millions of windows and Python tuples cost ~60 bytes each.
        # Every position is already a target, so stride-1 windows make each note
        # a target `seq_len` times per epoch. stride=8 keeps 4x coverage at 1/8 the cost.
        starts = [np.arange(0, len(s) - seq_len, stride) for s in self.songs]
        self.song_idx = np.repeat(np.arange(len(self.songs)), [len(a) for a in starts])
        self.starts = np.concatenate(starts) if starts else np.empty(0, int)

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

    def __init__(self, embed_dim=128, hidden_size=512, num_layers=2, dropout=0.1):
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


class PitchLSTMProj(nn.Module):
    """Embedding -> LSTM -> Linear -> LSTM -> Linear. Interleaved variant of the above.

    The projection is nonlinear on purpose: a plain Linear between two LSTMs gets
    absorbed into the next layer's input weights and buys nothing but parameters.
    """

    def __init__(self, embed_dim=128, hidden_size=512, proj_size=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(NUM_PITCHES, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, proj_size), nn.GELU(),
                                  nn.Dropout(dropout))
        self.lstm2 = nn.LSTM(proj_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, NUM_PITCHES)

    def forward(self, pitch_seq):
        out, _ = self.lstm1(self.embed(pitch_seq))
        out, _ = self.lstm2(self.proj(out))
        return self.head(out)  # (batch, seq_len, NUM_PITCHES)


class PitchTransformer(nn.Module):
    """Causal transformer over the same pitch sequence, sized to match PitchLSTM.

    Defaults land at ~3.4M parameters so the comparison against the LSTM is about
    architecture rather than capacity.
    """

    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(NUM_PITCHES, d_model)
        # Attention is permutation-invariant, so order has to be supplied explicitly.
        self.pos = nn.Embedding(SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, NUM_PITCHES)

    def forward(self, pitch_seq):
        seq_len = pitch_seq.size(1)
        positions = torch.arange(seq_len, device=pitch_seq.device)
        x = self.embed(pitch_seq) + self.pos(positions)
        # Mandatory: without the mask, position t attends to t+1, which is its
        # own target. That leak reads as ~100% accuracy and means nothing.
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=pitch_seq.device)
        return self.head(self.encoder(x, mask=mask, is_causal=True))


ARCHITECTURES = {'stacked': PitchLSTM, 'proj': PitchLSTMProj, 'xformer': PitchTransformer}


# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (last-position loss, all-position loss, top-1 accuracy, octave accuracy).

    Accuracy and the first loss are scored at the final position only -- that is
    the prediction generate() makes, and it keeps these numbers comparable to
    earlier runs. The all-position loss is the like-for-like partner of the
    training loss, which averages over every position in the window.
    """
    model.eval()
    loss_last, loss_all, correct, oct_correct, seen, seen_all = 0.0, 0.0, 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_all += criterion(logits.reshape(-1, NUM_PITCHES), y.reshape(-1)).item() * y.numel()
        seen_all += y.numel()

        last, y_last = logits[:, -1], y[:, -1]
        pred = last.argmax(1)
        loss_last += criterion(last, y_last).item() * len(y_last)
        correct += (pred == y_last).sum().item()
        # octave = pitch // 12. High octave acc with low pitch acc means the
        # pitch class (pitch % 12) is where the remaining error lives.
        oct_correct += (pred // 12 == y_last // 12).sum().item()
        seen += len(y_last)
    return (loss_last / max(seen, 1), loss_all / max(seen_all, 1),
            correct / max(seen, 1), oct_correct / max(seen, 1))


@torch.no_grad()
def accuracy_by_position(model, loader, device, out='accuracy_by_position.png'):
    """Top-1 accuracy at each position in the window, i.e. by amount of context.

    Position 1 predicts from a single note, the last from a full window. A curve
    still climbing at the end means SEQ_LEN is the binding constraint and longer
    windows will pay; a flat tail means they will not.
    """
    model.eval()
    correct, seen = None, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        hits = (model(x).argmax(-1) == y).sum(0)
        correct = hits if correct is None else correct + hits
        seen += len(y)
    accs = (correct / max(seen, 1)).tolist()
    marks = sorted({2 ** i for i in range(len(accs).bit_length())} & set(range(1, len(accs) + 1))
                   | {len(accs)})
    print("Accuracy by context length: " + "  ".join(f"{m}:{accs[m - 1]:.1%}" for m in marks))

    # ponytail: a failed plot must not lose a finished training run, hence the
    # broad except. Agg so it works headless on Kaggle/Colab.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(accs) + 1), [a * 100 for a in accs], marker='o', markersize=3)
        plt.xlabel('notes of context')
        plt.ylabel('top-1 accuracy (%)')
        plt.title('Next-pitch accuracy vs context length (test split)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Wrote {out}")
    except Exception as exc:
        print(f"Plot skipped: {exc}")
    return accs


def train(model, train_loader, val_loader, epochs, device, lr=1e-3, patience=5,
          optimizer_name='adam', clip_norm=1.0):
    """Trains up to `epochs`, stopping once val loss stalls and restoring the best weights."""
    # ponytail: label_smoothing=0.1 was measured and removed. It bought +0.5 top-1
    # (42.42% -> 42.90%) and cost perplexity (8.9 -> 9.0), which is the headline
    # metric here. Flattening the target necessarily raises NLL. Don't re-add it.
    criterion = nn.CrossEntropyLoss()
    # Adam's weight_decay is L2 folded into the gradient, so the adaptive scaling
    # then shrinks it for exactly the weights that need it most. AdamW decouples
    # it, which is why it is the standard transformer optimizer. 1e-5 alongside
    # plain Adam was close enough to no regularization at all.
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # ponytail: halve the LR whenever val loss stalls for 2 epochs. Paired with
    # patience=5 below, so the LR gets two chances to rescue a plateau before
    # training stops. Swap for CosineAnnealingLR if you want a fixed budget.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
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
            if optimizer_name == 'adamw':
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            seen += y.numel()

        train_secs = time.perf_counter() - epoch_start
        val_loss, val_all, val_acc, val_oct = evaluate(model, val_loader, criterion, device)
        val_secs = time.perf_counter() - epoch_start - train_secs
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | TRAIN loss {total_loss / seen:.4f} "
              f"pitch {correct / seen:.2%} | VAL loss {val_loss:.4f} ppl {math.exp(val_loss):.1f} "
              f"pitch {val_acc:.2%} oct {val_oct:.2%} | all-pos loss {val_all:.4f} "
              f"lr {optimizer.param_groups[0]['lr']:.1e} | {train_secs:.0f}s train {val_secs:.0f}s val")

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

def smoke_test(device, model_cls=None):
    """Trains on a repeating scale. The model must learn it near-perfectly."""
    scale = np.tile([60, 62, 64, 65, 67, 69, 71, 72], 300)
    loader = data.DataLoader(NextPitchDataset([scale]), batch_size=128, shuffle=True)
    model = (model_cls or PitchLSTM)().to(device)
    criterion = train(model, loader, loader, epochs=5, device=device)
    _, _, acc, _ = evaluate(model, loader, criterion, device)
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
    parser.add_argument('--optimizer', choices=('adam', 'adamw'), default='adam',
                        help='adamw also enables gradient clipping and weight_decay 0.01; '
                             'adam is the default so earlier results stay reproducible')
    parser.add_argument('--arch', choices=('stacked', 'proj', 'xformer'), default='stacked',
                        help='stacked = 2-layer LSTM; proj = LSTM-Linear-LSTM; '
                             'xformer = causal transformer')
    parser.add_argument('--stride', type=int, default=8,
                        help='window stride; 1 = every position (8x slower, ~no gain)')
    parser.add_argument('--temp', type=float, default=1.0, help='generation temperature')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true', help='run the synthetic self-check and exit')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.smoke:
        smoke_test(device, ARCHITECTURES[args.arch])
        return

    torch.manual_seed(42)
    if not 1 <= args.num_years <= len(ALL_YEARS):
        parser.error(f"--num-years must be between 1 and {len(ALL_YEARS)}")
    years = ALL_YEARS[-args.num_years:]
    print(f"Years -> {', '.join(years)}")
    train_songs, val_songs, test_songs = split_songs(load_pitches(years))
    print(f"Songs -> train {len(train_songs)}, val {len(val_songs)}, test {len(test_songs)}")

    loader = lambda songs, shuffle: data.DataLoader(
        NextPitchDataset(songs, stride=args.stride), batch_size=args.batch_size, shuffle=shuffle)
    train_loader, val_loader = loader(train_songs, True), loader(val_songs, False)

    # Context for the final perplexity: uniform over 128 pitches scores 128, and
    # this scores what you get from the note frequencies alone, ignoring order.
    counts = np.bincount(np.concatenate(train_songs), minlength=NUM_PITCHES)
    probs = counts[counts > 0] / counts.sum()
    print(f"Baselines -> uniform perplexity {NUM_PITCHES}, "
          f"unigram perplexity {math.exp(-(probs * np.log(probs)).sum()):.1f}")

    model = ARCHITECTURES[args.arch]().to(device)
    tag = f'{args.arch}_{args.optimizer}'
    print(f"Model -> {args.arch}, {args.optimizer}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    criterion = train(model, train_loader, val_loader, args.epochs, device,
                      optimizer_name=args.optimizer)
    # ponytail: every output is named after the architecture. Runs that shared
    # a filename have already clobbered each other's weights and logs twice.
    torch.save(model.state_dict(), f'pitch_{tag}.pt')
    print(f"Saved best weights to pitch_{tag}.pt")

    # ponytail: the test loader is built here, after training -- on purpose.
    # Nothing above this line can read the test split.
    test_loader = loader(test_songs, False)
    test_loss, test_all, test_acc, test_oct = evaluate(model, test_loader, criterion, device)
    accuracy_by_position(model, test_loader, device, out=f'accuracy_{tag}.png')
    print(f"\n=== FINAL TEST (held out) ===\nloss {test_loss:.4f} | "
          f"perplexity {math.exp(test_loss):.1f} | pitch acc {test_acc:.2%} | "
          f"octave acc {test_oct:.2%} | all-pos loss {test_all:.4f}")

    generate(model, test_songs[0][:SEQ_LEN], device, temperature=args.temp,
             out=f'generated_{tag}.mid')


if __name__ == '__main__':
    main()
