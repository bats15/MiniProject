import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ======================
# CONFIG
# ======================
TRAIN_CSV = r"C:/Documents/CODE Official/daicwoz/DAIC_woz/train_split_Depression_AVEC2017.csv"
DEV_CSV = r"C:/Documents/CODE Official/daicwoz/DAIC_woz/dev_split_Depression_AVEC2017.csv"
WAV_DIR = r"C:/Documents/CODE Official/daicwoz/DAIC_woz/.wav"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 20
LR = 3e-4

# ======================
# DATASET
# ======================
class DAICDataset(Dataset):
    def __init__(self, csv_path, wav_dir):
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir

        available_files = os.listdir(wav_dir)

        valid_rows = []
        for _, row in self.df.iterrows():
            pid = str(int(row["Participant_ID"]))
            match = [f for f in available_files if f.startswith(pid)]
            if match:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pid = int(row["Participant_ID"])
        label = int(row["PHQ8_Binary"])

        files = os.listdir(self.wav_dir)

        match = [f for f in files if f.startswith(str(pid))]
        if not match:
            raise FileNotFoundError(f"No file for {pid}")

        wav_path = os.path.join(self.wav_dir, match[0])

        audio, sr = librosa.load(wav_path, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # NEW: normalize
        audio = audio / (abs(audio).max() + 1e-6)

        chunk_size = 16000 * 10  # 10 seconds

        if len(audio) > chunk_size:
            audio = audio[:16000 * 10]
        
        return audio, label

# ======================
# COLLATE (padding)
# ======================
def collate_fn(batch):
    audios, labels = zip(*batch)
    return list(audios), torch.tensor(labels)

# ======================
# MODEL
# ======================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        # freeze backbone (important for small dataset)
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        # NEW (stronger learning)
        for p in self.wav2vec.encoder.layers[-4:].parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        device = next(self.parameters()).device

        # list of numpy arrays
        x = [a for a in x]

        inputs = self.processor(
            x,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)

        outputs = self.wav2vec(input_values).last_hidden_state

        x = outputs.max(dim=1).values

        return self.classifier(x)
# ======================
# TRAIN
# ======================
def train():
    best_acc = 0

    train_ds = DAICDataset(TRAIN_CSV, WAV_DIR)
    dev_ds = DAICDataset(DEV_CSV, WAV_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    model = Model().to(DEVICE)

    # ===== class imbalance handling =====
    labels = train_ds.df["PHQ8_Binary"].astype(int).to_numpy()
    pos_weight = (len(labels) - labels.sum()) / labels.sum()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ===== training loop =====
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for audio, label in train_loader:
            label = label.float().unsqueeze(1).to(DEVICE)

            output = model(audio)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # ===== evaluation =====
        acc = evaluate(model, dev_loader)

        # ===== save best model =====
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Saved new best model (acc={acc:.4f})")
# ======================
# EVAL
# ======================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, label in loader:
            label = label.to(DEVICE)

            output = model(audio)
            pred = (torch.sigmoid(output) > 0.5).int().squeeze()

            correct += (pred == label).sum().item()
            total += label.size(0)

    acc = correct / total
    print(f"Dev Accuracy: {acc:.4f}")
    return acc

# ======================
# RUN
# ======================
if __name__ == "__main__":
    train()