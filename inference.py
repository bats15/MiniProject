import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# MODEL (same as training)
# ======================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

        # freeze backbone
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        device = next(self.parameters()).device

        inputs = self.processor(
            x,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = self.wav2vec(input_values).last_hidden_state

        x = outputs.mean(dim=1)

        return self.classifier(x)


# ======================
# LOAD MODEL
# ======================
model = Model().to(DEVICE)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()


# ======================
# PREDICT FUNCTION
# ======================
def predict_full(wav_path):
    audio, _ = librosa.load(wav_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = audio / (abs(audio).max() + 1e-6)

    chunk = 16000 * 10
    probs = []

    for i in range(0, len(audio), chunk):
        seg = audio[i:i+chunk]
        if len(seg) < 16000:
            continue

        with torch.no_grad():
            out = model([seg])
            probs.append(torch.sigmoid(out).item())

    return sum(probs)/len(probs) if probs else 0.5


# ======================
# RUN
# ======================
if __name__ == "__main__":
    path = input("Enter audio path: ")

    prob = predict(path)

    print(f"Probability: {prob:.4f}")

    if prob > 0.5:
        print("Prediction: Depressed")
    else:
        print("Prediction: Not Depressed")