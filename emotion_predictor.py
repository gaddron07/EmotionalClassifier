import joblib
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Emotion code map (RAVDESS-style)
EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Statistical summarizer
def summarize(matrix):
    return np.concatenate([
        np.mean(matrix, axis=1),
        np.std(matrix, axis=1),
        np.min(matrix, axis=1),
        np.max(matrix, axis=1)
    ])

# Audio feature generator
def extract_audio_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=22050)
        stft = np.abs(librosa.stft(y))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        if mfcc.shape[1] == 0:
            return None
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)

        for feat in [chroma, contrast, mel, tonnetz]:
            if feat.shape[1] == 0:
                return None

        feature_vec = np.concatenate([
            summarize(np.vstack([mfcc, delta1, delta2])),
            summarize(chroma),
            summarize(contrast),
            summarize(mel),
            summarize(tonnetz),
            summarize(zcr),
            summarize(rms)
        ])
        return feature_vec
    except Exception as err:
        print(f"[Error] Failed to process {filepath}: {err}")
        return None

# Load model artifacts
def load_inference_assets():
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    selector = joblib.load("feature_selector.pkl")
    model = load_model("emotion_model.h5")
    return scaler, selector, encoder, model

# Run prediction
def predict_emotion(audio_file):
    scaler, selector, encoder, model = load_inference_assets()
    features = extract_audio_features(audio_file)
    if features is None:
        print("⚠️ Could not extract valid features from this file.")
        return

    scaled = scaler.transform([features])
    reduced = selector.transform(scaled)
    preds = model.predict(reduced)
    label = encoder.inverse_transform([np.argmax(preds)])[0]
    confidence = float(np.max(preds)) * 100
    print(f"\n🎧 Predicted Emotion: {label.upper()} ({confidence:.2f}% confidence)")

# Run if main
if __name__ == "__main__":
    print("\n🗣️  Speech Emotion Detection")
    print("----------------------------")
    audio_path = input("📁 Enter the path to a .wav audio file: ").strip()

    if not audio_path.lower().endswith(".wav"):
        print("❌ Only .wav files are supported.")
    elif not os.path.exists(audio_path):
        print("❌ File does not exist. Check your path.")
    else:
        predict_emotion(audio_path)
