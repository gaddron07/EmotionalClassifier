# test_model.py

import numpy as np
import librosa
import joblib
import sys
from tensorflow.keras.models import load_model

# Load trained components
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model = load_model("emotion_model.h5")

def summarize_feature(feature):
    return np.hstack([
        np.mean(feature, axis=1),
        np.std(feature, axis=1),
        np.min(feature, axis=1),
        np.max(feature, axis=1)
    ])

def extract_features(file_path, sr=22050):
    try:
        audio, _ = librosa.load(file_path, sr=sr)

        stft = np.abs(librosa.stft(audio))
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(audio)

        features = np.hstack([
            summarize_feature(np.vstack([mfcc, delta, delta2])),
            summarize_feature(chroma),
            summarize_feature(contrast),
            summarize_feature(mel),
            summarize_feature(tonnetz),
            summarize_feature(zcr),
            summarize_feature(rms)
        ])
        return features
    except Exception as e:
        print(f"[ERROR] Failed to extract features: {e}")
        return None

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    if features is None:
        print("[ERROR] Feature extraction failed.")
        return

    # Preprocessing
    features_scaled = scaler.transform([features])
    features_selected = selector.transform(features_scaled)

    # Prediction
    prediction = model.predict(features_selected)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = float(np.max(prediction)) * 100

    print(f"\n🎧 Predicted Emotion: {predicted_label[0].upper()}")
    print(f"🧠 Confidence Score: {confidence:.2f}%")

# -------- MAIN --------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <path_to_audio.wav>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    predict_emotion(audio_file_path)
