import streamlit as st
import numpy as np
import librosa
import joblib
import os
from tensorflow.keras.models import load_model

@st.cache_resource
def load_assets():
    scaler = joblib.load("model/scaler.pkl")
    selector = joblib.load("model/feature_selector.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    model = load_model("model/emotion_model.h5")
    return scaler, selector, encoder, model

def summarize(matrix):
    stats = [np.mean(matrix, axis=1), np.std(matrix, axis=1),
             np.min(matrix, axis=1), np.max(matrix, axis=1)]
    return np.concatenate(stats)

def get_audio_features(file_path):
    try:
        signal, sample_rate = librosa.load(file_path, sr=22050)

        stft = np.abs(librosa.stft(signal))

        # Primary features
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20)
        if mfcc.shape[1] == 0:
            return None
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        # Supplementary features
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sample_rate)
        zcr = librosa.feature.zero_crossing_rate(signal)
        rms = librosa.feature.rms(y=signal)

        # Check
        for feat in [chroma, contrast, mel_spec, tonnetz]:
            if feat.shape[1] == 0:
                return None

        combined = np.concatenate([
            summarize(np.vstack([mfcc, d1, d2])),
            summarize(chroma),
            summarize(contrast),
            summarize(mel_spec),
            summarize(tonnetz),
            summarize(zcr),
            summarize(rms)
        ])
        return combined
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Streamlit layout
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: teal;'>Speech Emotion Classifier</h1>"
    "<p style='text-align: center;'>Upload a WAV audio file to detect emotional tone</p><hr>",
    unsafe_allow_html=True
)

uploaded_audio = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_audio:
    st.audio(uploaded_audio, format='audio/wav')
    with open("temp_audio.wav", "wb") as audio_file:
        audio_file.write(uploaded_audio.read())

    with st.spinner("Analyzing emotion..."):
        scaler, selector, encoder, model = load_assets()
        features = get_audio_features("temp_audio.wav")

        if features is not None:
            scaled_input = scaler.transform([features])
            selected_input = selector.transform(scaled_input)
            probabilities = model.predict(selected_input)
            predicted_index = np.argmax(probabilities)
            predicted_label = encoder.inverse_transform([predicted_index])[0]
            confidence = probabilities[0][predicted_index] * 100

            st.markdown("---")
            st.markdown(
                f"<h2 style='text-align: center;'>{predicted_label.upper()}</h2>"
                f"<p style='text-align: center;'>Confidence: {confidence:.2f}%</p>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Failed to extract valid features. Try using a different audio clip.")

    os.remove("temp_audio.wav")
else:
    st.info("Upload a WAV file above to begin.")
