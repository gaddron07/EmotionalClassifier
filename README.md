**Emotion Detection from Speech using Deep Learning**

This project focuses on detecting human emotions from audio recordings using deep learning techniques and audio signal processing. It uses a combination of extracted audio features and a trained neural network to classify emotional states accurately.

---

**Overview**

Understanding emotions from speech is useful in many areas like:

* Voice assistants and conversational AI
* Customer sentiment monitoring in call centers
* Mental health monitoring
* Human-computer interaction

The system takes a `.wav` file as input and predicts the emotion expressed in the audio along with a confidence score.

---

**Dataset**

We used the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), which contains:

* 24 actors (12 male and 12 female)
* 8 emotional classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised

Dataset Source: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

---

**Feature Extraction**

Features are extracted using the Librosa library. The following features are computed from the audio signals:

* MFCCs (Mel-frequency cepstral coefficients), delta, and delta2
* Chroma features
* Spectral contrast
* Mel spectrogram
* Tonnetz (tonal centroid features)
* Zero crossing rate
* Root Mean Square (RMS) energy

Each feature type is summarized using the mean, standard deviation, minimum, and maximum values.

**Data Augmentation Techniques:**

* Pitch shifting
* Time stretching
* Adding random noise

---

**Preprocessing**

* StandardScaler is used for feature normalization.
* SelectKBest (ANOVA F-test) is used to keep the top 300 most relevant features.
* Labels are encoded using LabelEncoder.
* Class balancing is done through oversampling to ensure each emotion has equal representation.

---

**Model Architecture**

The model is built using Keras (TensorFlow backend) as a fully connected neural network with the following layers:

* Dense layer with 768 units and ReLU activation
* BatchNormalization and Dropout (0.4)
* Dense layer with 384 units and ReLU activation
* BatchNormalization and Dropout (0.3)
* Dense layer with 192 units and ReLU activation
* Dropout (0.2)
* Final Dense layer with softmax activation for 8 emotion classes

Other configurations:

* Optimizer: Adam (learning rate 0.0005)
* Loss: Categorical Crossentropy
* EarlyStopping and ReduceLROnPlateau callbacks
* 200 epochs with early stopping and batch size 128

---

**Results**

Final model performance:

* Test Accuracy: 80.00%
* Weighted F1 Score: 80.00%
* A confusion matrix and per-class accuracy are also included

---

**Project Files**

* emotion\_model.h5 : Trained model
* scaler.pkl : StandardScaler used during preprocessing
* label\_encoder.pkl : Label encoder for emotion labels
* feature\_selector.pkl : SelectKBest feature selector
* emotion\_features\_dataset.pkl : Final processed dataset with features and labels
* Audio\_Emotion\_Recognition.ipynb : Full training and evaluation notebook
* test\_model.py : Script to test a trained model with a new audio file
* app.py : Streamlit web application
* README.md : Project documentation
---

**Requirements**

Install all required packages using:
`pip install -r requirements.txt`

Main libraries used:

* librosa
* numpy
* pandas
* tensorflow
* scikit-learn
* streamlit
* seaborn
* matplotlib
---
**Stream-lit**

https://emotionalclassifier-2jwrzuv4yhrod4oo5itgzj.streamlit.app/

---

**Demo Video**

A 2-minute demo video demonstrates how the web app works by uploading audio and displaying predictions.

https://drive.google.com/file/d/1634I3SuFXnn-dmk_0J8ru4aoFBCQ43dD/view?usp=sharing
---

**Acknowledgements**

* Dataset: RAVDESS (Ryerson University)
* Tools: librosa, TensorFlow, scikit-learn, streamlit

---

**Future Improvements**

* Predict emotion intensity (not just class)
* Combine audio and video for improved emotion recognition
* Enable real-time streaming predictions

