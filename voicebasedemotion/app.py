import io
import numpy as np
import streamlit as st
import torch
import librosa
import soundfile as sf
import tempfile
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Optional: audio recorder widget (pip: streamlit-audiorecorder)
try:
    from audiorecorder import audiorecorder  # returns a pydub.AudioSegment
    HAS_RECORDER = True
except Exception:
    HAS_RECORDER = False


MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
MAX_DURATION_SEC = 30.0


@st.cache_resource(show_spinner=False)
def load_model_and_fe():
    """Load model and feature extractor once and cache the result."""
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, feature_extractor, id2label, device


def read_audio_from_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Read arbitrary audio bytes, convert to mono float32 and resample to target_sr."""
    # Read via soundfile (supports wav, flac, ogg, mp3 via libsndfile build)
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), always_2d=False)
    except Exception:
        # Fallback: write to a temp file (preserve format if possible) and let librosa decode
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            data, sr = librosa.load(tmp.name, sr=None, mono=False)
    # Convert stereo -> mono
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=1)
    # Ensure float32
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    data = data.astype(np.float32, copy=False)
    # Resample if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def preprocess_audio_array(audio_array: np.ndarray, feature_extractor, max_duration: float = MAX_DURATION_SEC):
    sr = feature_extractor.sampling_rate
    max_length = int(sr * max_duration)
    # Trim or pad
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        pad = max_length - len(audio_array)
        if pad > 0:
            audio_array = np.pad(audio_array, (0, pad))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=sr,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def predict_emotion_from_bytes(raw_bytes: bytes, model, feature_extractor, id2label, device):
    sr = feature_extractor.sampling_rate
    audio_array = read_audio_from_bytes(raw_bytes, target_sr=sr)
    inputs = preprocess_audio_array(audio_array, feature_extractor)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    # Top-3
    top_indices = np.argsort(probs)[::-1][:3]
    top = [(id2label[int(i)], float(probs[int(i)])) for i in top_indices]
    return pred_label, top


st.set_page_config(page_title="Voice Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice-based Emotion Detection")
st.caption("Upload or record audio and get the predicted emotion using a Whisper-based classifier.")

model, feature_extractor, id2label, device = load_model_and_fe()

tab_upload, tab_record = st.tabs(["Upload audio", "Record audio"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        accept_multiple_files=False,
        help="Max ~30s will be used; longer audio is truncated.",
    )
    if uploaded is not None:
        raw = uploaded.read()
        st.audio(raw, format=f"audio/{uploaded.type.split('/')[-1]}" if uploaded.type else None)
        if st.button("Predict emotion", type="primary"):
            with st.spinner("Running inference…"):
                label, top = predict_emotion_from_bytes(raw, model, feature_extractor, id2label, device)
            st.success(f"Prediction: {label}")
            st.subheader("Top probabilities")
            st.dataframe(
                {"label": [t[0] for t in top], "probability": [round(t[1], 4) for t in top]},
                hide_index=True,
            )

with tab_record:
    if not HAS_RECORDER:
        st.info(
            "Recording widget not available. Install 'streamlit-audiorecorder' to enable in-app recording."
        )
    else:
        st.write("Click to start recording, then click again to stop.")
        audio_segment = audiorecorder("Start recording", "Stop recording")
        # The widget returns a pydub.AudioSegment. When length>0, user recorded something
        if audio_segment and len(audio_segment) > 0:
            # Convert to WAV bytes
            wav_buf = io.BytesIO()
            audio_segment.export(wav_buf, format="wav")
            raw = wav_buf.getvalue()
            st.audio(raw, format="audio/wav")
            if st.button("Predict recorded emotion", type="primary"):
                with st.spinner("Running inference…"):
                    label, top = predict_emotion_from_bytes(raw, model, feature_extractor, id2label, device)
                st.success(f"Prediction: {label}")
                st.subheader("Top probabilities")
                st.dataframe(
                    {"label": [t[0] for t in top], "probability": [round(t[1], 4) for t in top]},
                    hide_index=True,
                )

st.markdown(
    """
    ---
    Notes:
    - Only the first 30 seconds are analyzed; longer clips are truncated.
    - For best results, use clear speech and limit background noise.
    - Model: `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`.
    """
)
