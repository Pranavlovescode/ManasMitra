import os
import json
from typing import List, Dict

import streamlit as st
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


APP_TITLE = " Intent Classifier"
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))


@st.cache_resource(show_spinner=True)
def load_classifier(model_dir: str = MODEL_DIR):
    """Load tokenizer, model, and create a classification pipeline."""
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Choose dtype based on hardware; BF16/FP16 only when CUDA is available
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
        return_all_scores=True,
        truncation=True,
    )

    # Extract labels (ordered by index) if available
    labels: List[str] = []
    if hasattr(config, "id2label") and isinstance(config.id2label, dict):
        try:
            # keys may be strings; sort by int index
            labels = [config.id2label[str(i)] for i in range(len(config.id2label))]
        except Exception:
            # fallback to dictionary values order (not guaranteed stable)
            labels = list(config.id2label.values())

    meta = {
        "num_labels": len(labels) if labels else getattr(config, "num_labels", None),
        "problem_type": getattr(config, "problem_type", "single_label_classification"),
        "model_type": getattr(config, "model_type", None),
    }

    return clf, labels, meta


def top_k_scores(scores: List[Dict[str, float]], k: int):
    return sorted(scores, key=lambda x: x["score"], reverse=True)[:k]


def format_scores(items: List[Dict[str, float]]):
    return {item["label"]: float(item["score"]) for item in items}


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="centered")
    st.title(APP_TITLE)
    st.caption("Local Hugging Face classifier loaded from this folder.")

    with st.sidebar:
        st.header("Settings")
        default_k = 5
        k = st.slider("Top-K labels to show", 1, 20, default_k)
        show_probs = st.checkbox("Show probabilities table", value=True)
        st.markdown("---")
        st.write("Model directory:")
        st.code(MODEL_DIR)

    with st.spinner("Loading model and tokenizer…"):
        classifier, labels, meta = load_classifier(MODEL_DIR)

    st.subheader("Try it")
    example_texts = [
        "I'm struggling with my emotions today",
        "Can you help me feel calmer?",
        "Thanks for your support",
        "I feel anxious about tomorrow",
        "Let's schedule a session",
    ]

    example = st.selectbox("Examples", options=["(none)"] + example_texts, index=0)
    user_text = st.text_area(
        "Enter a message",
        value="",
        height=120,
        placeholder="Type something like: I feel overwhelmed and need help coping",
    )
    if example != "(none)" and not user_text:
        user_text = example

    if st.button("Classify", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to classify.")
            st.stop()

        with st.spinner("Running inference…"):
            result = classifier(user_text)
            # pipeline returns List[List[{'label','score'}]] for return_all_scores=True
            scores = result[0]
            topk = top_k_scores(scores, k)

        st.markdown("### Prediction")
        if len(topk) > 0:
            best = topk[0]
            st.success(f"Top intent: {best['label']} • {best['score']:.3f}")

        # Chart
        chart_data = {item["label"]: item["score"] for item in topk}
        st.bar_chart(chart_data)

        if show_probs:
            st.markdown("#### Top-K probabilities")
            st.dataframe(
                {
                    "label": [x["label"] for x in topk],
                    "score": [round(float(x["score"]), 6) for x in topk],
                },
                use_container_width=True,
            )

    with st.expander("Model info"):
        st.json(meta)
        if labels:
            st.caption(f"Total labels: {len(labels)}")


if __name__ == "__main__":
    main()
