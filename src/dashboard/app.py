from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from scripts._common import generate_gradcam_overlay, load_experiment_bundle, predict_image


def main() -> None:
    st.set_page_config(page_title="Banknote Dashboard", layout="wide")
    st.title("Banknote Analysis Dashboard")
    st.write("Load a trained experiment, classify an image, and generate a Grad-CAM overlay.")

    with st.sidebar:
        st.header("Model")
        experiment_root = st.text_input(
            "Experiment root",
            value=str(Path("experiments")),
        )
        checkpoint_name = st.text_input("Checkpoint", value="best.pt")
        target_layer = st.text_input("Grad-CAM target layer", value="")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)

    uploaded_file = st.file_uploader("Upload a banknote image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    if not experiment_root:
        st.warning("Set an experiment root first.")
        return

    if st.button("Analyze"):
        bundle = load_experiment_bundle(Path(experiment_root), checkpoint_name)
        result, _ = predict_image(bundle.model, image, bundle.config.dataset.image_size)

        if bundle.class_names and result.predicted_class < len(bundle.class_names):
            label = bundle.class_names[result.predicted_class]
        else:
            label = str(result.predicted_class)

        st.subheader("Prediction")
        st.metric("Predicted class", label)
        st.metric("Confidence", f"{result.confidence:.4f}")
        st.json(result.probabilities.tolist())

        if show_gradcam:
            gradcam_result, overlay = generate_gradcam_overlay(
                bundle.model,
                image,
                bundle.config.dataset.image_size,
                target_layer=target_layer or None,
                target_class=None,
            )
            st.subheader("Grad-CAM")
            st.image(overlay, caption=f"Target class: {gradcam_result.target_class}", use_container_width=True)


if __name__ == "__main__":
    main()