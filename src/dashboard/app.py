from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.application.experiment import ExperimentApplication
from src.application.gradcam import GradCAMApplication
from src.application.inference import InferenceApplication


@st.cache_resource
def load_model(
    experiment_root: str,
    checkpoint_name: str,
):
    """
    Load an experiment and its model.

    Cached so the model is only loaded once unless the
    experiment path or checkpoint changes.
    """
    experiment_app = ExperimentApplication()
    inference_app = InferenceApplication()

    context = experiment_app.prepare(
        existing_run=Path(experiment_root),
    )

    model = inference_app.prepare_model(
        context=context,
        checkpoint=(
            context.paths.checkpoints
            / checkpoint_name
        ),
    )

    return context, model


def main() -> None:
    st.set_page_config(
        page_title="Banknote Dashboard",
        layout="wide",
    )

    st.title("Banknote Analysis Dashboard")
    st.write(
        "Load a trained experiment, classify an image, and generate a Grad-CAM overlay."
    )

    with st.sidebar:
        st.header("Model")

        experiment_root = st.text_input(
            "Experiment root",
            value=str(Path("experiments")),
        )

        checkpoint_name = st.text_input(
            "Checkpoint",
            value="best.pt",
        )

        target_layer = st.text_input(
            "Grad-CAM target layer",
            value="",
        )

        show_gradcam = st.checkbox(
            "Show Grad-CAM",
            value=True,
        )

    uploaded_file = st.file_uploader(
        "Upload a banknote image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "webp",
        ],
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Input image",
        use_container_width=True,
    )

    if not experiment_root:
        st.warning("Set an experiment root first.")
        return

    if st.button("Analyze"):
        try:
            context, model = load_model(
                experiment_root,
                checkpoint_name,
            )

            inference_app = InferenceApplication()

            result = inference_app.predict_with_module(
                context=context,
                model=model,
                image=image,
            )

            class_names = getattr(
                context.config.dataset,
                "class_names",
                None,
            )

            if (
                class_names is not None
                and result.predicted_class < len(class_names)
            ):
                label = class_names[result.predicted_class]
            else:
                label = str(result.predicted_class)

            st.subheader("Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted class",
                    label,
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{result.confidence:.4f}",
                )

            st.json(result.probabilities.tolist())

            if show_gradcam:
                gradcam_app = GradCAMApplication()

                gradcam_result, overlay = gradcam_app.generate(
                    context=context,
                    model=model,
                    image=image,
                    target_layer=(
                        target_layer
                        if target_layer
                        else None
                    ),
                )

                st.subheader("Grad-CAM")

                st.image(
                    overlay,
                    caption=(
                        f"Target class: "
                        f"{gradcam_result.target_class}"
                    ),
                    use_container_width=True,
                )

        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()