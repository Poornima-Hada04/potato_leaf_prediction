
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="LeafCare AI",
    page_icon="🌱",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "potato_cnn_model.keras"
CLASS_PATH = ROOT / "classification_report.txt"

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f8f2;
}

header {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.heading {
    color: #24583b;
    font-size: 38px;
    font-weight: 750;
}

.subheading {
    color: #718273;
    font-size: 16px;
}

.upload-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dce8dc;
}

.result-box {
    background: #e7f2e7;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d1e5d2;
}

.stButton button {
    background-color: #34764b;
    color: white;
    border-radius: 9px;
    width: 100%;
    height: 45px;
    border: none;
}

.stButton button:hover {
    background-color: #24583b;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


# Header
st.markdown(
    '<div class="heading">🌱 LeafCare AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subheading">Smart Potato Leaf Disease Detection using Deep Learning</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# Main layout
left, right = st.columns([1, 1], gap="large")

with left:

    st.subheader("📷 Upload Leaf Image")

    st.write("Select a potato leaf image to check its health.")

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Selected Leaf Image",
            use_container_width=True
        )

    else:
        st.info("Please upload an image to continue.")

with right:

    st.subheader("🔍 Analysis Result")

    if uploaded_file:

        st.write("Your image is ready for analysis.")

        if st.button("Analyze Leaf 🌿"):

            try:

                if not MODEL_PATH.exists() or not CLASS_PATH.exists():
                    st.error("Model files not found.")
                    st.stop()

                with st.spinner("Analyzing leaf image..."):

                    model = load_model()
                    classes = load_classes()

                    resized = image.resize((128, 128))

                    arr = np.array(
                        resized,
                        dtype=np.float32
                    )

                    arr = np.expand_dims(arr, axis=0)

                    predictions = model.predict(
                        arr,
                        verbose=0
                    )[0]

                    index = np.argmax(predictions)

                    disease = classes[index].replace(
                        "Potato___", ""
                    ).replace("_", " ")

                    confidence = float(
                        predictions[index] * 100
                    )

                    st.session_state.result = {
                        "disease": disease,
                        "confidence": confidence,
                        "scores": predictions
                    }

            except Exception as e:
                st.error(f"Error: {e}")

    if "result" in st.session_state:

        result = st.session_state.result

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown("### 🌿 Prediction")

        st.subheader(result["disease"].title())

        st.write("Confidence Score")

        st.progress(
            result["confidence"] / 100
        )

        st.markdown(
            f"**{result['confidence']:.2f}%**"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown("### Class Probabilities")

        classes = load_classes()

        for name, score in zip(
            classes,
            result["scores"]
        ):

            st.write(
                name.replace("Potato___", "").replace("_", " ").title()
            )

            st.progress(float(score))

    else:

        st.info(
            "Prediction will appear here after clicking Analyze Leaf."
        )

st.divider()

st.markdown(
    "<center>🌱 LeafCare AI | Powered by CNN & Deep Learning</center>",
    unsafe_allow_html=True
)

