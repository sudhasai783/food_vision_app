# app.py — Beautiful Food Vision Ultra UI
import streamlit as st
from src.model_loader import get_model, load_model_from_path
from src.preprocessing import preprocess_image
from src.predictor import predict_single, predict_topk
from PIL import Image
import io, os, json

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Food Vision Ultra",
    layout="wide",
    page_icon="🍽️"
)

# -----------------------
# Custom CSS for beautiful UI
# -----------------------
st.markdown("""
<style>
.big-pred-box {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 10px;
}
.food-label {
    font-size: 32px;
    font-weight: 700;
}
.confidence-text {
    font-size: 18px;
    color: #555;
}
.top5-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 5px;
}
.dataframe td {
    padding: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Title
# -----------------------
st.title("🍽️ Food Vision Ultra")
st.write("Upload a food image to get the AI prediction. View final prediction or expand to see Top-5.")

# -----------------------
# Config
# -----------------------
DEFAULT_MODEL_PATH = "./models/best_food101_resnet50.pth"
LABELS_PATH = "./src/label_map_food101.json"

# -----------------------
# Load Labels
# -----------------------
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
else:
    labels = {str(i): f"class_{i}" for i in range(101)}

# -----------------------
# Sidebar – Model Loading
# -----------------------
st.sidebar.header("⚙️ Model Settings")
uploaded_ckpt = st.sidebar.file_uploader("Upload .pth model (optional)", type=["pth", "pt", "bin"])
use_local = st.sidebar.checkbox("Use model from repo", value=True)

model = get_model(num_classes=len(labels))
loaded = False

if uploaded_ckpt is not None:
    try:
        model = load_model_from_path(model, uploaded_ckpt, from_bytes=True)
        st.sidebar.success("Model loaded successfully (uploaded).")
        loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

elif use_local and os.path.exists(DEFAULT_MODEL_PATH):
    try:
        model = load_model_from_path(model, DEFAULT_MODEL_PATH, from_bytes=False)
        st.sidebar.success("Model loaded from repo.")
        loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading repo model: {e}")

# -----------------------
# Main Image Upload
# -----------------------
uploaded_img = st.file_uploader("📤 Upload a food image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([1, 1])

if uploaded_img is not None:
    # Show uploaded image
    img = Image.open(io.BytesIO(uploaded_img.read())).convert("RGB")
    col1.image(img, caption="Uploaded Image", use_column_width=True)

    if not loaded:
        col2.warning("Model not loaded. Upload a checkpoint or enable repo model.")
    else:
        # Preprocess + Predict
        inp = preprocess_image(img)

        # Get final prediction
        pred_idx, pred_prob = predict_single(model, inp)
        pred_name = labels.get(str(pred_idx), str(pred_idx))
        pred_clean = pred_name.replace("_", " ").title()

        # -----------------------
        # BEAUTIFUL Prediction Box
        # -----------------------
        col2.markdown(f"""
        <div class="big-pred-box">
            <div class="food-label">🍽️ {pred_clean}</div>
            <div class="confidence-text">Confidence: {pred_prob:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------
        # Top-5 Predictions Expander
        # -----------------------
        with col2.expander("🔍 Show Top-5 Predictions"):
            top5 = predict_topk(model, inp, k=5)

            # Convert to readable table
            st.write("**Top-5 Predictions**")
            for idx, prob in top5:
                name = labels.get(str(idx), str(idx)).replace("_", " ").title()
                st.write(f"- **{name}** — {prob:.2%}")

else:
    st.info("Upload an image to get predictions.")
