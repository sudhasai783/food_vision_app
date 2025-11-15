# 🍽️ Food Vision Ultra — Deep Learning Image Classifier  
A production-ready demo for classifying 101 food categories using PyTorch and Streamlit.

This project follows a clean, professional repo structure and demonstrates best practices for model loading, preprocessing, inference, and interactive UI design.

---

## 🚀 Demo Overview
**Food Vision Ultra** is a Streamlit web application that:
- Accepts user-uploaded food images  
- Preprocesses them using the same transforms as training  
- Runs inference using a fine-tuned Food-101 classifier  
- Returns the **top prediction + confidence score**

Built for demos, interviews, and portfolio projects.

---

## 📁 Project Structure

```
food-vision-ultra/
│
├── app.py                     # Streamlit UI + prediction pipeline
│
├── requirements.txt           # Dependencies (Torch, Streamlit, Pillow, etc.)
│
├── models/                    # Place trained model here
│   └── food101_best.pth       # <not included in repo by default>
│
├── src/                       # Reusable Python modules
│   ├── model_loader.py        # Loads model from .pth
│   ├── transforms.py          # Image preprocessing transforms
│   ├── predict.py             # Inference utilities
│   └── labels.py              # Food-101 class labels
│
├── assets/                    # Demo images, screenshots, UI previews
│   ├── sample_1.jpg
│   └── sample_2.jpg
│
└── README.md                  # Project documentation
```

---

## 🧠 Model Details
- Architecture: **ResNet50 / MobileNetV2** (choose your version)
- Dataset: **Food-101**
- Training: Fine-tuned on 101 categories  
- Outputs: Softmax probabilities + class label  
- Format: PyTorch `.pth` file  

> Note: Model file not bundled. Add your model to the `models/` folder.

---

## ▶️ Run Locally

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/food-vision-ultra.git
   cd food-vision-ultra
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Add your model file  
   ```
   models/food101_best.pth
   ```

4. Start Streamlit app  
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ App Preview  
Upload any food image and get a prediction:

- 🍕 Pizza  
- 🍣 Sushi  
- 🍔 Burger  
- 🥗 Caesar Salad  
- 🍰 Cheesecake  
- …and **96 more categories**

Includes:
- Clean UI  
- Confidence display  
- Mobile-responsive layout  

---

## 🧩 Future Enhancements
- Add Grad-CAM heatmaps  
- Replace backbone with EfficientNet-V2  
- Deploy on Hugging Face Spaces / Render  
- Add batch prediction API (FastAPI)

---

## 🤝 Contributing
Pull requests welcome. For major changes, open an issue first.

---

## 📄 License
Open-source under the MIT License.

---

If you use this repo for your portfolio, feel free to link me—I’m happy to help you polish it.
