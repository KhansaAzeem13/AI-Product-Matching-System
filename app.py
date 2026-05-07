import streamlit as st
import numpy as np
from PIL import Image
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
model.eval()

# Image transform (same as ResNet50 preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load saved embeddings
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

def extract_features(img):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    features = features.squeeze().numpy().reshape(1, -1)
    return features

def recommend(img, top_n=5):
    features = extract_features(img)
    similarity = cosine_similarity(features, feature_list)[0]
    indices = similarity.argsort()[-top_n:][::-1]
    return [filenames[i] for i in indices]

# UI
st.title("🛍 AI Product Matching System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    results = recommend(img)
    st.subheader("🔍 Similar Products")
    for r in results:
        st.image(r, width=200)
