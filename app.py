import streamlit as st
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load saved embeddings
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

def extract_features(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
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
