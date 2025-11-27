import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pymongo import MongoClient
import base64
import io

# Conexão MongoDB
MONGO_URI = "mongodb+srv://papyrusbrecho_db_user:jQB2pMbR6kutyqoI@cluster0.20oilzs.mongodb.net/?appName=Cluster0"  # substitua pela sua URI
client = MongoClient(MONGO_URI)
db = client["fei"]
collection = db["faces"]

st.title("Reconhecimento Facial - Base FEI")

# Upload da nova imagem
uploaded_file = st.file_uploader("Escolha uma imagem para comparar", type=["jpg", "png"])

def image_to_array(file):
    image = Image.open(file).convert("RGB")
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Função para comparar duas imagens
def compare_faces(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    diff = cv2.absdiff(img1_gray, img2_gray)
    score = np.sum(diff)
    return score

if uploaded_file:
    uploaded_img = image_to_array(uploaded_file)
    best_score = float("inf")
    best_match = None

    # Percorrer todas as imagens do banco
    for doc in collection.find():
        img_data = base64.b64decode(doc["image"])
        img_array = np.array(Image.open(io.BytesIO(img_data)).convert("RGB"))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        score = compare_faces(uploaded_img, img_array)
        if score < best_score:
            best_score = score
            best_match = doc

    st.write(f"Face mais parecida: {best_match['person']}")
    st.image(uploaded_img, caption="Nova imagem", use_column_width=True)
    st.image(img_array, caption=f"Mais parecida: {best_match['filename']}", use_column_width=True)
