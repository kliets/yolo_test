import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Fundbüro", layout="wide")

# 1. YOLO MODELL LADEN (Kein TensorFlow nötig -> Fehler gelöst!)
@st.cache_resource
def load_yolo():
    # Lädt das kleine, schnelle Modell (wird bei Bedarf geladen)
    return YOLO('yolov8n.pt')

model = load_yolo()

st.title("🔍 KI Objekterkennung (YOLO)")

# 2. DATEI UPLOAD
uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)
    
    # Layout mit zwei Spalten: Links Original, Rechts KI-Ergebnis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dein Foto")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("KI Analyse")
        with st.spinner('Analysiere...'):
            # YOLO Vorhersage treffen
            results = model(image)
            
            # Das Ergebnis-Bild mit den Boxen zeichnen
            res_plotted = results[0].plot()
            
            # Von BGR zu RGB konvertieren (YOLO nutzt intern OpenCV Format)
            res_image = Image.fromarray(res_plotted[:, :, ::-1])
            
            st.image(res_image, use_column_width=True)

    # 3. TEXT-AUSGABE DER ERGEBNISSE
    st.divider()
    st.subheader("Gefundene Objekte:")
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.write("Keine Objekte erkannt.")
    else:
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            prob = float(box.conf[0])
            st.success(f"Gefunden: **{label}** (Sicherheit: {prob:.2%})")

else:
    st.info("Bitte lade ein Bild hoch, um die Erkennung zu starten.")
