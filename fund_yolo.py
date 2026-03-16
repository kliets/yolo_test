import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Objekterkennung")
st.title("🔍 YOLOv8 Objekterkennung")
st.write("Lade ein Bild hoch und die KI erkennt Objekte automatisch mit dem vortrainierten YOLO-Modell.")

# 1. Modell laden (vortrainiert auf 80 Alltags-Kategorien)
# 'yolov8n.pt' wird beim ersten Mal automatisch heruntergeladen
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt') 

model = load_yolo_model()

# 2. Upload
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)
    
    # Anzeige des Originalbildes
    st.image(image, caption='Originalbild', use_column_width=True)
    
    if st.button("Objekte erkennen"):
        with st.spinner('KI arbeitet...'):
            # 3. Vorhersage (Inference)
            results = model(image)
            
            # 4. Ergebnisbild zeichnen (YOLO macht das automatisch mit .plot())
            res_plotted = results[0].plot()
            
            # Konvertiere BGR (OpenCV Format) zu RGB (PIL Format)
            res_image = Image.fromarray(res_plotted[:, :, ::-1])
            
            # Ergebnis anzeigen
            st.image(res_image, caption='Erkanntes Ergebnis', use_column_width=True)
            
            # 5. Liste der erkannten Objekte ausgeben
            st.subheader("Gefundene Objekte:")
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                st.write(f"- **{label}** (Sicherheit: {conf:.2%})")
