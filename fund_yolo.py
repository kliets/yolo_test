import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Verhindert, dass alte TensorFlow-Reste geladen werden
st.cache_resource.clear()

st.title("🚀 YOLO Fundbüro Test")

# 1. Modell laden (YOLOv8 Nano)
@st.cache_resource
def get_model():
    return YOLO('yolov8n.pt')

try:
    model = get_model()
    st.success("YOLO Modell erfolgreich geladen!")
except Exception as e:
    st.error(f"Modell-Fehler: {e}")

# 2. Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Bild sofort anzeigen (damit wir sehen, dass der Upload klappt)
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    if st.button("KI Erkennung starten"):
        # Vorhersage
        results = model(image)
        
        # Ergebnis-Bild mit Boxen erstellen
        res_plotted = results[0].plot()
        # Konvertierung von BGR zu RGB
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        # Ergebnis anzeigen
        st.subheader("Ergebnis der KI:")
        st.image(res_image, caption="Erkanntes Bild", use_column_width=True)
        
        # Liste der Objekte
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"✅ Gefunden: **{label}** ({conf:.2%})")
