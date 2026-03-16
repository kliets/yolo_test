import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ==========================
# 1. SETUP & MODELL
# ==========================
st.set_page_config(page_title="YOLO KI-Erkennung", layout="centered")

@st.cache_resource
def load_yolo():
    # Lädt das Modell (wird bei Bedarf automatisch heruntergeladen)
    return YOLO('yolov8n.pt')

model = load_yolo()

# ==========================
# 2. BENUTZEROBERFLÄCHE
# ==========================
st.title("🔍 KI Objekterkennung")
st.write("Lade ein Bild hoch und die KI sagt dir, was darauf zu sehen ist.")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild für die Anzeige und KI öffnen
    image = Image.open(uploaded_file)
    
    # Sofort das Originalbild anzeigen
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🚀 KI-Analyse starten"):
        with st.spinner('Die KI denkt nach...'):
            try:
                # --- SCHRITT 1: YOLO Erkennung ---
                results = model(image)
                
                # --- SCHRITT 2: Visualisierung (Bounding Boxes) ---
                # .plot() zeichnet Boxen und Labels auf das Bild
                res_plotted = results[0].plot(conf=True, labels=True)
                # BGR (OpenCV) zu RGB (Streamlit) konvertieren
                res_image_rgb = Image.fromarray(res_plotted[:, :, ::-1])
                
                st.divider()
                st.subheader("Ergebnisbild mit Markierungen")
                st.image(res_image_rgb, use_column_width=True)

                # --- SCHRITT 3: Kategorie-Ausgabe ---
                st.divider()
                st.subheader("Gefundene Kategorien:")
                
                boxes = results[0].boxes
                if len(boxes) == 0:
                    st.warning("Die KI konnte leider nichts eindeutig erkennen.")
                else:
                    # Wir gehen alle erkannten Objekte durch
                    for box in boxes:
                        class_id = int(box.cls[0])
                        label = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Anzeige als großer grüner Balken
                        st.success(f"Erkannt: **{label}** ({confidence:.1%}-Sicherheit)")

            except Exception as e:
                st.error(f"Fehler bei der Analyse: {e}")
else:
    st.info("Bitte lade ein Bild hoch, um fortzufahren.")
