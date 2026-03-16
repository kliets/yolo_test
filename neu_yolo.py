import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

# ==========================
# 1. SETUP & YOLO-WORLD
# ==========================
st.set_page_config(page_title="KI Fundbüro Profi", layout="centered")

@st.cache_resource
def load_yolo_world():
    # Wir laden das 'Small' Modell für gute Performance in der Cloud
    model = YOLOWorld('yolov8s-worldv2.pt')
    
    # Unsere spezialisierte Fundbüro-Liste (20 Begriffe)
    fundbuero_klassen = [
        "backpack", "umbrella", "bottle", "hat", "dress", 
        "jacket", "shoes", "glasses", "cell phone", "keys",
        "wallet", "book", "pencil case", "lunch box", "scarf",
        "gloves", "cap", "watch", "laptop", "sports bag"
    ]
    
    # Das Modell auf diese Begriffe fixieren
    model.set_classes(fundbuero_klassen)
    return model

model = load_yolo_world()

# ==========================
# 2. BENUTZEROBERFLÄCHE
# ==========================
st.title("🏫 Profi-Schul-Fundbüro (KI)")
st.write("Diese KI erkennt 20 spezifische Fundstücke, inklusive Kleidung und Hüte.")

uploaded_file = st.file_uploader("Foto hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Dein Foto", use_column_width=True)

    if st.button("🔍 KI-Suche starten"):
        with st.spinner('Analysiere Bild auf 20 Kategorien...'):
            try:
                # Vorhersage mit etwas niedrigerem Schwellenwert für bessere Treffer
                results = model.predict(image, conf=0.15)
                
                # Ergebnis-Bild mit Boxen
                res_plotted = results[0].plot(conf=True, labels=True)
                res_image_rgb = Image.fromarray(res_plotted[:, :, ::-1])
                
                st.divider()
                st.subheader("Analyse-Ergebnis")
                st.image(res_image_rgb, use_column_width=True)

                # Liste der erkannten Objekte
                st.write("### 📝 Gefundene Gegenstände:")
                boxes = results[0].boxes
                
                if len(boxes) == 0:
                    st.warning("Nichts aus der Liste erkannt. Vielleicht ist es ein anderer Gegenstand?")
                else:
                    found_labels = []
                    for box in boxes:
                        label = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        
                        # Deutsche Übersetzung für die Anzeige (Mapping)
                        translations = {
                            "backpack": "Rucksack", "umbrella": "Regenschirm", "bottle": "Trinkflasche",
                            "hat": "Hut", "dress": "Kleid", "jacket": "Jacke", "shoes": "Schuhe",
                            "glasses": "Brille", "cell phone": "Handy", "keys": "Schlüssel",
                            "wallet": "Geldbeutel", "book": "Buch", "pencil case": "Mäppchen",
                            "lunch box": "Brotdose", "scarf": "Schal", "gloves": "Handschuhe",
                            "cap": "Mütze/Kappe", "watch": "Armbanduhr", "laptop": "Laptop",
                            "sports bag": "Sporttasche"
                        }
                        
                        de_label = translations.get(label, label)
                        st.success(f"Gefunden: **{de_label}** ({conf:.1%}-Sicherheit)")
                        found_labels.append(de_label)

            except Exception as e:
                st.error(f"Fehler: {e}")
else:
    st.info("Bitte lade ein Bild hoch (z.B. von einem Hut oder Rucksack).")
