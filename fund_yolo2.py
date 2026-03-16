import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import uuid
from supabase import create_client

# ==========================
# 1. SETUP & VERBINDUNG
# ==========================
st.set_page_config(page_title="KI Fundbüro YOLO", layout="wide")

# Supabase Verbindung (aus deinen Secrets)
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    st.error("Fehler: Supabase Secrets fehlen!")

BUCKET_NAME = "fundbuero"

# ==========================
# 2. YOLO MODELL LADEN
# ==========================
@st.cache_resource
def load_yolo():
    # Lädt das Modell (wird bei Bedarf automatisch heruntergeladen)
    return YOLO('yolov8n.pt')

model = load_yolo()

# ==========================
# 3. UI & UPLOAD
# ==========================
st.title("🏫 Schul-Fundbüro mit YOLO-KI")
st.write("Lade ein Bild hoch. Die KI erkennt Objekte und speichert sie in der Datenbank.")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])
fundort = st.text_input("Fundort (z.B. Schulhof, Mensa)")

if uploaded_file is not None:
    # Bild für die Anzeige und KI öffnen
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dein Foto")
        st.image(image, use_column_width=True)

    if st.button("🚀 Objekt erkennen & speichern"):
        with st.spinner('KI analysiert und speichert...'):
            try:
                # --- SCHRITT 1: YOLO Erkennung ---
                results = model(image)
                
                # Bounding Boxes einzeichnen
                res_plotted = results[0].plot(conf=True, labels=True)
                # BGR zu RGB konvertieren
                res_image_rgb = Image.fromarray(res_plotted[:, :, ::-1])
                
                with col2:
                    st.subheader("KI Ergebnis")
                    st.image(res_image_rgb, use_column_width=True)

                # Alle erkannten Objekte sammeln
                gefundene_dinge = []
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    gefundene_dinge.append(label)

                # Falls nichts erkannt wurde, Standardwert
                kategorie = gefundene_dinge[0] if gefundene_dinge else "Unbekannt"
                
                # --- SCHRITT 2: Bild-Upload zu Supabase ---
                filename = f"{uuid.uuid4()}.jpg"
                uploaded_file.seek(0)
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=filename,
                    file=uploaded_file.read(),
                    file_options={"content-type": "image/jpeg"}
                )
                
                img_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"

                # --- SCHRITT 3: Datenbank-Eintrag ---
                supabase.table("fundbuero").insert({
                    "kategorie": kategorie,
                    "beschreibung": f"KI erkannte: {', '.join(gefundene_dinge)}",
                    "fundort": fundort,
                    "bild_url": img_url,
                    "status": "Offen"
                }).execute()

                st.success(f"Erfolgreich gespeichert! Erkannt als: {kategorie}")
                st.balloons()

            except Exception as e:
                st.error(f"Fehler: {e}")

# ==========================
# 4. GALERIE DER FUNDSTÜCKE
# ==========================
st.divider()
st.header("🔍 Vorhandene Fundstücke")

try:
    response = supabase.table("fundbuero").select("*").order("created_at", desc=True).execute()
    items = response.data

    if items:
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i % 3]:
                st.image(item["bild_url"], use_column_width=True)
                st.markdown(f"**{item['kategorie']}**")
                st.caption(f"📍 {item['fundort']} | {item['created_at'][:10]}")
                st.write(f"_{item['beschreibung']}_")
                st.divider()
    else:
        st.info("Noch keine Fundstücke in der Datenbank.")
except Exception as e:
    st.error("Daten konnten nicht geladen werden.")
