import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import re
import os
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
import zipfile
import io

# Import untuk Fitur AI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import untuk PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --- Konfigurasi Halaman & Konstanta ---
st.set_page_config(page_title="FoodLens: Food Calory Detection", page_icon="🥗", layout="wide")

MODEL_PATH = "models/best.pt"
DB_PATH = "database/foodlens.db"
CLASS_NAMES = [
    'Ayam Goreng -260 kal per 100 gr-', 'Capcay -67 kal per 100gr-', 'Nasi -129 kal per 100gr-', 
    'Sayur bayam -36 kal per 100gr-', 'Sayur kangkung -98 kal per 100gr-', 'Sayur sop -22 kal per 100gr-', 
    'Tahu -80 kal per 100 gr-', 'Telur Dadar -93 kal per 100gr-', 'Telur Mata Sapi -110kal1butir-', 
    'Telur Rebus -78kal 1butir-', 'Tempe -225 kal per 100 gr-', 'Tumis buncis -65 kal per 100gr-', 'food-z7P4'
]
# --- Nilai Gizi Makanan (Dari Berbagai Source di Internet) ---
NUTRITION_MAP = {
    'Ayam Goreng': {'protein': 25, 'carbs': 0, 'fat': 17}, 'Capcay': {'protein': 3, 'carbs': 7, 'fat': 3},
    'Nasi': {'protein': 2.7, 'carbs': 28, 'fat': 0.3}, 'Sayur bayam': {'protein': 3, 'carbs': 4, 'fat': 0.3},
    'Sayur kangkung': {'protein': 2, 'carbs': 5, 'fat': 8}, 'Sayur sop': {'protein': 1, 'carbs': 4, 'fat': 0.5},
    'Tahu': {'protein': 8, 'carbs': 2, 'fat': 5}, 'Telur Dadar': {'protein': 9, 'carbs': 1, 'fat': 10},
    'Telur Mata Sapi': {'protein': 6, 'carbs': 0.5, 'fat': 9}, 'Telur Rebus': {'protein': 6, 'carbs': 0.5, 'fat': 5},
    'Tempe': {'protein': 19, 'carbs': 9, 'fat': 11}, 'Tumis buncis': {'protein': 2, 'carbs': 7, 'fat': 4},
    'food-z7P4': {'protein': 0, 'carbs': 0, 'fat': 0}
}

# --- Fungsi-Fungsi Aplikasi ---

# Fungsi AI
@st.cache_data(show_spinner=False)
def get_ai_suggestion(food_list, total_calories):
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key or api_key == "sk-...": 
            return "Error: Kunci API OpenAI tidak valid.", {}
        chat_model = ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini", temperature=0.7)
        prompt = f"Anda adalah seorang ahli gizi dan chef yang ramah. Anda Berasal dari Indonesia. Berdasarkan makanan: **{', '.join(food_list)}** (~{total_calories} kalori), berikan saran dalam format Markdown:\n\n1. **💡 Rekomendasi Lebih Sehat:** (1-2 saran konkret)\n2. **🍳 Ide Resep Cepat & Sehat:** (Nama, Bahan, 2-3 Langkah)"
        response = chat_model.invoke([HumanMessage(content=prompt)])
        token_usage = response.response_metadata.get('token_usage', {})
        return response.content, token_usage
    except Exception as e:
        return f"Gagal menghubungi AI: {e}.", {}

# Fungsi Database
def init_db():
    conn = None
    try:
        # Pastikan direktori untuk database ada
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, timestamp DATETIME, food_name TEXT, calories INTEGER, confidence REAL, protein REAL, carbs REAL, fat REAL)''')
        c.execute("PRAGMA table_info(history)")
        cols = [info[1] for info in c.fetchall()]
        if 'protein' not in cols: c.execute("ALTER TABLE history ADD COLUMN protein REAL")
        if 'carbs' not in cols: c.execute("ALTER TABLE history ADD COLUMN carbs REAL")
        if 'fat' not in cols: c.execute("ALTER TABLE history ADD COLUMN fat REAL")
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def save_detection(summary_df):
    conn = sqlite3.connect(DB_PATH)
    ts = datetime.now()
    for _, row in summary_df.iterrows():
        conn.execute("INSERT INTO history (timestamp, food_name, calories, confidence, protein, carbs, fat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (ts, row['Makanan'], row['Kalori (est.)'], float(row['Confidence']), row['Protein (g)'], row['Karbohidrat (g)'], row['Lemak (g)']))
    conn.commit()
    conn.close()

def fetch_history(days=0):
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM history WHERE timestamp >= date('now', '-{days} days') ORDER BY timestamp DESC" if days > 0 else "SELECT * FROM history ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty: df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def delete_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()

# --- Model Object Detection dan Fungsi Processing ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception:
        return None

def parse_calories(class_name):
    match = re.search(r'-(\d+)\s*kal', class_name)
    return int(match.group(1)) if match else 0

def analyze_image_and_get_summary(image_pil, model, conf_threshold, iou_threshold):
    results = model(image_pil, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    summary = []
    if len(detections) > 0:
        for det_idx in range(len(detections)):
            class_id = detections.class_id[det_idx]
            confidence = detections.confidence[det_idx]

            full_class_name = CLASS_NAMES[class_id]
            food_name = full_class_name.split(' -')[0]
            calories = parse_calories(full_class_name)
            macros = NUTRITION_MAP.get(food_name, {'protein': 0, 'carbs': 0, 'fat': 0})
            summary.append({
                "Makanan": food_name,
                "Kalori (est.)": calories,
                "Protein (g)": macros['protein'],
                "Karbohidrat (g)": macros['carbs'],
                "Lemak (g)": macros['fat'],
                "Confidence": f"{confidence:.2f}"
            })
    return pd.DataFrame(summary), detections

def draw_annotations_and_calc_totals(image, detections):
    image_with_boxes = np.array(image.convert('RGB'))
    totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}

    if len(detections) == 0:
        return image_with_boxes, totals

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    labels = [
        f"{CLASS_NAMES[class_id].split(' -')[0]}"
        for class_id in detections.class_id
    ]
    annotated_image = box_annotator.annotate(scene=image_with_boxes.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    for class_id in detections.class_id:
        full_class_name = CLASS_NAMES[class_id]
        food_name = full_class_name.split(' -')[0]
        calories = parse_calories(full_class_name)
        macros = NUTRITION_MAP.get(food_name, {'protein': 0, 'carbs': 0, 'fat': 0})
        totals['calories'] += calories
        totals['protein'] += macros['protein']
        totals['carbs'] += macros['carbs']
        totals['fat'] += macros['fat']

    return annotated_image, totals

# Fungsi Laporan PDF
def generate_pdf_report(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Laporan Analisis Makanan - FoodLens</b>", styles['h1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Laporan Dibuat: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Ringkasan
    total_cals = df['Kalori (est.)'].sum()
    total_prot = df['Protein (g)'].sum()
    total_carbs = df['Karbohidrat (g)'].sum()
    total_fat = df['Lemak (g)'].sum()
    
    summary_text = f"<b>Total Gambar:</b> {df['filename'].nunique()} | <b>Total Kalori:</b> {total_cals} kal | <b>Total Protein:</b> {total_prot:.1f}g | <b>Total Karbohidrat:</b> {total_carbs:.1f}g | <b>Total Lemak:</b> {total_fat:.1f}g"
    elements.append(Paragraph(summary_text, styles['h3']))
    elements.append(Spacer(1, 24))

    elements.append(Paragraph("<b>Rincian per Gambar</b>", styles['h2']))
    elements.append(Spacer(1, 12))

    # Style untuk tabel
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#262730')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F2F6')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    for filename, group in df.groupby('filename'):
        elements.append(Paragraph(f"<b>File:</b> {os.path.basename(filename)}", styles['h3']))
        
        img_total_cals = group['Kalori (est.)'].sum()
        img_total_prot = group['Protein (g)'].sum()
        img_total_carbs = group['Karbohidrat (g)'].sum()
        img_total_fat = group['Lemak (g)'].sum()
        img_summary_text = f"Total Nutrisi Gambar: {img_total_cals} kal, {img_total_prot:.1f}g Protein, {img_total_carbs:.1f}g Karbo, {img_total_fat:.1f}g Lemak"
        elements.append(Paragraph(img_summary_text, styles['Normal']))
        elements.append(Spacer(1, 12))

        # Table for this image
        table_df = group.drop('filename', axis=1)
        table_data = [table_df.columns.to_list()] + table_df.values.tolist()
        table = Table(table_data)
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 24))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


init_db()

with st.sidebar:
    # Construct the absolute path for the logo
    script_dir = os.path.dirname(__file__)
    logo_path = os.path.join(script_dir, "assets", "LOGO.png")

    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("## FoodLens")
    st.markdown("---")
    st.markdown("**Smart App untuk analisis nilai gizi makanan Anda. Cukup upload foto makanan kamu, dapatkan rincian kalori & makro, catat history, dan dapat saran sehat dari AI!**")
    st.markdown("---")
    with st.expander("💡 Cara Penggunaan"):
        st.markdown("""
        1.  **Pilih Halaman:** Gunakan navigasi di atas untuk memilih mode analisis.
        2.  **Atur Deteksi:** Sesuaikan 'Confidence Level' dan 'IoU' jika perlu (Semakin tinggi, semakin strict).
        3.  **Upload Foto:** Pilih foto makanan kamu yang ingin dianalisis.
        4.  **Lihat Hasil:** Periksa tabel nutrisi dan total kalori.
        5.  **Simpan & Dapatkan Saran:** Simpan hasil ke history atau minta saran dari AI.
        """)
    page = st.radio("Navigasi Utama", ["Analisis Gambar", "Dashboard History Gizi", "Batch Food Detection"])
    st.markdown("---")
    st.header("⚙️ Pengaturan Deteksi")
    conf_threshold_slider = st.slider("Confidence Level", 0.0, 1.0, 0.5, 0.05)
    iou_threshold_slider = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

model = load_model(MODEL_PATH)

if page == "Analisis Gambar":
    st.title("FoodLens 🔍🥗")
    st.write("Upload satu foto makanan kamu untuk mendapatkan estimasi kalori, makronutrien, dan rekomendasi dari AI.")
    st.divider()

    uploaded_file = st.file_uploader("Pilih satu foto makanan...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and model:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Foto Makanan Kamu")
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, use_container_width=True)
        
        with st.spinner('Menganalisis gambar...'):
            summary_df, detections = analyze_image_and_get_summary(image_pil, model, conf_threshold_slider, iou_threshold_slider)
            image_annotated, totals = draw_annotations_and_calc_totals(image_pil, detections)

        with col2:
            st.subheader("🔍 Hasil Deteksi")
            st.image(image_annotated, use_container_width=True)

        st.divider()
        st.subheader("📝 Ringkasan Nutrisi")

        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("### Total Nutrisi Makanan Ini")
            m_cols = st.columns(4)
            m_cols[0].metric("Total Kalori", f"{totals['calories']} kal")
            m_cols[1].metric("Total Protein", f"{totals['protein']:.1f} g")
            m_cols[2].metric("Total Karbohidrat", f"{totals['carbs']:.1f} g")
            m_cols[3].metric("Total Lemak", f"{totals['fat']:.1f} g")
            
            st.divider()
            col_save, col_ai = st.columns(2)
            with col_save:
                if st.button("💾 Simpan Hasil ke Riwayat"):
                    save_detection(summary_df)
                    st.success("Hasil berhasil disimpan ke riwayat!")
                    st.balloons()
            with col_ai:
                if st.button("🤖 Dapatkan Saran AI & Resep"):
                    with st.spinner("AI sedang menyiapkan saran untuk Anda..."):
                        food_names = summary_df['Makanan'].tolist()
                        ai_response, token_usage = get_ai_suggestion(food_names, totals['calories'])
                        
                        st.expander("Lihat Saran dari AI").markdown(ai_response)

                        if token_usage:
                            with st.expander("AI Usage & Price Estimate"):
                                input_tokens = token_usage.get('input_tokens', 0)
                                output_tokens = token_usage.get('output_tokens', 0)
                                total_tokens = token_usage.get('total_tokens', 0)
                                
                                # Rate: IDR 17,000 per 1 million tokens
                                price_per_million_tokens_idr = 17000
                                estimated_price_idr = (total_tokens / 1_000_000) * price_per_million_tokens_idr
                                
                                st.markdown(f"""
                                - **Total tokens (est):** `{total_tokens}`
                                - **Estimated price (IDR):** `{estimated_price_idr:.4f}`
                                """)
        else:
            st.warning("Tidak ada makanan yang terdeteksi. Coba gambar lain atau sesuaikan pengaturan deteksi.")

    elif uploaded_file:
        st.error("Model tidak dapat dimuat. Pastikan file 'best.pt' ada di folder 'models/'.")
    else:
        st.info("Unggah foto makanan kamu untuk memulai detection dan analisis.")

elif page == "Dashboard History Gizi":
    st.title("📊 Dashboard History Gizi")
    st.write("Lacak asupan nutrisi Kamu dari waktu ke waktu.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        time_filter = st.selectbox("Pilih Jangka Waktu:", ("7 Hari Terakhir", "30 Hari Terakhir", "Semua Riwayat"))
    
    with col2:
        st.write("")
        if st.button("🗑️ Hapus Semua Riwayat"):
            delete_history()
            st.success("Semua riwayat berhasil dihapus!")
            st.rerun()

    st.divider()

    days_map = {"7 Hari Terakhir": 7, "30 Hari Terakhir": 30, "Semua Riwayat": 0}
    history_df = fetch_history(days=days_map[time_filter])

    if history_df.empty:
        st.warning("Belum ada riwayat yang tersimpan. Silakan lakukan analisis dan simpan hasilnya terlebih dahulu.")
    else:
        # Metrik Utama
        total_cals = history_df['calories'].sum()
        avg_daily_cals = history_df.groupby(history_df['timestamp'].dt.date)['calories'].sum().mean()
        total_prot = history_df['protein'].sum()
        total_carbs = history_df['carbs'].sum()
        total_fat = history_df['fat'].sum()

        st.markdown("### Ringkasan Periode Ini")
        d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
        d_col1.metric("Total Kalori", f"{total_cals} kal")
        d_col2.metric("Rata-Rata Harian", f"{avg_daily_cals:.0f} kal")
        d_col3.metric("Total Protein", f"{total_prot:.1f} g")
        d_col4.metric("Total Karbohidrat", f"{total_carbs:.1f} g")
        d_col5.metric("Total Lemak", f"{total_fat:.1f} g")
        st.divider()
        
        # Visualisasi
        st.markdown("### Tren Asupan Kalori Harian")
        daily_calories = history_df.groupby(history_df['timestamp'].dt.date)['calories'].sum().reset_index()
        daily_calories.rename(columns={'timestamp': 'Tanggal', 'calories': 'Total Kalori'}, inplace=True)
        fig_line = px.line(daily_calories, x='Tanggal', y='Total Kalori', title="Asupan Kalori per Hari", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("### Komposisi Makronutrien")
            macro_totals = history_df[['protein', 'carbs', 'fat']].sum().reset_index()
            macro_totals.columns = ['Makronutrien', 'Jumlah (g)']
            fig_pie = px.pie(macro_totals, names='Makronutrien', values='Jumlah (g)', title="Komposisi Makronutrien Total", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
        with v_col2:
            st.markdown("### Frekuensi Makanan")
            food_counts = history_df['food_name'].value_counts().reset_index()
            food_counts.columns = ['Makanan', 'Jumlah']
            fig_bar = px.bar(food_counts, x='Makanan', y='Jumlah', title="Frekuensi Konsumsi Makanan", color='Makanan')
            st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("Lihat Data Riwayat Lengkap"):
            st.dataframe(history_df, use_container_width=True)

elif page == "Batch Food Detection":
    st.title("🗂️ Batch Food Detection")
    st.write("Upload beberapa foto makanan kamu (JPG, PNG) atau satu file ZIP untuk dianalisis sekaligus. Hasil akan digabungkan dalam satu laporan.")
    st.divider()
    
    uploaded_files = st.file_uploader("Pilih file...", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

    if uploaded_files and model:
        if st.button("Mulai Proses Batch", key="batch_process_button"):
            all_results_df = []
            image_files_to_process = []

            with st.spinner("Mempersiapkan file..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/zip":
                        with zipfile.ZipFile(uploaded_file, 'r') as z:
                            for filename in z.namelist():
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    image_data = z.read(filename)
                                    image_files_to_process.append({'name': filename, 'data': io.BytesIO(image_data)})
                    else:
                        image_files_to_process.append({'name': uploaded_file.name, 'data': uploaded_file})

            if image_files_to_process:
                progress_bar = st.progress(0, text="Memulai proses batch...")
                for i, file_info in enumerate(image_files_to_process):
                    try:
                        image_pil = Image.open(file_info['data'])
                        summary_df, _ = analyze_image_and_get_summary(image_pil, model, conf_threshold_slider, iou_threshold_slider)
                        if not summary_df.empty:
                            summary_df['filename'] = file_info['name']
                            all_results_df.append(summary_df)
                    except Exception as e:
                        st.warning(f"Gagal memproses file {file_info['name']}: {e}")
                    progress_bar.progress((i + 1) / len(image_files_to_process), text=f"Memproses: {file_info['name']}")
                
                progress_bar.empty()

                if all_results_df:
                    final_df = pd.concat(all_results_df, ignore_index=True)
                    st.success(f"Selesai! {len(image_files_to_process)} gambar diproses, {len(final_df)} makanan terdeteksi.")
                    st.dataframe(final_df, use_container_width=True)
                    
                    st.divider()
                    st.subheader("Ekspor Laporan")
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            label="📥 Unduh Laporan (CSV)",
                            data=final_df.to_csv(index=False).encode('utf-8'),
                            file_name="laporan_analisis_makanan.csv",
                            mime="text/csv",
                        )
                    with dl_col2:
                        pdf_buffer = generate_pdf_report(final_df)
                        st.download_button(
                            label="📄 Unduh Laporan (PDF)",
                            data=pdf_buffer,
                            file_name="laporan_analisis_makanan.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.warning("Tidak ada makanan yang terdeteksi dari semua gambar yang diunggah.")
            else:
                st.warning("Tidak ada file gambar yang valid untuk diproses.")
    elif uploaded_files:
        st.error("Model tidak dapat dimuat. Pastikan file 'best.pt' ada di folder 'models/'.")
    else:
        st.info("Unggah file gambar atau ZIP untuk memulai pemrosesan batch.")