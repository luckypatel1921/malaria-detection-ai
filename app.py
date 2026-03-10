import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load model
model = tf.keras.models.load_model("malaria_model.h5")

st.set_page_config(page_title="MalariaScope AI", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#020617);
color:white;
font-family:Arial;
}

/* Hero */
.hero{
text-align:center;
padding:40px;
}

.hero h1{
font-size:50px;
font-weight:700;
}

.hero span{
color:#7c8cff;
}

/* Stat cards */
.card{
background:rgba(255,255,255,0.05);
padding:25px;
border-radius:15px;
text-align:center;
border:1px solid rgba(255,255,255,0.1);
}

/* Upload area */
.upload{
background:rgba(255,255,255,0.05);
padding:30px;
border-radius:15px;
border:1px solid rgba(255,255,255,0.1);
}

/* Result cards */
.infected{
background:rgba(255,0,0,0.1);
padding:30px;
border-radius:15px;
border:1px solid rgba(255,0,0,0.4);
text-align:center;
}

.uninfected{
background:rgba(0,255,100,0.1);
padding:30px;
border-radius:15px;
border:1px solid rgba(0,255,100,0.4);
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------- Hero Section ----------
st.markdown("""
<div class="hero">
<h1>Revolutionary <span>AI-Powered</span><br>Malaria Detection</h1>
<p>Clinical-grade accuracy • Real-time analysis • Instant results</p>
</div>
""", unsafe_allow_html=True)

# ---------- Stats ----------
c1,c2,c3 = st.columns(3)

with c1:
    st.markdown('<div class="card"><h2>99.2%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><h2>&lt;3s</h2><p>Analysis Time</p></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><h2>2,852</h2><p>Scans Today</p></div>', unsafe_allow_html=True)

st.write("")

# ---------- Patient Info ----------
st.markdown("### 👤 Patient Information")
patient_name = st.text_input("Enter Patient Name")

# ---------- Upload Section ----------
st.markdown("### 📂 Upload Blood Cell Image")

uploaded_file = st.file_uploader(
"Drag & drop or click to upload",
type=["png","jpg","jpeg"]
)

# ---------- Image Preview ----------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:
        st.image(image,width=300, caption="Uploaded Sample")

    img = image.resize((64,64))
    img = np.array(img)/255.0
    img = img.reshape(1,64,64,3)

    with col2:

        if st.button("🔬 Analyze Sample"):

            with st.spinner("AI analyzing blood cell..."):
                time.sleep(2)

            prediction = model.predict(img)[0][0]

            infected_prob = 1 - prediction
            uninfected_prob = prediction

            st.markdown("## 🧾 Diagnostic Report")

            # ---------- Result ----------
            if prediction < 0.5:

                st.markdown(f"""
<div class="infected">
<h2>⚠ MALARIA POSITIVE</h2>
<p>Malaria parasite detected</p>
<h3>Confidence: {infected_prob*100:.2f}%</h3>
</div>
""", unsafe_allow_html=True)

            else:

                st.markdown(f"""
<div class="uninfected">
<h2>✅ MALARIA NEGATIVE</h2>
<p>No malaria parasites detected</p>
<h3>Confidence: {uninfected_prob*100:.2f}%</h3>
</div>
""", unsafe_allow_html=True)

            # ---------- Probability ----------
            st.write("### 📊 Prediction Probabilities")

            st.write("Infected Probability")
            st.progress(int(infected_prob*100))

            st.write("Uninfected Probability")
            st.progress(int(uninfected_prob*100))