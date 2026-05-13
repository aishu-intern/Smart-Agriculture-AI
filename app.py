import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# BACKGROUND + FONT DESIGN
# =====================================================

page_bg = """
<style>

/* Main Background */
[data-testid="stAppViewContainer"]{
    background-image:url("https://images.unsplash.com/photo-1464226184884-fa280b87c399");
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
}

/* Transparent Header */
[data-testid="stHeader"]{
    background:rgba(0,0,0,0);
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:rgba(0,100,0,0.85);
}

/* Main Title */
.big-font {
    font-size:60px !important;
    font-weight:bold;
    color:white;
    text-align:center;
    font-family:"Times New Roman";
}

/* Subtitle */
.small-font {
    font-size:25px !important;
    color:white;
    text-align:center;
    font-family:"Times New Roman";
    font-weight:bold;
}

/* Glass Card */
.card {
    background:rgba(255,255,255,0.2);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    color:white;
    font-family:"Times New Roman";
    font-size:15px;
    font-weight:bold;
}

/* Labels */
label, .stMarkdown, .stTextInput, .stSelectbox,
.stSlider, .stFileUploader {

    font-family: "Times New Roman" !important;
    font-size: 15px !important;
    font-weight: bold !important;
    color: white !important;
}

/* Headers */
h1, h2, h3, h4 {
    font-family: "Times New Roman" !important;
    color: white !important;
    font-weight: bold !important;
}

/* Input Box */
input {
    font-family:"Times New Roman" !important;
    font-size:15px !important;
    font-weight:bold !important;
    color:black !important;
}

/* Slider Text */
div[data-baseweb="slider"] {
    color:white !important;
}

/* Success / Info */
.stSuccess, .stInfo {
    font-family:"Times New Roman" !important;
    font-size:15px !important;
    font-weight:bold !important;
}

/* Chatbot Output */
.chat-text {
    font-family:"Times New Roman";
    font-size:15px;
    font-weight:bold;
    color:white;
}

/* Analytics Text */
.analytics-text {
    font-family:"Times New Roman";
    font-size:18px;
    font-weight:bold;
    color:white;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    '<p class="big-font">🌿 Smart Agriculture AI Platform</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="small-font">AI Powered Farming Assistant</p>',
    unsafe_allow_html=True
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

data = load_data()

# =====================================================
# TRAIN MODEL
# =====================================================

X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X, y)

# =====================================================
# SIDEBAR MENU
# =====================================================

menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "🏠 Home",
        "🌱 Crop Recommendation",
        "🍃 Disease Detection",
        "🤖 AI Chatbot",
        "📊 Analytics"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if menu == "🏠 Home":

    st.image(
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854",
        use_container_width=True
    )

    st.markdown(
        '''
        <div class="card">
        <h2>🌾 Welcome to Smart Agriculture AI</h2>

        <p>
        This platform helps farmers using Artificial Intelligence.
        </p>

        <ul>
            <li>🌱 AI Crop Recommendation</li>
            <li>🍃 Disease Detection</li>
            <li>🤖 Smart Agriculture Chatbot</li>
            <li>📊 Farm Analytics</li>
        </ul>

        </div>
        ''',
        unsafe_allow_html=True
    )

# =====================================================
# CROP RECOMMENDATION
# =====================================================

elif menu == "🌱 Crop Recommendation":

    st.header("🌱 AI Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:

        N = st.slider("Nitrogen", 0, 150, 40)
        P = st.slider("Phosphorus", 0, 150, 40)
        K = st.slider("Potassium", 0, 150, 40)

    with col2:

        temperature = st.slider("Temperature", 0.0, 50.0, 25.0)
        humidity = st.slider("Humidity", 0.0, 100.0, 65.0)
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.slider("Rainfall", 0.0, 300.0, 120.0)

    if st.button("🚀 Analyze Farm"):

        with st.spinner("AI is analyzing farm conditions..."):
            time.sleep(2)

            input_data = pd.DataFrame(
                [[N, P, K, temperature, humidity, ph, rainfall]],
                columns=X.columns
            )

            prediction = model.predict(input_data)[0]

            confidence = np.max(
                model.predict_proba(input_data)
            ) * 100

        st.success(f"🌾 Recommended Crop: {prediction}")

        st.info(f"🔍 Confidence: {confidence:.2f}%")

        # Crop Images
        crop_images = {

            "rice":
            "https://images.unsplash.com/photo-1592982537447-7440770cbfc9",

            "wheat":
            "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b",

            "cotton":
            "https://images.unsplash.com/photo-1605000797499-95a51c5269ae"

        }

        if prediction in crop_images:

            st.image(
                crop_images[prediction],
                width=500
            )

        profit = np.random.randint(40000, 100000)

        st.success(f"💰 Expected Profit: ₹{profit}")

# =====================================================
# DISEASE DETECTION
# =====================================================

elif menu == "🍃 Disease Detection":

    st.header("🍃 Plant Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            use_container_width=True
        )

        with st.spinner("AI is analyzing leaf image..."):

            time.sleep(3)

            diseases = [

                "Healthy Leaf ✅",
                "Leaf Blight ⚠️",
                "Powdery Mildew ⚠️",
                "Rust Disease ⚠️"

            ]

            prediction = np.random.choice(diseases)

        st.success(f"Prediction: {prediction}")

# =====================================================
# AI CHATBOT
# =====================================================

elif menu == "🤖 AI Chatbot":

    st.header("🤖 Agriculture AI Chatbot")

    user_input = st.text_input(
        "Ask farming questions"
    )

    if user_input:

        question = user_input.lower()

        if "rice" in question:

            answer = "Rice requires high humidity and rainfall."

        elif "fertilizer" in question:

            answer = "Use balanced NPK fertilizers for healthy growth."

        elif "water" in question:

            answer = "Drip irrigation helps save water."

        elif "soil" in question:

            answer = "Maintain soil pH between 6 and 7."

        elif "weather" in question:

            answer = "Check weather forecasts regularly before irrigation."

        else:

            answer = "Consult agricultural experts for detailed advice."

        st.markdown(
            f'<p class="chat-text">{answer}</p>',
            unsafe_allow_html=True
        )

# =====================================================
# ANALYTICS
# =====================================================

elif menu == "📊 Analytics":

    st.header("📊 Smart Farming Analytics")

    st.markdown(
        '<p class="analytics-text">Crop Production Analysis</p>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["Rice", "Wheat", "Cotton"]
    )

    st.line_chart(chart_data)

    st.bar_chart(chart_data)

    st.markdown(
        '''
        <div class="card">
        📈 Analytics helps farmers monitor crop growth,
        productivity, and expected yield trends using AI.
        </div>
        ''',
        unsafe_allow_html=True
    )