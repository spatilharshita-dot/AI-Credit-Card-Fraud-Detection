import streamlit as st
import pickle
import numpy as np
import pandas as pd
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .alert-box {
    background-color: #550000;
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    border: 3px solid red;
    animation: blink 1s infinite;
    box-shadow: 0px 0px 20px red;
}

@keyframes blink {
    0% {opacity: 1;}
    50% {opacity: 0.4;}
    100% {opacity: 1;}
}

    .stApp {
        background-color: #0e1117;
        color: white;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: lightgray;
        margin-bottom: 30px;
    }

    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL + SCALER ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- LOAD DATASET ----------------
data = pd.read_csv("creditcard.csv")

fraud_samples = data[data["Class"] == 1].drop("Class", axis=1)
legit_samples = data[data["Class"] == 0].drop("Class", axis=1)
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {f"V{i}": 0.0 for i in range(1, 29)}

# ---------------- LOAD METRICS ----------------
try:
    acc = pickle.load(open("accuracy.pkl", "rb"))
except:
    acc = None
    cm = None

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🏦 AI Fraud Detection Banking System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Credit Card Transaction Monitoring Dashboard</div>',
    unsafe_allow_html=True
)

st.write("---")

# ---------------- PROJECT INFO ----------------
st.subheader(" How This System Works")

st.markdown("""
1. Transaction data is collected  
2. Data is scaled using StandardScaler  
3. Logistic Regression model analyzes patterns  
4. System predicts Fraud / Legit transaction  
5. Dashboard shows approval status + fraud risk score  
""")

st.write("---")

# ---------------- MODEL PERFORMANCE ----------------

if acc is not None:
    st.subheader(" Model Accuracy")

    st.success(f"Model Accuracy: {acc * 100:.2f}%")
st.write("---")
st.write("---")

# ADD STEP 2 RIGHT HERE ↓
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔴 Load Random Fraud Example"):
        sample = fraud_samples.sample(1).iloc[0]
        for i in range(1, 29):
            st.session_state[f"V{i}"] = float(sample[f"V{i}"])

with col_b:
    if st.button("🟢 Load Random Legit Example"):
        sample = legit_samples.sample(1).iloc[0]
        for i in range(1, 29):
            st.session_state[f"V{i}"] = float(sample[f"V{i}"])

with col_c:
    if st.button("⬜ Reset All to Zero"):
        for i in range(1, 29):
            st.session_state[f"V{i}"] = 0.0

st.subheader("⚙ Transaction Pattern Features")  # ← this line already exists
# ---------------- DEMO BUTTONS ----------------
st.subheader("🏦 Demo Transactions")

col1, col2 = st.columns(2)

# LEGIT DEMO
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("Legit Transaction Demo"):

        sample = legit_samples.sample(1).values

        input_data = scaler.transform(sample)

        prediction = model.predict(input_data)

        prob = model.predict_proba(input_data)[0][1]

        st.write(f"⚠ Fraud Risk Score: {prob*100:.2f}%")

        if prediction[0] == 0:
            st.markdown(
    """
    <div style="
        background-color:#062e1f;
        color:white;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:26px;
        font-weight:bold;
        border:2px solid #00ff95;
        box-shadow:0px 0px 15px #00ff95;
    ">
        💳 TRANSACTION APPROVED ✅
    </div>
    """,
    unsafe_allow_html=True
)
        else:
            st.error("⚠ FALSE ALERT")

    st.markdown('</div>', unsafe_allow_html=True)

# FRAUD DEMO
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button(" Fraud Transaction Demo"):

        sample = fraud_samples.sample(1).values

        input_data = scaler.transform(sample)

        prediction = model.predict(input_data)

        prob = model.predict_proba(input_data)[0][1]

        st.write(f"⚠ Fraud Risk Score: {prob*100:.2f}%")

        if prediction[0] == 1:
            st.markdown(
    """
    <div class="alert-box">
         FRAUD DETECTED  <br>
        TRANSACTION DECLINED ❌
    </div>
    """,
    unsafe_allow_html=True
)
        else:
            st.warning("⚠ MODEL MISSED FRAUD")

    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")

# ---------------- MANUAL INPUT SECTION ----------------
st.subheader("💳 Credit Card Transaction Form")

# ---------------- CARD DETAILS ----------------

col1, col2 = st.columns(2)

with col1:

    card_holder = st.text_input("👤 Card Holder Name")

    card_number = st.text_input("💳 Card Number")

    amount = st.number_input(
        " Transaction Amount",
        min_value=0.0
    )

with col2:

    location = st.text_input(" Transaction Location")

    merchant = st.text_input(" Merchant Name")

    transaction_type = st.selectbox(
        " Transaction Type",
        [
            "Online Purchase",
            "ATM Withdrawal",
            "POS Payment"
        ]
    )

# ---------------- CARD VALIDATION ----------------

if card_number:

    if len(card_number) != 16 or not card_number.isdigit():

        st.markdown(
            """
            <div style="
                background-color:#3b0a0a;
                color:brown;
                padding:15px;
                border-radius:10px;
                border:2px solid red;
            ">
                ❌ Invalid Card Number
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        if card_number.startswith("4"):

            st.markdown(
                """
                <div style="
                    background-color:#062e1f;
                    color:red;
                    padding:15px;
                    border-radius:10px;
                    border:2px solid #00ff95;
                ">
                    💳 VISA Card Detected
                </div>
                """,
                unsafe_allow_html=True
            )

        elif card_number.startswith("5"):

            st.markdown(
                """
                <div style="
                    background-color:#062e1f;
                    color:white;
                    padding:15px;
                    border-radius:10px;
                    border:2px solid #00ff95;
                ">
                    💳 MasterCard Detected
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning("⚠ Unknown Card Type")

# ---------------- TRANSACTION FEATURES ----------------

st.write("---")

st.subheader("⚙ Transaction Pattern Features")

inputs = [0.0,amount]
col1, col2, col3 = st.columns(3)

for i in range(1, 29):  # V1 to V28 only
    column = [col1, col2, col3][i % 3]
    with column:
        val = st.slider(
            f"V{i}",
            min_value=-10.0,
            max_value=10.0,
            value=float(st.session_state.get(f"V{i}", 0.0)),
            key=f"V{i}"    # ← THIS is what makes sliders update on button click
        )
        inputs.append(val)
# ---------------- MANUAL PREDICTION ----------------
if st.button("🔍 Predict Manual Transaction"):

    input_data = scaler.transform(np.array(inputs).reshape(1, -1))

    prediction = model.predict(input_data)

    prob = model.predict_proba(input_data)[0][1]

    st.write(f"⚠ Fraud Risk Score: {prob*100:.2f}%")

    # Risk Level
    if prob < 0.30:
        st.success(" LOW RISK TRANSACTION")

    elif prob < 0.70:
        st.warning(" MEDIUM RISK TRANSACTION")

    else:
        st.error(" HIGH RISK TRANSACTION")

    # Final Prediction
    if prediction[0] == 1:

        st.markdown(
            """
            <div class="card">
                <h2> FRAUD DETECTED</h2>
                <h3 style="color:red;">Transaction Status: DECLINED ❌</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="card">
                <h2>💳 LEGITIMATE TRANSACTION</h2>
                <h3 style="color:lightgreen;">Transaction Status: APPROVED ✅</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("---")