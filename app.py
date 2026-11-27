import inspect
import io
import random
import string
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from gtts import gTTS

# -------------------- Configuration --------------------
API_BASE = "https://signbridge-backend-1-y2dy.onrender.com"
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"
_STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters \
    else {"use_container_width": True}
ASSET_DIR = Path(__file__).parent / "assets" / "isl"
LANGUAGES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
LETTERS = list(string.ascii_uppercase)
EXCLUDED_LETTERS = {"J", "Z"}  # not in the training dataset
LETTER_NAMES = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ",
    "G": "जी", "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल",
    "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
    "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू",
    "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}

st.set_page_config(page_title="SignBridge | ISL Communication", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --panel:#0b0b0b; --line:#242424; --line2:#3a3a3a; --muted:#8a8a8a; --accent:#ff5a1f; --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace; }
html, body, [class*="css"], .stApp { font-family:'Inter Tight',sans-serif; }
.stApp { background:#000000; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer { display:none !important; }
.block-container { padding-top:2.2rem; padding-bottom:4rem; max-width:1240px; }
[data-testid="stSidebar"] { background:#000000; border-right:1px solid var(--line); }
.brand { font-family:var(--mono); font-weight:500; letter-spacing:.28em; font-size:1rem; color:#ffffff; margin-top:.4rem; }
.brand b { color:var(--accent); font-weight:500; }
.brand-sub { font-family:var(--mono); font-size:.62rem; letter-spacing:.18em; color:var(--muted); margin-top:.45rem; text-transform:uppercase; }
.hero { border-top:1px solid var(--line2); border-bottom:1px solid var(--line); padding:2.4rem 0 1.8rem; margin-bottom:2rem; }
.hero h1 { font-size:3.6rem; font-weight:600; letter-spacing:-.035em; line-height:1.02; margin:0 0 1.1rem; color:#ffffff; padding:0; }
.hero p { font-size:1.05rem; color:#b5b5b5; max-width:720px; line-height:1.65; margin:0 0 1.6rem; }
.eyebrow { font-family:var(--mono); text-transform:uppercase; letter-spacing:.18em; font-size:.7rem; color:var(--accent); margin-bottom:1.1rem; }
.meta-row { display:flex; gap:3rem; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:1rem; }
.meta-row div { font-size:.95rem; color:#ededed; }
.meta-row span { display:block; font-family:var(--mono); font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:.25rem; }
.metric-card,.feature-card,.result-card { background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:1.2rem 1.3rem; }
.feature-card { transition:border-color .15s; height:100%; }
.feature-card:hover { border-color:var(--accent); }
.idx { font-family:var(--mono); color:var(--accent); font-size:.75rem; letter-spacing:.14em; margin-bottom:1.6rem; }
.metric-value { font-family:var(--mono); font-size:1.6rem; color:var(--accent); font-weight:500; }
.feature-title { font-size:1.05rem; font-weight:600; color:#ffffff; margin-bottom:.3rem; }
.feature-copy { font-size:.9rem; color:#9a9a9a; line-height:1.55; }
.section-title { font-size:2rem; font-weight:600; letter-spacing:-.025em; color:#ffffff; margin:.3rem 0 .6rem; }
.section-title::before { content:""; display:inline-block; width:.6rem; height:.6rem; background:var(--accent); margin-right:.8rem; transform:translateY(-.18rem); }
.small-muted { color:var(--muted); font-size:.85rem; }
.result-letter { font-size:6rem; font-weight:600; letter-spacing:-.04em; color:var(--accent); line-height:1; text-align:center; padding:.4rem 0; }
.target-letter { font-size:5rem; font-weight:600; letter-spacing:-.04em; color:var(--accent); line-height:1; text-align:center; padding:.4rem 0; }
.sign-missing { background:var(--panel); border:1px dashed var(--line2); border-radius:2px; color:var(--muted); font-family:var(--mono); font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; padding:2.2rem .5rem; text-align:center; }
.status { font-family:var(--mono); font-size:.64rem; letter-spacing:.16em; margin-bottom:.9rem; }
.status.on { color:#3ddc97; }
.status.off { color:#5c5c5c; }
.footer-note { font-family:var(--mono); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:#5c5c5c; text-align:center; padding-top:1.1rem; }
[data-testid="stMetric"] { border-top:1px solid var(--line); padding-top:.8rem; }
[data-testid="stMetricLabel"] p { font-family:var(--mono); font-size:.66rem !important; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
[data-testid="stMetricValue"] { font-weight:600; letter-spacing:-.02em; }
h3, h4 { font-weight:600; letter-spacing:-.015em; }
.stButton>button, .stDownloadButton>button { border-radius:2px; font-family:var(--mono); font-weight:500; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; padding:.6rem 1.1rem; background:transparent; color:#ffffff; border:1px solid var(--line2); }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#ffffff; background:#0f0f0f; color:#ffffff; }
.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] { background:var(--accent); border:1px solid var(--accent); color:#000000; }
.stButton>button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color:#000000 !important; font-weight:600; }
.stButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background:#ffffff; border-color:#ffffff; }
div[data-testid="stFileUploader"] section { background:var(--panel); border:1px dashed var(--line2); border-radius:2px; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"] { border-radius:2px !important; }
[data-testid="stAlert"] { border-radius:2px; border:1px solid var(--line); }
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { border-radius:2px; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def make_speech_audio(text: str, language_code: str) -> bytes:
    audio = io.BytesIO()
    gTTS(text=text, lang=language_code).write_to_fp(audio)
    return audio.getvalue()


def spoken_letter(letter: str, language_code: str) -> str:
    if language_code == "en":
        return letter
    name = LETTER_NAMES.get(letter, letter)
    return f"अक्षर {name} है" if language_code == "hi" else f"अक्षर {name} आहे"


def predict_image(image_file):
    image_bytes = image_file.getvalue() if hasattr(image_file, "getvalue") else image_file
    files = {"file": (getattr(image_file, "name", "sign.jpg"), image_bytes, "application/octet-stream")}
    started = time.perf_counter()
    try:
        response = requests.post(PREDICT_URL, files=files, timeout=90)
        elapsed = time.perf_counter() - started
        if response.status_code == 200:
            return response.json(), None, elapsed
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return None, f"API returned {response.status_code}: {detail}", elapsed
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond. Render's free service may need time to wake up.", time.perf_counter() - started
    except requests.exceptions.RequestException as exc:
        return None, f"Could not connect to the backend: {exc}", time.perf_counter() - started


def backend_status():
    try:
        response = requests.get(HEALTH_URL, timeout=12)
        return (True, response.json()) if response.ok else (False, None)
    except requests.RequestException:
        return False, None


def add_prediction_to_history(letter, confidence, mode, latency=None):
    st.session_state.prediction_history.append({
        "letter": letter, "confidence": confidence, "mode": mode,
        "latency": latency, "time": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.prediction_history = st.session_state.prediction_history[-150:]


def confidence_feedback(confidence):
    if confidence >= 0.80:
        st.success("High model confidence. Still verify that the displayed letter matches your sign.")
    elif confidence >= 0.50:
        st.warning("Moderate model confidence. Review the result and consider taking another image.")
    else:
        st.error("Low model confidence. Retake the image with better lighting and your full hand in frame.")


def render_prediction(saved, language, audio_key):
    if not saved:
        return
    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="result-letter">{saved["letter"]}</div>', unsafe_allow_html=True)
        st.caption("PREDICTED ISL LETTER")
    with right:
        st.markdown("#### Recognition result")
        st.metric("Model confidence", f'{saved["confidence"]:.1%}')
        st.progress(max(0.0, min(1.0, saved["confidence"])))
        st.caption(saved.get("message", "Prediction returned by the backend."))
        confidence_feedback(saved["confidence"])
        if saved.get("latency") is not None:
            st.caption(f'Backend response time: {saved["latency"]:.2f} seconds')
        if st.button("Speak this letter", key=audio_key):
            try:
                st.session_state.recognition_audio = make_speech_audio(spoken_letter(saved["letter"], LANGUAGES[language]), LANGUAGES[language])
            except Exception as exc:
                st.error(f"Speech generation failed. Check your internet connection. ({exc})")
    if st.session_state.get("recognition_audio"):
        st.audio(st.session_state.recognition_audio, format="audio/mp3")


def safe_metric_div(num, den):
    return num / den if den else 0.0


def evaluate_predictions(actual, predicted):
    labels = sorted(set(actual) | set(predicted))
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
    total = len(actual)
    correct = sum(matrix[label][label] for label in labels)
    per_class = []
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = safe_metric_div(tp, tp + fp)
        recall = safe_metric_div(tp, tp + fn)
        f1 = safe_metric_div(2 * precision * recall, precision + recall)
        support = sum(matrix[label].values())
        per_class.append({"Letter": label, "Precision": precision, "Recall": recall, "F1-score": f1, "Support": support})
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    return {
        "accuracy": safe_metric_div(correct, total),
        "precision_macro": sum(precisions) / len(precisions) if precisions else 0,
        "recall_macro": sum(recalls) / len(recalls) if recalls else 0,
        "f1_macro": sum(f1s) / len(f1s) if f1s else 0,
        "labels": labels, "matrix": matrix, "per_class": per_class, "total": total,
    }


def sign_image(letter: str):
    """Path to the verified ISL reference image for a letter, or None."""
    for ext in ("png", "jpg", "jpeg", "webp"):
        path = ASSET_DIR / f"{letter}.{ext}"
        if path.exists():
            return str(path)
    return None


def sign_placeholder():
    st.markdown('<div class="sign-missing">Reference image<br>not added</div>', unsafe_allow_html=True)


def heat_cell(value, vmax):
    """Black-to-orange cell shading for the confusion matrix (no matplotlib needed)."""
    if not vmax or value <= 0:
        return "color:#5a5a5a"
    alpha = 0.18 + 0.82 * (value / vmax)
    text = "#000000" if alpha > 0.6 else "#ffffff"
    return f"background-color: rgba(255,90,31,{alpha:.2f}); color:{text}"


# -------------------- Session state --------------------
defaults = {
    "spelled_text": "", "prediction_history": [], "challenge_score": 0,
    "challenge_attempts": 0, "challenge_streak": 0, "challenge_best_streak": 0,
    "challenge_target": None, "challenge_feedback": None, "challenge_round_size": 10,
    "challenge_done": False, "last_recognition": None, "recognition_audio": None,
    "message_audio": None, "practice_target": None, "practice_score": 0,
    "practice_attempts": 0, "practice_streak": 0, "practice_best_streak": 0,
    "practice_feedback": None, "practice_hint_level": 0,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown('<div class="brand">SIGN<b>BRIDGE</b></div><div class="brand-sub">ISL communication platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Home", "Recognition Mode", "Guided Practice", "ISL Challenge",
        "Live Spelling", "ISL Alphabet", "Model Insights", "My Progress", "How It Works"
    ])
    st.markdown("---")
    language = st.selectbox("Speech language", list(LANGUAGES.keys()), index=0)
    st.markdown("---")
    st.caption("A two-way communication and learning prototype.")
    st.caption("Images are sent to the configured API for inference; this frontend does not save captured images to disk.")

# -------------------- Home --------------------
if page == "Home":
    st.markdown('''<div class="hero"><div class="eyebrow">Indian Sign Language · Alphabet recognition · Prototype</div><h1>Indian Sign Language,<br>recognised and spoken.</h1><p>SignBridge recognises Indian Sign Language alphabet signs from a single image and supports guided practice, model evaluation and speech-assisted spelling.</p><div class="meta-row"><div><span>Input</span>Still image</div><div><span>Output</span>Letter, text, speech</div><div><span>Speech</span>English / Hindi / Marathi</div></div></div>''', unsafe_allow_html=True)
    ok, _ = backend_status()
    history = st.session_state.prediction_history
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Platform", "SignBridge")
    c2.metric("Predictions this session", len(history))
    c3.metric("Average confidence", f'{sum(x["confidence"] for x in history)/len(history):.1%}' if history else "—")
    c4.metric("Backend", "Connected" if ok else "Unreachable / waking")
    st.markdown('<div class="section-title">Explore the platform</div>', unsafe_allow_html=True)
    cards = st.columns(4)
    features = [
        ("Recognition", "Camera or image upload with confidence-aware feedback."),
        ("Guided Practice", "Practise target letters with hints, retries and scoring."),
        ("Model Insights", "Evaluate a labelled test set using classification metrics."),
        ("Word Builder", "Confirm recognized letters and generate speech."),
    ]
    for idx, (col, (title, copy)) in enumerate(zip(cards, features), 1):
        with col:
            st.markdown(f'<div class="feature-card"><div class="idx">0{idx}</div><div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>', unsafe_allow_html=True)
    st.info("Current backend scope: one static ISL alphabet sign per image. Continuous video, sentence-level sign recognition and translation are not exposed by the current API.")

# -------------------- Recognition --------------------
elif page == "Recognition Mode":
    st.markdown('<div class="section-title">ISL Recognition</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Show one alphabet sign clearly. The image is sent to the existing FastAPI model.</p>', unsafe_allow_html=True)
    source = st.radio("Choose image source", ["Use webcam", "Upload image"], horizontal=True)
    captured = st.camera_input("Capture an ISL hand sign", key="recognition_camera") if source == "Use webcam" else st.file_uploader("Upload a hand-sign image", type=["png", "jpg", "jpeg"], key="recognition_upload")
    if captured:
        a, b = st.columns([1, 1.25])
        with a:
            st.image(captured, caption="Image submitted for recognition", **_STRETCH)
        with b:
            st.markdown("#### Before you submit")
            st.write("• Keep the complete hand visible\n• Use even lighting\n• Avoid cluttered backgrounds\n• Show one sign at a time")
            if st.button("Recognize sign", type="primary", **_STRETCH):
                with st.spinner("Sending image to the recognition service..."):
                    result, error, latency = predict_image(captured)
                if error:
                    st.session_state.last_recognition = None
                    st.error(error)
                elif result and result.get("success"):
                    letter = str(result.get("letter", "")).upper()
                    confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
                    st.session_state.last_recognition = {"letter": letter, "confidence": confidence, "message": result.get("message", "Prediction successful."), "latency": latency}
                    st.session_state.recognition_audio = None
                    add_prediction_to_history(letter, confidence, "Recognition", latency)
                else:
                    st.session_state.last_recognition = None
                    st.warning((result or {}).get("message", "No hand detected. Try again."))
    render_prediction(st.session_state.last_recognition, language, "recognition_speak")
    st.caption("Confidence is the model's reported score, not a guarantee that a prediction is correct. The frontend uses it to guide retries, not to change the model output.")

# -------------------- Guided Practice --------------------
elif page == "Guided Practice":
    st.markdown('<div class="section-title">Guided Practice Studio</div>', unsafe_allow_html=True)
    st.write("Practise one target letter at a time. Unlike the timed challenge, you can retry until you are ready to move on.")
    setup1, setup2, setup3 = st.columns([1, 1, 1])
    difficulty = setup1.selectbox("Practice set", ["Starter · A–H", "Intermediate · A–R (no J)", "Full set · A–Z (no J, Z)"])
    hint = setup2.selectbox("Guidance", ["Basic hints", "Detailed capture checklist"])
    setup3.metric("Best streak", st.session_state.practice_best_streak)
    upper = {"Starter · A–H": 8, "Intermediate · A–R (no J)": 18, "Full set · A–Z (no J, Z)": 26}[difficulty]
    pool = [l for l in LETTERS[:upper] if l not in EXCLUDED_LETTERS]
    if st.session_state.practice_target not in pool:
        st.session_state.practice_target = random.choice(pool)
        st.session_state.practice_feedback = None
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown(f'<div class="result-card"><div class="small-muted" style="text-align:center;font-weight:700">YOUR TARGET LETTER</div><div class="target-letter">{st.session_state.practice_target}</div></div>', unsafe_allow_html=True)
        if st.button("New target letter", **_STRETCH):
            st.session_state.practice_target = random.choice(pool)
            st.session_state.practice_feedback = None
            st.session_state.practice_hint_level = 0
            st.rerun()
        if st.checkbox("Show reference sign", key="practice_show_ref"):
            ref = sign_image(st.session_state.practice_target)
            if ref:
                st.image(ref, caption=f"Reference: {st.session_state.practice_target}", width=240)
            else:
                sign_placeholder()
    with right:
        st.markdown("#### Practice tips")
        st.write("1. Position your hand fully inside the camera frame.\n2. Keep your fingers and thumb visible.\n3. Use a plain background and good lighting.\n4. Hold a static sign steady while capturing.")
        if hint == "Detailed capture checklist":
            st.info("Check that the hand is not cropped, motion blur is minimal, the camera is focused, and the hand is not hidden by jewellery, sleeves or another object. These are image-quality suggestions, not letter-specific sign instructions.")
    photo = st.camera_input("Capture your target sign", key=f"practice_camera_{st.session_state.practice_target}")
    if photo is not None and st.button("Check my sign", type="primary"):
        with st.spinner("Evaluating your attempt..."):
            result, error, latency = predict_image(photo)
        if error:
            st.error(error)
        elif not result or not result.get("success"):
            st.warning((result or {}).get("message", "No hand detected. Please retry."))
        else:
            guess = str(result.get("letter", "")).upper()
            conf = max(0.0, min(1.0, float(result.get("confidence", 0))))
            target = st.session_state.practice_target
            st.session_state.practice_attempts += 1
            add_prediction_to_history(guess, conf, "Guided Practice", latency)
            if guess == target:
                st.session_state.practice_score += 1
                st.session_state.practice_streak += 1
                st.session_state.practice_best_streak = max(st.session_state.practice_best_streak, st.session_state.practice_streak)
                st.session_state.practice_feedback = (True, guess, conf)
            else:
                st.session_state.practice_streak = 0
                st.session_state.practice_feedback = (False, guess, conf)
            st.rerun()
    if st.session_state.practice_feedback:
        correct, guess, conf = st.session_state.practice_feedback
        if correct:
            st.success(f"Correct! The model predicted {guess} for your target.")
        else:
            st.warning(f"The model predicted {guess}; your target was {st.session_state.practice_target}. You can retry or choose a new target.")
        st.caption(f"Reported model confidence: {conf:.1%}")
        if not correct:
            if st.button("Retry same letter"):
                st.session_state.practice_feedback = None
                st.rerun()
        if st.button("Next letter", type="primary"):
            st.session_state.practice_target = random.choice(pool)
            st.session_state.practice_feedback = None
            st.rerun()
    q1, q2, q3 = st.columns(3)
    q1.metric("Correct practice attempts", st.session_state.practice_score)
    q2.metric("Total attempts", st.session_state.practice_attempts)
    q3.metric("Current streak", st.session_state.practice_streak)

# -------------------- ISL Challenge --------------------
elif page == "ISL Challenge":
    st.markdown('<div class="section-title">ISL Challenge Arena</div>', unsafe_allow_html=True)
    st.write("A scored alphabet challenge using one still-image prediction per round.")
    cfg, info = st.columns([1, 2])
    with cfg:
        difficulty = st.selectbox("Difficulty", ["Easy · A–H", "Medium · A–R (no J)", "Hard · A–Z (no J, Z)"])
        round_size = st.selectbox("Questions per round", [5, 10, 15], index=1)
        if st.button("Start / restart challenge", **_STRETCH):
            st.session_state.challenge_score = st.session_state.challenge_attempts = st.session_state.challenge_streak = st.session_state.challenge_best_streak = 0
            st.session_state.challenge_round_size = round_size
            st.session_state.challenge_target = None
            st.session_state.challenge_feedback = None
            st.session_state.challenge_done = False
            st.rerun()
    with info:
        st.markdown('<div class="feature-card"><div class="feature-title">How to play</div><div class="feature-copy">1. Start a round and note the target letter.<br>2. Make that ISL alphabet sign in front of your camera.<br>3. Capture one image and submit it.<br>4. Earn a point when the model prediction matches the target.</div></div>', unsafe_allow_html=True)
    max_letters = {"Easy · A–H": 8, "Medium · A–R (no J)": 18, "Hard · A–Z (no J, Z)": 26}[difficulty]
    pool = [l for l in LETTERS[:max_letters] if l not in EXCLUDED_LETTERS]
    if st.session_state.challenge_target not in pool and not st.session_state.challenge_done:
        st.session_state.challenge_target = random.choice(pool)
    total = st.session_state.challenge_round_size
    attempts = st.session_state.challenge_attempts
    score = st.session_state.challenge_score
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score", f"{score}/{total}"); m2.metric("Attempted", f"{attempts}/{total}")
    m3.metric("Accuracy", f"{score/attempts:.0%}" if attempts else "—"); m4.metric("Current streak", st.session_state.challenge_streak)
    st.progress(min(attempts / total, 1.0) if total else 0)
    if attempts >= total or st.session_state.challenge_done:
        st.session_state.challenge_done = True
        st.success(f"Challenge complete: {score}/{total}. Best streak: {st.session_state.challenge_best_streak}.")
        if st.button("Play another round", type="primary"):
            st.session_state.challenge_score = st.session_state.challenge_attempts = st.session_state.challenge_streak = st.session_state.challenge_best_streak = 0
            st.session_state.challenge_target = random.choice(pool)
            st.session_state.challenge_feedback = None
            st.session_state.challenge_done = False
            st.rerun()
    else:
        st.markdown(f'<div class="result-card"><div class="small-muted" style="text-align:center;font-weight:700">SHOW THIS SIGN</div><div class="target-letter">{st.session_state.challenge_target}</div></div>', unsafe_allow_html=True)
        photo = st.camera_input("Capture your target sign", key=f"challenge_camera_{attempts}")
        if photo is not None and st.button("Check my sign", type="primary", **_STRETCH):
            with st.spinner("Checking your sign..."):
                result, error, latency = predict_image(photo)
            if error:
                st.error(error)
            elif not result or not result.get("success"):
                st.warning((result or {}).get("message", "No hand detected. Try again."))
            else:
                guess = str(result.get("letter", "")).upper()
                conf = max(0.0, min(1.0, float(result.get("confidence", 0))))
                target = st.session_state.challenge_target
                add_prediction_to_history(guess, conf, "ISL Challenge", latency)
                st.session_state.challenge_attempts += 1
                if guess == target:
                    st.session_state.challenge_score += 1
                    st.session_state.challenge_streak += 1
                    st.session_state.challenge_best_streak = max(st.session_state.challenge_best_streak, st.session_state.challenge_streak)
                    st.session_state.challenge_feedback = (True, guess, conf, target)
                else:
                    st.session_state.challenge_streak = 0
                    st.session_state.challenge_feedback = (False, guess, conf, target)
                st.rerun()
        if st.session_state.challenge_feedback:
            correct, guess, conf, target = st.session_state.challenge_feedback
            st.success(f"Correct! {guess} matched the target.") if correct else st.error(f"Model predicted {guess}; target was {target}.")
            st.caption(f"Model confidence: {conf:.1%}")
            if st.session_state.challenge_attempts < total and st.button("Next challenge", type="primary"):
                st.session_state.challenge_target = random.choice(pool)
                st.session_state.challenge_feedback = None
                st.rerun()
    st.caption("Challenge scoring compares the model's predicted letter with the target. It is an educational alphabet game, not a sign-language fluency assessment.")

# -------------------- Live Spelling --------------------
elif page == "Live Spelling":
    st.markdown('<div class="section-title">Live Spelling & Word Builder</div>', unsafe_allow_html=True)
    st.write("Recognize one letter at a time, confirm it, and build a message before speaking it aloud.")
    photo = st.camera_input("Capture one ISL letter", key="live_spelling_camera")
    if photo is not None and st.button("Predict captured letter", type="primary"):
        with st.spinner("Recognizing letter..."):
            result, error, latency = predict_image(photo)
        if error:
            st.error(error)
        elif result and result.get("success"):
            letter = str(result.get("letter", "")).upper()
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
            st.session_state.live_prediction = (letter, confidence)
            add_prediction_to_history(letter, confidence, "Live Spelling", latency)
            st.session_state.message_audio = None
            st.rerun()
        else:
            st.warning((result or {}).get("message", "No hand detected."))
    if "live_prediction" in st.session_state:
        letter, confidence = st.session_state.live_prediction
        st.success(f"Detected: {letter} · Confidence: {confidence:.1%}")
        confidence_feedback(confidence)
        col1, col2 = st.columns(2)
        if col1.button("Confirm and add letter", **_STRETCH):
            st.session_state.spelled_text += letter
            del st.session_state.live_prediction
            st.rerun()
        if col2.button("Discard prediction", **_STRETCH):
            del st.session_state.live_prediction
            st.rerun()
    st.markdown("---")
    st.markdown("#### Your message")
    st.markdown(f'<div class="result-card"><div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:1.8rem;font-weight:500;color:#ededed;letter-spacing:.08em;min-height:42px">{st.session_state.spelled_text or "Your confirmed letters will appear here..."}</div></div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    if b1.button("Add space", **_STRETCH):
        st.session_state.spelled_text += " "; st.session_state.message_audio = None; st.rerun()
    if b2.button("Backspace", **_STRETCH):
        st.session_state.spelled_text = st.session_state.spelled_text[:-1]; st.session_state.message_audio = None; st.rerun()
    if b3.button("Clear message", **_STRETCH):
        st.session_state.spelled_text = ""; st.session_state.message_audio = None; st.session_state.pop("live_prediction", None); st.rerun()
    if st.button("Speak message", type="primary", disabled=not st.session_state.spelled_text.strip()):
        try:
            st.session_state.message_audio = make_speech_audio(st.session_state.spelled_text, LANGUAGES[language])
        except Exception as exc:
            st.error(f"Speech generation failed. Check your internet connection. ({exc})")
    if st.session_state.message_audio:
        st.audio(st.session_state.message_audio, format="audio/mp3")

# -------------------- ISL Alphabet Reference --------------------
elif page == "ISL Alphabet":
    st.markdown('<div class="section-title">ISL Alphabet Reference</div>', unsafe_allow_html=True)
    st.write("Reference handshapes for the Indian Sign Language alphabet. Select a letter to view it, or browse the full set below.")
    loaded = [letter for letter in LETTERS if sign_image(letter)]
    if not loaded:
        st.info("No reference images have been added yet. Place one image per letter (A.png to Z.png) in assets/isl/, "
                "or run prepare_sign_images.py on your ISL dataset to generate them. See README.md for details.")
    else:
        st.caption(f"{len(loaded)} of 26 reference images loaded.")
    selected = st.selectbox("Choose a letter", LETTERS)
    detail_img, detail_info = st.columns([1, 1.6])
    with detail_img:
        path = sign_image(selected)
        if path:
            st.image(path, caption=f"ISL letter {selected}", width=320)
        else:
            sign_placeholder()
    with detail_info:
        st.markdown(f'<div class="result-card"><div class="small-muted" style="font-weight:600">LETTER</div><div class="feature-title" style="font-size:1.6rem">{selected}  ·  {LETTER_NAMES[selected]}</div><div class="feature-copy">Use Guided Practice to test this sign with your camera.</div></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Practise this letter", type="primary"):
            st.session_state.practice_target = selected
            st.session_state.practice_feedback = None
            st.session_state.practice_hint_level = 0
            st.info("Open Guided Practice from the sidebar to start your camera attempt.")
    st.markdown("#### Browse all letters")
    for start in range(0, 26, 7):
        cols = st.columns(7)
        for col, letter in zip(cols, LETTERS[start:start + 7]):
            with col:
                tile = sign_image(letter)
                if tile:
                    st.image(tile, caption=letter, **_STRETCH)
                else:
                    sign_placeholder()
                    st.caption(letter)

# -------------------- Model Insights --------------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model Evaluation and Transparency</div>', unsafe_allow_html=True)
    st.write("Upload labelled test results to calculate genuine classification metrics for the CNN model.")
    st.info("Expected CSV columns: actual and predicted (or y_true and y_pred). Each row must contain the ground-truth letter and the model's prediction for the same test image. Do not upload training predictions as test performance.")
    uploaded = st.file_uploader("Upload model evaluation CSV", type=["csv"], key="evaluation_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            lower_map = {c.lower().strip(): c for c in df.columns}
            actual_col = next((lower_map[x] for x in ["actual", "y_true", "true", "ground_truth", "label"] if x in lower_map), None)
            predicted_col = next((lower_map[x] for x in ["predicted", "y_pred", "prediction", "pred"] if x in lower_map), None)
            if not actual_col or not predicted_col:
                st.error("Could not find the required columns. Please use 'actual' and 'predicted' or 'y_true' and 'y_pred'.")
            else:
                data = df[[actual_col, predicted_col]].dropna().copy()
                actual = data[actual_col].astype(str).str.strip().str.upper().tolist()
                predicted = data[predicted_col].astype(str).str.strip().str.upper().tolist()
                valid = [(a, p) for a, p in zip(actual, predicted) if len(a) == 1 and len(p) == 1 and a in LETTERS and p in LETTERS]
                if not valid:
                    st.error("No valid A–Z letter pairs found in the uploaded file.")
                else:
                    actual, predicted = zip(*valid)
                    metrics = evaluate_predictions(list(actual), list(predicted))
                    st.caption(f"Evaluation based on {metrics['total']} valid labelled rows from the uploaded file.")
                    a, b, c, d = st.columns(4)
                    a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                    b.metric("Macro precision", f"{metrics['precision_macro']:.1%}")
                    c.metric("Macro recall", f"{metrics['recall_macro']:.1%}")
                    d.metric("Macro F1-score", f"{metrics['f1_macro']:.1%}")
                    labels = metrics["labels"]
                    matrix_df = pd.DataFrame(metrics["matrix"], index=labels, columns=labels)
                    matrix_df.index.name = "Actual (rows) / Predicted (columns)"
                    st.markdown("#### Confusion matrix")
                    st.caption("Rows are actual labels; columns are predicted labels. Diagonal cells are correct predictions.")
                    _vmax = matrix_df.values.max()
                    _styler = matrix_df.style
                    _shade = lambda v: heat_cell(v, _vmax)
                    _styler = _styler.map(_shade) if hasattr(_styler, "map") else _styler.applymap(_shade)
                    st.dataframe(_styler, **_STRETCH)
                    st.markdown("#### Per-letter performance")
                    class_df = pd.DataFrame(metrics["per_class"])
                    for col in ["Precision", "Recall", "F1-score"]:
                        class_df[col] = class_df[col].map(lambda x: f"{x:.1%}")
                    st.dataframe(class_df, **_STRETCH, hide_index=True)
                    st.download_button("Download per-letter report", class_df.to_csv(index=False).encode("utf-8"), "signbridge_per_letter_metrics.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.markdown("#### What will be calculated?")
        cols = st.columns(4)
        cols[0].metric("Accuracy", "—")
        cols[1].metric("Precision", "—")
        cols[2].metric("Recall", "—")
        cols[3].metric("F1-score", "—")
        st.markdown("#### Example CSV format")
        st.dataframe(pd.DataFrame({"actual": ["A", "B", "C", "A"], "predicted": ["A", "B", "A", "A"]}), **_STRETCH, hide_index=True)
        st.caption("The rows above only illustrate the required file structure. They are not SignBridge model results.")
    st.markdown("#### Evaluation notes")
    st.write("• Use a held-out test set that was not used to train the CNN.\n• Include all relevant alphabet classes for a fair evaluation.\n• Macro averages give each represented letter equal weight.\n• The confusion matrix helps identify pairs of letters the model mixes up.")

# -------------------- My Progress --------------------
elif page == "My Progress":
    st.markdown('<div class="section-title">My Progress & Achievements</div>', unsafe_allow_html=True)
    history = st.session_state.prediction_history
    attempts = len(history)
    avg_conf = sum(x["confidence"] for x in history) / attempts if attempts else 0
    challenge_done = st.session_state.challenge_attempts
    challenge_acc = st.session_state.challenge_score / challenge_done if challenge_done else 0
    a, b, c, d = st.columns(4)
    a.metric("Predictions", attempts); b.metric("Average confidence", f"{avg_conf:.1%}" if attempts else "—")
    c.metric("Practice correct", st.session_state.practice_score); d.metric("Challenge accuracy", f"{challenge_acc:.1%}" if challenge_done else "—")
    st.markdown("#### Recognition activity")
    if history:
        history_df = pd.DataFrame(list(reversed(history)))
        st.dataframe(history_df, **_STRETCH, hide_index=True)
        st.markdown("#### Most frequently predicted letters")
        counts = pd.Series([x["letter"] for x in history]).value_counts().sort_values(ascending=False).head(10)
        st.bar_chart(counts)
        st.download_button("Export session history", pd.DataFrame(history).to_csv(index=False).encode("utf-8"), "signbridge_session_history.csv", "text/csv")
    else:
        st.info("No predictions recorded yet. Try Recognition Mode, Guided Practice or ISL Challenge.")
    st.markdown("#### Achievements")
    badges = [
        ("First Steps", attempts >= 1, "Complete your first prediction"),
        ("Explorer", attempts >= 10, "Record 10 predictions"),
        ("Practice Starter", st.session_state.practice_attempts >= 5, "Make 5 guided-practice attempts"),
        ("Streak Builder", st.session_state.practice_best_streak >= 3, "Get 3 correct practice attempts in a row"),
        ("Challenge Finisher", challenge_done >= 5, "Complete at least 5 challenge attempts"),
        ("Word Builder", len(st.session_state.spelled_text.strip()) >= 3, "Build a message with 3 or more characters"),
    ]
    badge_cols = st.columns(3)
    for i, (title, earned, description) in enumerate(badges):
        with badge_cols[i % 3]:
            status = "UNLOCKED" if earned else "LOCKED"
            status_cls = "on" if earned else "off"
            st.markdown(f'<div class="feature-card"><div class="status {status_cls}">{status}</div><div class="feature-title">{title}</div><div class="feature-copy">{description}</div></div>', unsafe_allow_html=True)
    st.caption("History, achievements and scores are session-only. They reset when the Streamlit session restarts; no account or database is connected.")

# -------------------- How It Works --------------------
elif page == "How It Works":
    st.markdown('<div class="section-title">How SignBridge Works</div>', unsafe_allow_html=True)
    st.write("SignBridge uses a client–API–model architecture to convert a single static ISL alphabet image into a letter and optional speech output.")
    stages = [
        ("01", "Image Capture", "Streamlit camera input or image upload"),
        ("02", "Hand Detection", "MediaPipe hand detection (as described in the project design)"),
        ("03", "Preprocessing", "Prepare the hand image for model inference"),
        ("04", "CNN Classification", "TensorFlow/Keras CNN predicts an alphabet class"),
        ("05", "FastAPI Response", "Backend returns the predicted letter and confidence"),
        ("06", "Communication Output", "Streamlit displays text; gTTS generates optional speech"),
    ]
    for number, title, description in stages:
        x, y = st.columns([1, 5])
        x.markdown(f'<div class="metric-card"><div class="metric-value">{number}</div></div>', unsafe_allow_html=True)
        y.markdown(f'<div class="feature-card"><div class="feature-title">{title}</div><div class="feature-copy">{description}</div></div>', unsafe_allow_html=True)
    st.markdown("#### Technology stack")
    tech = pd.DataFrame({"Layer": ["Frontend", "Backend API", "Computer vision", "Model", "Training", "Speech", "Data processing"], "Technology": ["Streamlit", "FastAPI", "MediaPipe", "TensorFlow / Keras CNN", "Google Colab", "gTTS / pyttsx3", "PySpark"]})
    st.dataframe(tech, **_STRETCH, hide_index=True)
    st.markdown("#### Current scope and limitations")
    st.warning("The deployed API currently accepts one image and returns one alphabet prediction. It does not provide continuous video inference, dynamic sign recognition, complete word/sentence recognition or a translation engine. The architecture steps above describe the project design; verify which preprocessing and hand-detection steps are active in the deployed backend before claiming them as implemented.")
    st.caption("Project workflow reference: Webcam → hand detection → image preprocessing → CNN → ISL letter → text → speech.")

st.markdown("---")
st.markdown('<div class="footer-note">SignBridge · ISL alphabet recognition &amp; communication</div>', unsafe_allow_html=True)
