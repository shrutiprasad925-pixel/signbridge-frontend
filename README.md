# SignBridge — Panel-ready Streamlit Frontend

A Streamlit frontend for the deployed SignBridge FastAPI prediction service.

## Included modules

- **Home Dashboard:** platform overview, backend health and session statistics.
- **Recognition Mode:** webcam still capture or image upload, API prediction, confidence feedback, response time and speech playback.
- **Guided Practice:** target alphabet, difficulty bands, camera attempts, retries, score and streak tracking.
- **ISL Challenge:** timed-by-round alphabet challenge with score, accuracy and streaks.
- **Live Spelling & Word Builder:** confirm one predicted letter at a time, edit the message and generate speech.
- **ISL Alphabet Reference:** A–Z reference handshapes loaded from `assets/isl/`, with a single-letter viewer and a full browse grid. Letters without an image show a neutral "Reference image not added" tile.
- **Model Insights:** upload a labelled test CSV to calculate accuracy, macro precision, macro recall, macro F1-score, confusion matrix and per-letter metrics.
- **My Progress:** session history, frequent predictions, CSV export and achievement badges.
- **How It Works:** architecture, workflow, technology stack and scope/limitations.

## Run locally

Use Python 3.10 or newer with compatible Streamlit wheels.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Windows Device Guard blocks direct `pip.exe` or `streamlit.exe`, use the `python -m ...` commands above.

## Adding the ISL reference images

The app reads one image per letter from `assets/isl/` (`A.png` to `Z.png`). To build them from your ISL dataset (one folder per letter):

```bash
python prepare_sign_images.py PATH_TO_DATASET
```

The script keeps the sharpest image for each letter and saves a uniform square version. Restart the app afterwards. Commit the `assets/isl/` folder if you deploy on Streamlit Community Cloud.

## Backend

Configured API: `https://signbridge-backend-1-y2dy.onrender.com`

- `GET /health` checks service health.
- `POST /predict` accepts an image as multipart form field `file`.
- Expected response fields: `success`, `letter`, `confidence`, `message`.

## Model Insights CSV format

Upload a CSV containing one row per held-out test image, with columns named `actual` and `predicted` (or `y_true` and `y_pred`). Example structure only:

```csv
actual,predicted
A,A
B,C
C,C
D,D
```

The values above are illustrative and are **not** SignBridge evaluation results. Use predictions from a labelled test set that was not used for training. The dashboard computes metrics from the uploaded rows; it does not invent or assume model accuracy.

## Important scope and limitations

- The deployed backend predicts one static ISL alphabet sign from one image. It does not expose continuous video streaming or whole-word/continuous-sign recognition.
- Streamlit webcam capture is a still image; each practice/challenge attempt sends one captured image.
- Challenge and practice scores compare the model's predicted letter with the selected target. They are educational exercises, not assessments of sign-language fluency.
- Confidence is the backend's reported score, not a guarantee of correctness. The UI uses confidence bands only as guidance.
- Reference images are taken from the project's own ISL dataset (see below). No ASL images or generated handshapes are used.
- The How It Works page distinguishes the intended project architecture from the deployed endpoint. Verify that MediaPipe and preprocessing steps are active in the current backend before presenting them as deployed functionality.
- Speech uses gTTS and requires internet access. Hindi and Marathi output currently speaks the letter name or supplied text; it is not a full translation engine.
- History, scores and badges are stored in Streamlit session state only. They reset when the session restarts; there is no login, persistent database or cross-session leaderboard.
- Captured/uploaded images are sent to the configured backend for inference. This frontend does not save them to local disk.

## Privacy note

This frontend does not intentionally persist captured images. The image is transmitted to the configured backend for prediction. Review the backend hosting and retention configuration separately before making privacy guarantees to end users.
