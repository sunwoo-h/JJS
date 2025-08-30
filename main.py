from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, requests

HF_MODEL_ID = "Junginn/kcelectra-toxic-comment-detector_V1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Render에 Protected로 설정
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

app = FastAPI()

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Request):
    text = (req.text or "").strip()
    if not text:
        return {"error": "empty input"}

    # HF Inference API로 프록시
    resp = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": text}, timeout=15)
    if resp.status_code == 503:
        # 모델 콜드스타트 중일 수 있음 → 잠시 후 재요청 안내
        return {"error": "Model is loading. Please retry in a few seconds."}
    if not resp.ok:
        return {"error": f"HF API error {resp.status_code}", "detail": resp.text[:300]}

    data = resp.json()
    # 응답 형태: [[{"label": "LABEL_0", "score": 0.87}, ...]]
    preds = data[0] if isinstance(data, list) else []
    best = max(preds, key=lambda x: x.get("score", 0.0)) if preds else {"label": "LABEL_1", "score": 0.5}

    label_id = 0 if best["label"].endswith("0") else 1
    prob = float(best["score"])
    label_text = "악플" if label_id == 0 else "일반 댓글"

    color = None
    if label_id == 0:
        if prob >= 0.65: color = "red"
        elif prob >= 0.5: color = "orange"
        else:
            label_text = "일반 댓글"

    return {
        "text": text,
        "predicted_label": label_id,
        "label_name": label_text,
        "probability": round(prob, 4),
        "confidence_color": color,
        "provider": "hf-inference-api"
    }

# 정적 파일(선택)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join("static", "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Toxic Comment Detector (Proxy Mode)</h1>"
