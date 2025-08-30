from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os, time, requests
from typing import List, Dict, Any

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Junginn/kcelectra-toxic-comment-detector_V1")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

app = FastAPI(title="Toxic Comment Detector (Proxy Mode)")

# 정적 파일(있을 때만)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class RequestBody(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model": HF_MODEL_ID, "has_token": bool(HF_API_TOKEN)}

@app.get("/hf-debug")
def hf_debug():
    """HF API 연결/권한/모델ID 점검용."""
    try:
        resp = requests.post(
            HF_API_URL,
            headers=HEADERS,
            params={"wait_for_model": "true"},  # 콜드스타트 자동 대기
            json={"inputs": "테스트 문장"},
            timeout=60,
        )
        return {
            "status_code": resp.status_code,
            "ok": resp.ok,
            "preview": resp.text[:500],
        }
    except Exception as e:
        return {"error": str(e)}

def parse_hf_output(data: Any) -> Dict[str, Any]:
    """HF 출력 -> 단일 라벨/확률로 정리."""
    preds: List[Dict[str, Any]] = []
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            preds = data[0]              # [[{...}]] 형태
        elif data and isinstance(data[0], dict):
            preds = data                 # [{...}] 형태
    if not preds:
        return {"label_id": 1, "prob": 0.5, "label_text": "일반 댓글", "color": None}

    best = max(preds, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label", "LABEL_1")).upper()
    prob = float(best.get("score", 0.0))

    # LABEL_0/LABEL_1 or 텍스트 라벨 방어
    if raw_label.endswith("0"):
        label_id = 0
    elif raw_label.endswith("1"):
        label_id = 1
    else:
        label_id = 0 if "TOXIC" in raw_label else 1

    label_text = "악플" if label_id == 0 else "일반 댓글"

    # 확신 낮은 악플은 일반 처리(기존 로직 유지)
    color = None
    if label_id == 0:
        if prob >= 0.65:
            color = "red"
        elif prob >= 0.5:
            color = "orange"
        else:
            label_id, label_text, color = 1, "일반 댓글", None

    return {"label_id": label_id, "prob": prob, "label_text": label_text, "color": color}

def call_hf(text: str, max_retries: int = 3) -> requests.Response:
    """503(콜드스타트) 시 지수 백오프로 재시도."""
    delay = 2
    for attempt in range(1, max_retries + 1):
        resp = requests.post(
            HF_API_URL,
            headers=HEADERS,
            params={"wait_for_model": "true"},  # ★ 대기
            json={"inputs": text},
            timeout=60,                          # ★ 넉넉한 타임아웃
        )
        if resp.status_code != 503:
            return resp
        print(f"[HF] loading 503 (try {attempt}/{max_retries}) → retry in {delay}s")
        time.sleep(delay)
        delay *= 2
    return resp

@app.post("/predict")
def predict(req: RequestBody):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty input"})

    try:
        resp = call_hf(text)
        if not resp.ok:
            # 서버 콘솔 로그로 원문 남겨 원인 파악
            print("[HF][ERROR]", resp.status_code, resp.text[:1000])
            if resp.status_code == 503:
                return JSONResponse(status_code=503, content={"error": "Model is loading. Try again shortly."})
            if resp.status_code in (401, 403):
                return JSONResponse(status_code=502, content={"error": "Auth failed. Check HF_API_TOKEN."})
            if resp.status_code == 404:
                return JSONResponse(status_code=502, content={"error": "Model not found. Check HF_MODEL_ID."})
            if resp.status_code == 429:
                return JSONResponse(status_code=429, content={"error": "Rate limited by HF. Slow down."})
            return JSONResponse(status_code=resp.status_code, content={"error": "HF API error", "detail": resp.text[:500]})

        data = resp.json()
        parsed = parse_hf_output(data)
        return {
            "text": text,
            "predicted_label": parsed["label_id"],
            "label_name": parsed["label_text"],
            "probability": round(parsed["prob"], 4),
            "confidence_color": parsed["color"],
            "provider": "hf-inference-api",
            "model": HF_MODEL_ID,
        }

    except requests.Timeout:
        return JSONResponse(status_code=504, content={"error": "Upstream timeout"})
    except Exception as e:
        print("[SERVER][EXC]", repr(e))
        return JSONResponse(status_code=500, content={"error": "Server error", "detail": str(e)})

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join("static", "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<h1>Toxic Comment Detector (Proxy Mode)</h1><p>Model: {HF_MODEL_ID}</p><p>POST /predict</p>"
