from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_LOCAL_DIR = os.path.join(BASE_DIR, "hf_model")
TOKENIZER_DIR = os.path.join(HF_LOCAL_DIR, "tokenizer")
MODEL_DIR = os.path.join(HF_LOCAL_DIR, "model")

# ★ 런타임 부트스트랩: 폴더 없거나 비어 있으면 즉시 다운로드
def ensure_hf_assets():
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    def is_empty(path: str) -> bool:
        return (not os.path.isdir(path)) or (len(os.listdir(path)) == 0)

    need_tokenizer = is_empty(TOKENIZER_DIR)
    need_model = is_empty(MODEL_DIR)

    if need_tokenizer or need_model:
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_AUTH_TOKEN")  # 있으면 사용, 없어도 공개모델이면 동작

        if need_tokenizer:
            snapshot_download(
                repo_id="beomi/KcELECTRA-base",
                local_dir=TOKENIZER_DIR,
                local_dir_use_symlinks=False,   # 심볼릭 링크 금지
                token=hf_token,
            )

        if need_model:
            snapshot_download(
                repo_id="Junginn/kcelectra-toxic-comment-detector_V1",
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,    # 심볼릭 링크 금지
                token=hf_token,
            )

def _ensure_dir(path, label):
    if not os.path.isdir(path):
        raise RuntimeError(f"[Startup] {label} 디렉터리가 없습니다: {path}")
    if not os.listdir(path):
        raise RuntimeError(f"[Startup] {label} 디렉터리가 비어 있습니다: {path}")

# ↓↓↓ 여기서 보장
ensure_hf_assets()
_ensure_dir(TOKENIZER_DIR, "TOKENIZER")
_ensure_dir(MODEL_DIR, "MODEL")

# 토크나이저/모델 로드
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    local_files_only=True,   # 이제 로컬에 확실히 존재함
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    low_cpu_mem_usage=True,
)

# (선택) 양자화
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

model.eval()
torch.set_grad_enabled(False)
torch.set_num_threads(1)
device = torch.device("cpu")
model.to(device)

app = FastAPI()

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(request: Request):
    text = request.text.strip()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=150,
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    label = torch.argmax(probs, dim=-1).item()
    prob = probs[0][label].item()
    label_text = "악플" if label == 0 else "일반 댓글"

    color = None
    if label == 0:
        if prob >= 0.65:
            color = "red"
        elif prob >= 0.5:
            color = "orange"
        else:
            label_text = "일반 댓글"

    return {
        "text": text,
        "predicted_label": label,
        "label_name": label_text,
        "probability": round(prob, 4),
        "confidence_color": color,
    }

# static 연결
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()
