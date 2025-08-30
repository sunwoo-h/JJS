from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # ← 여기
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ← 절대경로로 안전하게
HF_LOCAL_DIR = os.path.join(BASE_DIR, "hf_model")
TOKENIZER_DIR = os.path.join(HF_LOCAL_DIR, "tokenizer")
MODEL_DIR = os.path.join(HF_LOCAL_DIR, "model")

# ★ 토크나이저: AutoTokenizer 사용 (fast 자동 판단)
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_DIR,
    local_files_only=True,
)

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    low_cpu_mem_usage=True,
)

# 양자화(선택)
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

    # ★ 여기서 tokenizer 전역변수를 사용해야 합니다!
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=150,
        return_attention_mask=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    label = torch.argmax(probs, dim=-1).item()
    prob = probs[0][label].item()
    label_text = '악플' if label == 0 else '일반 댓글'

    color = None
    if label == 0:
        if prob >= 0.65:
            color = "red"
        elif prob >= 0.5:
            color = "orange"
        else:
            label_text = '일반 댓글'

    return {
        "text": text,
        "predicted_label": label,
        "label_name": label_text,
        "probability": round(prob, 4),
        "confidence_color": color
    }

# static 연결
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()
