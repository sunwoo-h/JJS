from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# 환경변수 불러오기
load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

MODEL_NAME = "Junginn/kcelectra-toxic-comment-detector_V1"  # 실제 모델 경로로 수정
TOKENIZER_NAME = "beomi/KcELECTRA-base"   # ✅ 토크나이저는 베이스에서


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    token=HF_AUTH_TOKEN,       # use_auth_token 대신
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    token=HF_AUTH_TOKEN       # 동일하게 token 파라미터 사용
)
model.tokenizer = tokenizer  # infer 함수에서 쓰기 위해
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(request: Request):
    text = request.text.strip()

    inputs = model.tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=150,
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    label = torch.argmax(probs, dim=-1).item()
    prob = probs[0][label].item()
    label_text = '악플' if label == 0 else '일반 댓글'

    # 🔴 확신도 기반 색상 정의
    color = None
    if label == 0:  # 악플로 예측된 경우
        if prob >= 0.65:
            color = "red"
        elif prob >= 0.5:
            color = "orange"
        else:
            label_text = '일반 댓글'  # 악플로 예측됐지만 확신 낮음 → 일반으로 간주

    return {
        "text": text,
        "predicted_label": label,
        "label_name": label_text,
        "probability": round(prob, 4),
        "confidence_color": color
    }
# static 디렉토리 연결
app.mount("/static", StaticFiles(directory="static"), name="static")

# 루트 경로에 index.html 반환
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
