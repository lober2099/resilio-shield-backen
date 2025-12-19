from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# 1. 极简语义引擎加载 (使用 L3 版本以节省内存)
# 第一次启动时会下载模型，大约需要 1 分钟
try:
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")

# 2. 预定义逻辑基准
FIRM_ANCHORS = ["目前项目优先级已满", "资源无法支持", "建议重新对齐权重"]
FIRM_EMBEDDINGS = model.encode(FIRM_ANCHORS, convert_to_tensor=True)

# 3. 数据模型
class UserInput(BaseModel):
    text: str
    context: str = "default"

# 4. 核心路由：语义审计
@app.post("/audit")
async def audit_text(user_input: UserInput):
    text = user_input.text
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # 语义相似度计算
    user_emb = model.encode(text, convert_to_tensor=True)
    cos_sim = torch.max(util.cos_sim(user_emb, FIRM_EMBEDDINGS)).item()
    
    # 判定逻辑
    intent = "PROFESSIONAL_FIRM" if cos_sim > 0.6 else "SUBMISSIVE_OR_NEUTRAL"
    
    return {
        "intent": intent,
        "score_delta": round(cos_sim * 20, 2),
        "guidance": "保持逻辑，不要道歉" if intent == "SUBMISSIVE_OR_NEUTRAL" else "防御稳健"
    }

@app.get("/")
async def root():
    return {"status": "Shield of Resilience API is Online"}
