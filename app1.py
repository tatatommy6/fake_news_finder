from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import feedparser
import torch

model_name = "/Users/kimminkyeol/Desktop/fake_news_detect"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 클래스 번호 -> 라벨 이름 매핑
label_map = {
    0: "FAKE",
    1: "REAL"
}


app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")
templates = Jinja2Templates(directory = "templates")


def crawl_website(query: str, max_articles: int = 5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    return [f"{i+1}. {entry.title}" for i, entry in enumerate(feed.entries[:max_articles])]


@app.get("/", response_class = HTMLResponse)
async def get_index(request: Request):
    print("Index page accessed")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search_news(request: Request):
    try:
        data = await request.json()
        print("받은 JSON:", data)
        keyword = data.get("keyword", "")
        if not keyword:
            print("검색어가 없음")
            return JSONResponse(status_code=400, content={"error": "검색어가 없습니다."})
        result = crawl_website(keyword, 5)
        return {"results": result}
    except Exception as e:
        print("예외 발생:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.post("/predict", response_class=HTMLResponse)
async def post_predict(request: Request):
    keyword = "네가 index에서 검색한 키워드"  # 또는 form에서 받아올 수도 있음
    articles = crawl_website(keyword, 5)

    results = []
    for title in articles:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        results.append({
            "title": title,
            "label": label_map[pred],
            "confidence": f"{confidence*100:.2f}"
        })

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "results": results
    })