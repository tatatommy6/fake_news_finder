from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form
from fastapi import Body

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
import urllib.parse
import feedparser
import torch

#가짜 뉴스 탐지 모델 로드
model_name = "/Users/kimminkyeol/Desktop/fake_news_detect"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 클래스 번호 -> 라벨 이름 매핑
label_map = {
    0: "FAKE",
    1: "REAL"
}


#기본 객체 선언
app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")
templates = Jinja2Templates(directory = "templates")


def crawl_website(query: str, max_articles: int = 5):
    encoded_query = urllib.parse.quote(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title if hasattr(entry, "source") else "Unknown"
        })

    return articles


def extract_article_body(url: str) -> dict:
    try:
        article = Article(url, language="ko")
        article.download()
        article.parse()
        
        return {
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None
        }
    except Exception as e:
        print(f"[ERROR] {url} 기사 본문 추출 실패: {e}")
        return {
            "title": None,
            "text": None,
            "publish_date": None
        }


#메인 페이지
@app.get("/", response_class = HTMLResponse) # HTMLResponse:HTML 형식의 응답을 브라우저로 보내기 위한 FastAPI의 응답 클래스 / response_class: "이 경로는 HTML 반환이야" 라고 FastAPI에 알려주는 역할
async def get_index(request: Request):
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


# 예측 페이지
@app.post("/predict", response_class=HTMLResponse)
async def post_predict(request: Request):
    try:
        data = await request.json() # await: 비동기 함수의 실행 결과를 기다리기 위해 사용하는 키워드임 이 작업이 끝나기 전까진 다음 줄로 안 넘어감
        titles = data.get("titles", [])

        results = []
        for title in titles:
            inputs = tokenizer(title, 
                                return_tensors="pt", # PyTorch 텐서 형식으로 반환
                                truncation=True,  #512 토큰 넘으면 자름
                                padding=True,  # 입력 길이를 맞춰줌
                                max_length=512)
            with torch.no_grad(): #추론 모드
                outputs = model(**inputs) # **inputs: 딕셔너리 형태로 언팩하여 모델에 전달
                probs = torch.softmax(outputs.logits, dim=1) # softmax 함수를 사용하여 확률로 변환
                pred = torch.argmax(probs, dim=1).item() # 가장 높은 확률을 가진 클래스 인덱스
                confidence = probs[0][pred].item() 

            # 결과를 빈 리스트에 추가
            results.append({
                "title": title,
                "label": label_map[pred],
                "confidence": f"{confidence * 100:.2f}" # % 형식으로 변환
            })

        return templates.TemplateResponse("predict.html", { #templates.TemplateResponse: predict.html에 아래의 데이터를 전달해주는 함수
            "request": request,
            "results": results
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/repredict", response_class=HTMLResponse)
async def post_repredict(request: Request, data: dict = Body(...)):
    articles = data.get("articles", [])
    results = []
    for article in articles:
        print(f"[INFO] 본문 추출 중: {article['title']}")
        extracted = extract_article_body(article["link"])
        
        content = extracted["text"] or article["title"]
        inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        results.append({
            "title": article["title"],
            "source": article.get("source", "Unknown"),
            "link": article["link"],
            "label": label_map[pred],
            "confidence": f"{confidence * 100:.2f}",
            "text": extracted["text"]
        })

    return templates.TemplateResponse("repredict.html", {
        "request": request,
        "results": results
    })
