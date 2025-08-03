from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import feedparser

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