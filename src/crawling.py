import feedparser
import json

def crawl_website(query, max_articles=5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:max_articles]]

# 실행
results =[]
titles = crawl_website("인공지능")
for i, t in enumerate(titles, 1):
    res = f"{i}.{t}"
    print(res)
    results.append(res)
with open("crawling_result.json","w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)