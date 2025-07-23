import feedparser
import json
import sys

def crawl_website(query, max_articles=5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(url)
    return [f"{i+1}. {entry.title}" for i, entry in enumerate(feed.entries[:max_articles])]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("검색어가 필요합니다.")
        sys.exit(1)

    keyword = sys.argv[1]
    titles = crawl_website(keyword)
    
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=4)
