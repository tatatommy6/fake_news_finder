import feedparser

def crawl_website(query, max_articles=5):
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:max_articles]]

# 실행
titles = crawl_website("인공지능")
for i, t in enumerate(titles, 1):
    print(f"{i}. {t}")