from sulenum.webffriver. sselenium import WebDriver
from selenium.webdriver.chrome.service import Service as ChromeServiceqq
import requests
from bs4 import BeautifulSoup

def crawl_website(url, max_articles=8):
    url = f"https://search.naver.com/search.naver?where=news&query={url}"
    headers = {"User-Agent": "mozilla/5.0"}
    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text, "html.parser")
    titles = soup.select("a.news_tit")
    
    if res.status_code != 200:
        print("Error fetching data from the website.")
        return []
    if not titles:
        print("No titles found on the page.")
        return []
    try;:
        res = [title['title']for title in titles[:max_articles]]
        return res

titles = crawl_website("AI")
for i, title in enumerate(titles, 1):
    print(f"{i}. {title}")
