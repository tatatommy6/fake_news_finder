from flask import Flask, render_template, request, redirect
from transformers import AutoTokenizer, BertForSequenceClassification
from os.path import join, dirname
import torch, json, subprocess


app = Flask(__name__)

#모델이 있을떄만
# model_path = "/Users/kimminkyeol/Programming/fake_news_finder/server/kobert_clickbait_model_final"

# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
# model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors = "pt", truncation = True, padding = "max_length", max_length = 512)
    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(prob, dim = 1).item()
        confidence = prob[0][pred].item()*100
    label = "가짜뉴스" if pred == 1 else "진짜 뉴스"
    return label, confidence


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crawl', methods=['POST'])
def crawl():
    keyword = request.form.get("keyword","")
    subprocess.run(["python", "crawling.py", keyword], check=True)
    
    with open("result.json","r", encoding = "utf-8") as f:
        titles = json.load(f)
    return render_template('index.html', titles=titles)


@app.route('/predict', methods=['POST'])
def predict_route():
    with open("result.json", "r", encoding="utf-8") as f:
        titles = json.load(f)

    results = []
    for title in titles:
        label, confidence = predict(title)
        results.append({
            "title": title,
            "label": label,
            "confidence": f"{confidence:.2f}%"
        })

    return render_template('result.html', titles=titles, results=results)


if __name__ == '__main__':
    app.run(debug=True, host='5000')