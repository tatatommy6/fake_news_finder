import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. 데이터 불러오기
df_fake = pd.read_csv('1/Fake.csv')
df_real = pd.read_csv('1/True.csv')

df_fake['label'] = 1 #1: 가짜 뉴스
df_real['label'] = 0 #0: 진짜 뉴스

# 2. 두 데이터셋 합치기
df = pd.concat([df_fake, df_real]).reset_index(drop = True)

# 3. 전처리 및 라벨 변환
X = df['text']
y = df['label']  # 0: 진짜, 1: 가짜

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# 4. 학습용/평가용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, 
    random_state=42
    )

# 5. 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# 데이터셋 다운로드
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

# print("Path to dataset files:", path)