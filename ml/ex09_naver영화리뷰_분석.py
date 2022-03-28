!pip install --upgrade pip

import sys
sys.version

!pip install JPype1-1.1.2-cp38-cp38-win_amd64.whl

!pip install konlpy

# 한글 형태소 분류기
from konlpy.tag import Okt, Kkma  # Kkma와 Okt 가 성능이 좋은 편

okt = Okt()
okt.morphs('아버지가방에들어가신다')  # 형태소별로 내용을 분류

kkma = Kkma()
kkma.morphs('아버지가방에들어가신다')

text = '아버지가방에들어가신다'
okt.pos(text)  # 형태소별로 내용 및 형태소 분류

kkma.pos(text)

# tagset : 분류할 수 있는 형태소를 나열
okt.tagset

kkma.tagset

okt.nouns(text)  # 명사만 추출

!pip install sklearn numpy pandas matplotlib

# countvectorizer 와 연결해서 사용하기 (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

text = ['검색하려면 여기에 입력하십시오.',
       '오전 10:45 2022-03-28',
       '네이버 영화리뷰 분석 재밌겠다']

# 토큰화 및 단어사전 구축
cv.fit(text)  

# 띄어쓰기 단위로 토큰화 하기
cv.vocabulary_

# 토큰화 방법 정의 함수
def mytoken(text):
    return okt.nouns(text)

cv_okt= CountVectorizer( tokenizer = mytoken )
cv_okt.fit(text)  

cv_okt.vocabulary_

# 문제 정의
### 한글영화 리뷰 데이터 셋으로 감성분석을 진행
### KoNLPy 를 사용하여 형태소 분류

# 데이터 수집

import pandas as pd

text_train = pd.read_csv('./data/ratings_train.txt', delimiter ='\t')
text_test = pd.read_csv('./data/ratings_test.txt', delimiter ='\t')

text_train.shape, text_test.shape

# 0은 부정 1은 긍정으로 라벨링 되어있음.
text_test.tail()

text_train.tail()

# 데이터 전처리

# 결측치 확인
text_train.info()

# 결측치 삭제 dropna()  : 결측치가 존재하는 행을 삭제
text_train.dropna(inplace = True)
text_test.dropna(inplace = True)

X_train = text_train['document'][:10000]
y_train = text_train['label'][:10000]
X_test = text_test['document'][:1000]
y_test = text_test['label'][:1000]

## 토큰화

def mytoken(text):
    return okt.nouns(text)

cv_okt = CountVectorizer( tokenizer = mytoken )

cv_okt.fit(X_train)

len(cv_okt.vocabulary_)

# 수치화
X_train_okt = cv_okt.transform(X_train)
X_test_okt = cv_okt.transform(X_test)

X_train_okt.shape

# 탐색적 데이터 분석 
### 여기서는 생략

# 모델선택 및 하이퍼 파라미터 튜닝

# 선형회귀모델 사용
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

# 학습

lr_model.fit(X_train_okt, y_train)

# 평가

lr_model.score(X_train_okt, y_train)

lr_model.score(X_test_okt, y_test)

# 감성분석

voca = cv_okt.vocabulary_  # 단어사전
word_weight = lr_model.coef_  # 선형회귀를 통한 단어 가중치                 

df = pd.DataFrame([voca.keys(),
                  voca.values()]) # voca 데이터를 데이터 프레임으로 변환

df = df.T
df_sorted = df.sort_values(by = 1)  # 단어사전 index 순서대로 정렬
df_sorted['coef'] = word_weight.reshape(-1)  # 가중치를 데이터 프레임에 추가
df_sorted.sort_values(by ='coef' , inplace = True)  # 가중치를 기준으로 정렬
top30_df = pd.concat([
    df_sorted.head(30), # 가중치가 높은 30개
    df_sorted.tail(30)  # 가중치가 낮은 30갸
])

import matplotlib.pyplot as plt
import matplotlib

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\malgun.ttf").get_name()
rc('font',family=font_name)  # 한글 지원 폰투로 명경

matplotlib.rcParams['axes.unicode_minus'] = False    # 값 변경은 
plt.figure(figsize=(15,5)) # 가로,세로 비율
plt.bar(top30_df[0], top30_df['coef']) # x축은 단어, y축은 가중치
plt.xticks(rotation = 90) # x축 눈금 각도 조정
plt.show()


