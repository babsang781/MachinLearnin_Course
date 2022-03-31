# 토픽 모델링
### 비지도 학습
### 하나 또는 여러가지 문서를 토픽으로 할당하는 작업
### LDA(Latent Dirichlet Allocation, 잠재 디리클레 할당)

# LDA 
### 자주 나타나는 단어의 그룹(토픽)을 찾는 것
### 주제와는 거리가 멀 수도 있음
### 의미가 있는 성분을 찾는 것이기 때문

!pip install pandas numpy sklearn mglearn

## 데이터 받아오기

import pandas as pd

df_train = pd.read_csv('data/ratings_train.txt', delimiter = '\t')
df_test = pd.read_csv('data/ratings_test.txt', delimiter = '\t')

## 결츳치 제거

df_train.dropna(inplace = True)
df_test.dropna(inplace = True)

#### 필요한 텍스트 데이터 컬럼만 가진 text_train 생성
text_train = df_train['document']

## 토큰화 

from sklearn.feature_extraction.text import CountVectorizer

### max_features = 가장 많이 등장하는 단어만 사용
#### : 빈도 상위 10,000 까지만 사용하기
### max_df = .15 : 15% 이상의 문서에서 등장하는 단어 제거 
cv = CountVectorizer(max_features = 10000, max_df = .15)
X = cv.fit_transform(text_train)

###14995 개 문장을 10000 개 단어로 표시한 데이터를 벡터화
X

## LDA

from sklearn.decomposition import LatentDirichletAllocation

## n_components = 10 : 10개의 토픽그룹을 만들어주세요
## learning_method = 'batch': 기본값 online, batch 느리지만 성능이 좋음

lda = LatentDirichletAllocation(n_components = 10, learning_method = 'batch',
                               max_iter = 25, random_state = 0)

document_topics = lda.fit_transform(X)

#### 각 토픽은 지정한 개수로 그룹화되어있고, 각각의 합은 1 
document_topics[0]

### 토픽 확인

import numpy as np
import mglearn

#### 비지도 학습이기 때문에 어떤 내용으로 될지 확실히 알 수 없음


sorting = np.argsort(lda.components_, axis = 1)[:, ::-1]
feature_names = np.array(cv.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names = feature_names, 
                           sorting = sorting, topics_per_chunk = 5, n_words = 10)

#### 문서의 크기가 커진다면 과소 적합을 예방할 수 있음
text_train[0]

document_topics[0]
