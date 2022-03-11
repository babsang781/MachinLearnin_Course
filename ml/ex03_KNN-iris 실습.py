# 문제 정의
-붓꽃(iris) 의 품종을 구분하는 머신러닝 모델을 만들어보자

# 데이터 수집
-sklearn 에서 학습용 데이터로 제공

from sklearn.datasets import load_iris
iris= load_iris()

# iris 데이터가 딕셔너리 형태여서 키 값을 따로 확인할 필요가 있음
iris.keys() 

iris

# DESCR 은 데이터 상세 설명으로, 
# 개행 문자가 포함되어 있어서 print 로 해줘야 깔끔하게 나옴
print(iris['DESCR'])

iris['feature_names']

# 데이터 전처리 - 생략

# 탐색적 데이터 분석

## 데이터 셋 구성하기 
-문제와 정답으로 구분하기

import pandas as pd
X = pd.DataFrame(iris['data'], columns = iris['feature_names'])
X.head()

y=iris['target']

# 여러 특성을 토대로 한 종류별 분류표를 그려줌 
pd.plotting.scatter_matrix(X,
figsize = (15,15),
marker='^',
c = y,
alpha = 1
)

# 모델 선택 및 하이퍼 파라미터 튜닝

## train test 분할

y

# X (문제) , y(정답) 을 넣으면 X_train, X_test, y_train, y_test로 만듦
# 데이터를 섞어줌
# train과 test의 비율을 조절
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state = 6)

# X_train, X_test, y_train, y_test 의 순서를 맞추어줘야 함

X_train.shape, y_test.shape

## 모델 로드

from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors = 9, p=2, weights='distance')

# 학습

knn_model.fit(X_train, y_train)

# 평가

knn_model.score(X_train, y_train)

knn_model.score(X_test, y_test)

# 어떤 게 적합한 모델인지를 생ㅎㅏㄱ하면서 하기

# 하이퍼 파라미터 튜닝
- 5, 6, 7 을 한 번에 진행
- k 값에 따라 train 과 test 의 score 가 어떻게 변하는지 확인

train_score = []
test_score = []

n_neighbors_set = range(1,51)

for n_n in n_neighbors_set:
    
    # 모델 생성
    knn_model = KNeighborsClassifier(n_neighbors = n_n)
    
    # 모델 학습
    knn_model.fit(X_train, y_train)
    
    # 점수 확인 
    train_knn = knn_model.score(X_train, y_train)
    test_knn = knn_model.score(X_test, y_test)
    
    # 점수 입력
    train_score.append(train_knn)
    test_score.append(test_knn)

train_score

test_score

# 시각화
import matplotlib.pyplot as pit

pit.plot(n_neighbors_set, train_score, label = 'Train')
pit.plot(n_neighbors_set, test_score, label = 'Test')

# test 데이터가 train 데이터 보다 높은 경우는 
# 데이터가 적어서 모델이 그렇게 나온 경우

# k 의 권장 수
# 샘 권장 수 : 데이터의 총 개수 / 정답 종류의 개수
# 105 / 2 / 3

# GridSearch
- 하이퍼 파라미터 튜닝을 쉽게 해주는 도구
- 하이퍼파라미터의 범위를 저장해놓으면
  학습을 전부 진행하고 가장 좋은 값을 출력

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors' : range(1,13),
    'p' : [1,2],
    'weights' : ['uniform' , 'distance']
}

# cv=5
# 전체 데이터를 5 등분하고 하나를 테스트, 
# 나머지 4 개를 train으로 사용
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)

grid.fit(X_train, y_train)

grid.best_score_

grid.best_params_

df_eval



