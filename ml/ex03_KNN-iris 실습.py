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

y









