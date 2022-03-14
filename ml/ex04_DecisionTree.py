



# 문제정의
### 버섯의 특징을 사용해서 독/ 식용 버섯을 분류

## 목표
### Decision Tree 과대적합 제어/ 시각화 /특성 선택

# 데이터 수집

import pandas as pd
data = pd.read_csv('./data/mushroom.csv')
data.head()

data.shape

# 전처리

## 결측치

data.info()

# EDA 탐색적 데이터 분석

# 모델 선택 및 하이퍼 파라미터 튜닝

data.head()

X = data.loc[ : , 'cap-shape' : ]
y = data.loc[ : , 'poisonous']

## 인코딩
### Lavel 인코딩: 글자데이터를 수치 데이터로 변환하는 작업

y.head()

### Label Encoding : 수치값을 직접 대입하여 mapping 하는 작업
### One-hot Encoding : 0 , 1 의 값을 가진 여러 개의 새로운 특성으로 변경
#### X_one_hot = pd.get_dummies(X2)
##### 원핫의 경우 하나 하나 떼어서 하나의 칼럼으로 구분하여 분석할 수 있는 점이 분석에서 도움이 될 수 있음.
##### 결정트리에서, 머신 모델이 알아보기도 더 좋다고 함.
#### x1 =x 가 아니라 x1 = x.copy() ㄱㄱ 
##

X1 = X.copy()
X1['cap-shape'].unique()  # 이게 뭐였지? 

X1['cap-shape'].map({"x":0, "f":1, "k":2, "b":3, "s":4, "c":5}) 

X2 = X.copy()

# 원핫인코딩을 할 컬럼을 뽑아서 넣어주어야 함
X_one_hot = pd.get_dummies(X2)
X_one_hot.head()

## 훈련과 평가로 데이터 분리

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_one_hot, y, 
                                                   test_size = 0.3)

## 모델 불러오기

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()

# 학습

tree_model.fit(X_train, y_train)

# 평가

tree_model.score(X_train, y_train)

tree_model.score(X_test, y_test)



