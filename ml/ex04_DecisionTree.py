



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

# 시각화

!pip install graphviz

import os
os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'

from sklearn.tree import export_graphviz
export_graphviz(tree_model, out_file='tree.dot',
               class_names=['p','e'],
               feature_names=X_one_hot.columns,
               impurity=True,
               filled=True)

import graphviz

with open('tree.dot', encoding='UTF8') as f:
    dot_graph = f.read()

display(graphviz.Source(dot_graph))

from subprocess import check_call
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])

# 과대적합 제어

## leaf node : 10 개로 조정
### 1. leaf node 의 수를 10개로 조정한 모델
### 2. 학습
### 3. 평가
### 4. 시각화

from sklearn.tree import DecisionTreeClassifier
tree_model2 = DecisionTreeClassifier(max_leaf_nodes=10)

tree_model2.fit(X_train, y_train)

tree_model2.score(X_train, y_train)

tree_model2.score(X_test, y_test)

from sklearn.tree import export_graphviz
export_graphviz(tree_model2, out_file='tree2.dot',
               class_names=['p','e'],
               feature_names=X_one_hot.columns,
               impurity=True,
               filled=True)

import graphviz
with open('tree2.dot', encoding='UTF8') as f:dot_graph = f.read()
display(graphviz.Source(dot_graph))

### png로 바꾸는 코드
from subprocess import check_call
check_call(['dot','-Tpng','tree2.dot','-o','tree2.png'])


# 교차검증

from sklearn.model_selection import cross_val_score
### 네 가지가 필요
### 사용할 모델, 문제, 정답,데이터 분할 수(cv)
cross_val_score(tree_model, X_train, y_train, cv=5).mean()
### 앞으로 데이터 결과 값은 이전에 낸 스코어 보다 이 값을 더 신뢰하는 게 좋음

# 특성 선택

### 특성의 중요도를 볼 수 있는 것
### 117개 컬럼 각 중요도의 합은 1
fi = tree_model.feature_importances_

fi_df = pd.DataFrame(fi,index = X_train.columns)
fi_df.sort_values( by = 0, ascending = False).head(10)

### 추후 분석 권장 방식: (하이퍼 파라미터 수정, fit, 교차검증 값)의 반복
tree_model = DecisionTreeClassifier(max_leaf_nodes=10)

tree_model.fit(X_train, y_train)

cross_val_score(tree_model, X_train, y_train, cv=5).mean()

tree_model.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth' : range(1,10),
    'max_leaf_nodes' : range(10,20)
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5)
grid.fit(X_train, y_train)



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

# 시각화

!pip install graphviz

import os
os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'

from sklearn.tree import export_graphviz
export_graphviz(tree_model, out_file='tree.dot',
               class_names=['p','e'],
               feature_names=X_one_hot.columns,
               impurity=True,
               filled=True)

import graphviz

with open('tree.dot', encoding='UTF8') as f:
    dot_graph = f.read()

display(graphviz.Source(dot_graph))

from subprocess import check_call
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])

# 과대적합 제어

## leaf node : 10 개로 조정
### 1. leaf node 의 수를 10개로 조정한 모델
### 2. 학습
### 3. 평가
### 4. 시각화

from sklearn.tree import DecisionTreeClassifier
tree_model2 = DecisionTreeClassifier(max_leaf_nodes=10)

tree_model2.fit(X_train, y_train)

tree_model2.score(X_train, y_train)

tree_model2.score(X_test, y_test)

from sklearn.tree import export_graphviz
export_graphviz(tree_model2, out_file='tree2.dot',
               class_names=['p','e'],
               feature_names=X_one_hot.columns,
               impurity=True,
               filled=True)

import graphviz
with open('tree2.dot', encoding='UTF8') as f:dot_graph = f.read()
display(graphviz.Source(dot_graph))

### png로 바꾸는 코드
from subprocess import check_call
check_call(['dot','-Tpng','tree2.dot','-o','tree2.png'])


# 교차검증

from sklearn.model_selection import cross_val_score
### 네 가지가 필요
### 사용할 모델, 문제, 정답,데이터 분할 수(cv)
cross_val_score(tree_model, X_train, y_train, cv=5).mean()
### 앞으로 데이터 결과 값은 이전에 낸 스코어 보다 이 값을 더 신뢰하는 게 좋음

# 특성 선택

### 특성의 중요도를 볼 수 있는 것
### 117개 컬럼 각 중요도의 합은 1
fi = tree_model.feature_importances_

fi_df = pd.DataFrame(fi,index = X_train.columns)
fi_df.sort_values( by = 0, ascending = False).head(10)

### 추후 분석 권장 방식: (하이퍼 파라미터 수정, fit, 교차검증 값)의 반복
tree_model = DecisionTreeClassifier(max_leaf_nodes=10)

tree_model.fit(X_train, y_train)

cross_val_score(tree_model, X_train, y_train, cv=5).mean()

tree_model.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth' : range(1,10),
    'max_leaf_nodes' : range(10,20)
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5)
grid.fit(X_train, y_train)
