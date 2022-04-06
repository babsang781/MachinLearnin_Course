# 학습한 모델 저장히기

# pickle 모듈이  데이터 저장에 도움을 줌 
# 원래 숫자나 글자, db, csv 파일 
# pickle 모듈의 특징 : 원래 타입 그대~로 저장해줌 , 모델도 가능 
# 모든 데이터 타입 그대로 가능, class 도 가능, 학습된 모델도 가능


import pickle

# pickle 을 통해서 저장할 때 사용하는 것
# 확장자는 .pkl
# ( title , 모드: 읽기와 쓰기 둘 중 하나)
# wb -> 쓰기 모드 write binary 
#  as f :  f 라는 이름으로 파일을 open 한 것이고, 안에 하나하나 써주면 됨.

with open('testmodel.pkl', 'wb') as f : 
    pickle.dump(tree_model, f)  #  data 입력 : pickle.dump(data, file)
    

#이렇게 하면, 같은 위치에 작성한 title 대로 pkl 파일이 생김

# 플라스크는 소규모 파이썬 프로젝트에 적합
# 순서는 아래와 같음 

## 1. 모델 학습에 사용한 모든 컬럼을 form 을 통해서 jsp-> flask 로 전달 주로 post 방식 

## 2. 학습했던 모델을 로딩, 받은 데이터를 학습한 모델에 넣고 predict
### 2-1 보통의 경우, 웹에서 사용자가 넘겨준 raw 데이터는 너무 날 것이라 전처리 필수!!

## 3. 예측 결과를 다시 tomcat 서버로 넘겨주면 끝.

### 2-1 전처리 관련 작업
# 전처리 및 분석 작업에 컬럼 이름이 필요
# 입력해줄 컬럼들이 담긴 변수, 예시 샘플(걍 수업 영상 따라친 것) X_train.columns 은 타이타닉 X_train 데이터의.column 한 것.

# 데이터 pkl 파일로 저장하기
with open('titanic_column_name.pkl', 'wb')as f2 :
    pickle.dump(X_train.columns, f2)



# 아이리스 knn 모델 저장하기
from sklearn.datasets import load_iris
iris= load_iris()

import pandas as pd
X = pd.DataFrame(iris['data'], columns = iris['feature_names'])
y = iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state = 6)

from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_neighbors = 9, p=2, weights='distance')

knn_model.fit(X_train, y_train)

with open('knn_model.pkl', 'wb') as f : 
    pickle.dump(knn_model, f)  #  data 입력 : pickle.dump(data, file)
    
with open('iris_colunms.pkl', 'wb')as f2 :
    pickle.dump(iris['feature_names'], f2)



