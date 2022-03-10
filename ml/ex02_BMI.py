# 문제정의
- 비만을 판단하는 모델 만들기
- 분류 문제로 접근 

# 데이터 수집
- csv 파일로 데이터 수집
- 500 명에 대한 성별, 키, 몸무게, Label 이 있음

import pandas as pd 
bmi = pd.read_csv('./data/bmi_500.csv')
bmi.head()

# 데이터 전처리
- 사전 데이터 확인하기 
- info() : 데이터 수, 인덱스 번호, 결측치 여부 확인
- describe() : 이상치 확인 

## 결측치 확인
- info(): 데이터 수, 인덱스 번호, 결측치 여부 확인

bmi.info()

## 이상치 확인 
- describe()
  1. std : 표준 편차 - 데이터의 범위 
      - 클수록 이상치가 있을 가능성을 염두에 둬야 함.
  2. mean 과 50%(중앙값) 을 통해 데이터를 이해하기 
      - 평균이 중앙값 오른쪽, 왼쪽인지에 따라 데이터 편중(?)을 볼 수 있음. 
      - 비슷한 경우 이상치가 없을 가능성이 큰 것
  3. 4분위수( min/25%/50%/75%/max ) 간의 범위 차이를 보기   

bmi.describe()

# 탐색적 데이터 분석 (EDA)
- 시각화를 통해서 원하는 정보를 한 눈에 보기
- 비만도별로 데이터가 잘 나오는지 확인하기

## 비만도 등급 확인
- .unique() : 중복을 제거하고 값을 출력 

bmi['Label'].unique()  # 중복을 제거하고 값을 출력 

## Weak 인 사람들을 그래프로 확인해보기
- boolean indexing 으로 전체 중에서 label 이 weak 인 사람만

# boolean indexing 으로 전체 중에서 Label 이 Weak 인 사람만
bmi[bmi['Label'] == 'Weak']

import matplotlib.pyplot as plt

# 산점 도표를 볼 데이터를 scatter_data 라는 이르믕로 저장
scatter_data = bmi[bmi['Label'] == 'Weak']
plt.scatter( scatter_data['Height'], scatter_data['Weight'], 
            color = 'blue', label= 'Weak')
plt.legend()  # 라벨 출력
plt.show()  

## Overweight 그래프 그리기

scatter_data = bmi[ bmi['Label']=='Overweight']
plt.scatter( scatter_data['Height'], scatter_data['Weight'], 
            color = 'red', label= 'Overweight')
plt.legend()  
plt.show() 

## 전체 그래프 그리기
- 데이터 별로 색깔 줘서 구분하기 위해 각각의 케이스 별로 plt에 그리고 합치기

### 반복되는 부분 함수로 만들기

# 전체 케이스가 반복되기 때문에 각 케이스별로 함수 만들기
def myScatter(label, color):
    scatter_data = bmi[ bmi['Label']==label]
    plt.scatter( scatter_data['Height'], scatter_data['Weight'], 
            color = color, label= label)
    plt.legend()  

plt.figure(figsize = (12,8))
myScatter('Extremely Weak','black')
myScatter('Weak','blue')
myScatter('Normal','green')
myScatter('Overweight','pink')
myScatter('Obesity','purple')
myScatter('Extreme Obesity','red')

plt.show() 

####################################################


## 이상치 확인하기

# 몸무게 78kg ,  키가 153cm - 라벨이 잘못 달린 것, 정상-> 비만으로 변경하기
bmi[bmi['Weight'] == 78]

## 이상치 값 변경
- 직접 뽑아와서 변경하는 방법

#
bmi.iloc[231, 3] = 'Obesity'

# 지금까지 1. 데이터부 의 4 단계 진행함.
# 이제부터 2. 모델부 : 5)모델 선택 및 하이퍼 파라미터 튜닝, 6) 학습  7) 평가 

# 모델 선택 및 하이퍼 파라미터 튜닝
- 문제와 답으로 분리
- trainning훈련 set과 evaluation평가 set으로 분리
- 모델 생성 및 하이퍼 파라미터 튜닝

bmi.head()

# 머신러닝에서의 데이터 학습은 숫자 값만 가능, 
# 분류의 정답인 경우만 문자 형태 라벨이 가능
# 데이터를 숫자로 변환해주는 방법도 있지만, 일단 이번에는 성별 데이터를 삭제해볼 것임

X= bmi.loc[ : , 'Height' : 'Weight']
y= bmi.loc[ : , 'Label']

X.shape , y.shape

## 훈련과 평가로 분리
- 훈련 7 : 평가 3

X_train = X.iloc[:350]
X_test = X.iloc[350:]
y_train = y.iloc[:350]
y_test = y.iloc[350:]

X_train.shape, y_train.shape, X_test.shape,y_test.shape

## 모델 불러오기
-KNN : k( 상수) Nearest Neighbors 
  -지정한 상수 k 의 범위 안에서 문제와 가장 가까운(많은) 답을 도출

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
#knn_model = KNeighborsClassifier() <- 괄호 안에 세부 매개 변수 수정하여 사용
# shift + tab 으로 설명 확인

# hyper parameter tuning
# 세부 매개변수 수정을 위한 것

# 학습

knn_model.fit( X_train, y_train )

# 평가

knn_model.predict( [ [ 174, 68 ] ] )

# test 데이터를 이용한 모델 정확도 확인
knn_model.score(X_test, y_test)

# train 데이터로도 확인 가능 
knn_model.score(X_train, y_train)

# 성별 컬럼을 추가해서 학습하기
- male > 1
- female > 2
- map 함수 사용

bmi['Gender'] = bmi['Gender'].map( {"Male" : 1 , "Female": 2 } )
bmi.head()

## 문제와 정답으로 분리

X = bmi.iloc[ : , : -1]
y = bmi.iloc[ : , -1]

## 훈련과 평가로 분리

X_train = X.iloc[:350]
X_test = X.iloc[350:]
y_train = y.iloc[:350]
y_test = y.iloc[350:]

## 모델 불러오기

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()

## 학습

knn_model.fit( X_train, y_train )

## 평가

# 성별 컬러

# test 데이터를 이용한 모델 정확도 확인
knn_model.score(X_test, y_test)

# train 데이터로도 확인 가능 
knn_model.score(X_train, y_train)
 
