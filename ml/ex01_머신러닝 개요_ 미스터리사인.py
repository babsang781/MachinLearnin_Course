# 미스터리 사인
 - 주어진 수를 보고 규칙을 찾는 문제

## Question 1 
- 큰수 + 두 수의 차
- 22 ? 7 = 37

### 데이터 만들기

list1 = []
for i in range(1, 200):  # 첫 번째 숫자의 범위
    for j in range(1,200):  # 두 번째 숫자의 범위
        result = (i + (i-j)) if i > j else( j +(j-i))
        list1.append([i, j, result])

list1

### 데이터 전처리

import pandas as pd
data1 = pd.DataFrame(list1, columns = ['N1', 'N2', 'Result'])
data1.head()

### 상기 내용은 문제(n1, n2) 와 답(result)이 같이 전달되고 있는 것
- 이것을 분리해줘야 함
- x : 문제 / y : 답

x = data1.loc[ :, ['N1', 'N2']]
y = data1.loc[ :, 'Result']

### 머신러닝 모델 불러오기
- 사이킷런(sklearn)의 
- RandomForestRegressor 를 import 해서 불러옴


from sklearn.ensemble import RandomForestRegressor
# 안 될 시, !pip install sklearn 하면 될 것임.

rf = RandomForestRegressor()

### 모델 사용하기(학습)

rf.fit(x, y)

### 모델 사용하기(예측)

rf.predict([[2000,1900]])

rf.predict([[200,190]])

## Question 2 
- (큰 수 / 작은수) 의 나머지
- 4 ? 19 = 3

### 데이터 만들기

list2 = []
for i in range(100, 500):  # 첫 번째 숫자의 범위
    for j in range(1, 500):  # 두 번째 숫자의 범위
        result = ( i % j ) if i > j else(j % i)
        list2.append([i, j, result])

list2

### 데이터 전처리

data2 = pd.DataFrame(list2, columns = ['N1', 'N2', 'Result'])
data2.head()

### 문제와 정답 분리

x2 = data2.loc[ :, ['N1', 'N2']]
y2 = data2.loc[ :, 'Result']

### 모델 불러오기

rf2 = RandomForestRegressor()

### 모델 사용하기(학습)

rf2.fit(x2, y2)

### 모델 사용하기(예측)

rf2.predict([[501,497]])











