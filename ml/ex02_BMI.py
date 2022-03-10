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

