# 문제 정의
### 타이타닉 데이터를 통해 생존자와 사망자를 예측하기
### Kaggle 에서 높은 점수를 목표

# 데이터 수집

import pandas as pd 
train = pd.read_csv('./data/train.csv') 
test = pd.read_csv('./data/test.csv')

train.shape, test.shape

- 분석 feature : Pclass, Age, SibSp, Parch, Fare...
- 예측 target label : Survived

- feature 

<table border=0 align=left width=700>
  <tr><th>feature<th width=200>의미<th width=300>설명<th> 타입
  <tr><td>Survivied<td>생존여부<td>target 라벨 (0 : 사망, 1 : 생존)<td>integer
  <tr><td>Pclass<td>티켓의 클래스<td>1 = 1등석, 2 = 2등석, 3 = 3등석<td>integer 
  <tr><td>Sex<td>성별<td>male, female로 구분<td>string    
  <tr><td>Age<td>나이<td>0-80세<td>integer
  <tr><td>SibSp<td>함께 탑승한 형제와 배우자의 수<td><td>integer
  <tr><td>Parch<td>함께 탑승한 부모, 아이의 수<td><td>integer
  <tr><td>Ticket<td>티켓 번호<td>alphabat + integer<td>integer
  <tr><td>Fare<td>탑승료<td><td>float
  <tr><td>Cabin<td>객실 번호<td>alphabat + integer<td>string
  <tr><td>Embarked<td>탑승 항구<td>C = Cherbourg, Q = Queenstown, S = Southampton<td>string
</table>

# 데이터 전처리
## 정답컬럼 분리

y_train = train['Survived']

## 결측치 확인

train.info()

test.info()

train.head()

## PassengerId 삭제

# 승객 번호는 분석에 의미가 없을 것으로 판단되어서 컬럼을 삭제함.
# axis 0/1 주의! 
train.drop("PassengerId", axis= 1, inplace = True)
test.drop("PassengerId", axis= 1, inplace = True)

test.head(), train.head()

## Embarked 채우기
### 최빈값으로 채우기 : 수를 확인해보니, 결측치가 많지 않고, 압도적인 최빈값이 존재하여 그렇게 판단함

train['Embarked'].value_counts() # data의 수를 확인하기 위해 value_counts() 를 사용함

### .fillna( data 이름, (inplace 여부 ))
###  결측치 na 를 최빈값인'S'로 채워줌 
train['Embarked'].fillna('S', inplace=True)

## test 데이터에 Fare 컬럼 결측치 채우기
test.info()

### Fare 값은 실수단위 데이터이기 때문에 최빈값으로 확인할 수 없고, 
### 중앙값이나 평균값을 이용하여 채워줌
test['Fare'].describe()  # describe() 로 데이터 확인

### std : 표준편차(standard deviation) : 분포 비율 : 클수로 더 이상하게 분포됨.
### max 값이 이상치로 의심되는 값이기 때문에 평균이 아닌 중앙값을 이용하여 채워줌.
test['Fare'].fillna(14.4542, inplace=True)

test.info()

## Age 채우기
### 다른 컬럼과의 상관 관계를 통해서 채우기

### 수치형 컬럼들, 수치형 컬럼들 간의 상관관계를 확인 .corr()
train.corr()  # + : 양의 상관관계, - : 음의 상관관계

### .groupby(by = 'Pclass') : 티켓 등급이 같은 데이터들 끼리 묶음
age_table=train[['Pclass', 'Sex', 'Age']].groupby(by = ['Pclass', 'Sex']).median()

test[['Pclass', 'Sex', 'Age']].groupby(by = ['Pclass', 'Sex']).median()

### age의 결측치를 채워줄 함수 만들기

import numpy as np

### 한 사람의 데이터를 불러와서 테스트를 하는 함수 생성
def fill_age(person):  
    if np.isnan(person['Age']):  # NaN 인지 아닌지 True/ False 구분
        return age_table.loc[ person['Pclass'], person['Sex'] ][0]  # 
    else : 
        return person['Age']

### 한 사람의 데이터만 불러오기
### apply : 행이나 열 별로 데이터를 출력하고 지정해둔 함수에 적용
train['Age'] = train.apply(fill_age, axis = 1)  # (함수, 축 방향) -> 행별로 하나씩 뽑아서 함수에 적용

#### 팁 , 1.'esc' 버튼으로 커맨드 모드 변경  2. 'f' 단축키로 해당 셀 찾아 바꾸기 가능!
test['Age'] = test.apply(fill_age, axis = 1)

#### train 데이터 기준으로 test data 값을 변경시켜준 것은 많은 사람들이 이렇게 해줘서,
#### 이것은 train 데이터의 규칙으로 결과를 찾기 때문에, test 데이터에도 train 데이터의 규칙을 적용한 결측치 처리를 적용한 것임
#### 하지만 이런 방법이 모든 데이터에 적용 되는 것은 아님.

## Cabin 채우기
### 결측치의 비율이 너무 높기 때문에 결측치 자체를 하나의 데이터로 활용
#### 보통 결측치가 너무 많은 것은 컬럼 삭제를 하는 경우가 많지만, 
#### 경우에 따라 결측 자체를 하나의 응답으로 보고 처리할 수도 있음.

train['Cabin'].unique()  # 중복을 제외한 데이터를 표시 

train['Cabin'] = train['Cabin'].str[0]  # str 을 통해서 컬럼을 string 타입으로 변경, [0] 0 번째 인덱스만 가져오기
test['Cabin'] = test['Cabin'].str[0]

train['Cabin'].unique()

train['Cabin'].fillna('N', inplace = True)
test['Cabin'].fillna('N', inplace = True)

# 탐색적 데이터 분석

### 시각화 라이브러리 import
import seaborn as sns

#### seaborn은 코드가 조금 더 단순한 편
#### 옵션이 알아서 달리는 것이 특징, hue 카테고리 변수명을 지정
sns.countplot( data = train, x = 'Cabin', hue = 'Survived')

#### 결측치 N에서 사망한 사람의 비율이 특히 더 높게 나타남
#### 분석에 사용해도 될 것 같다고 판단할 수 있음.

## Pclass 시각화

sns.countplot( data = train, x = 'Pclass', hue = 'Survived')

## Embarked 와 Pclass 를 시각화

sns.countplot( data = train, x = 'Embarked', hue = 'Pclass')

train['Embarked'].value_counts()

## Sex, Age, Survived
sns.violinplot(data = train, x = 'Sex', y = 'Age', hue = 'Survived', split = True)

sns.violinplot(data = train, x = 'Survived', y = 'Age', hue = 'Sex', split = True)

## SibSp, Parch 시각화
### 특성공학 : 컬럼에 연산을 통해서 의미있는 새로운 정보를 추출하는 행위

### SibSp + Parch +1 = Family_size 컬럼 생성
train['Family_size'] = train['SibSp'] + train['Parch'] + 1
test['Family_size'] = test['SibSp'] + test['Parch'] + 1

sns.countplot(data=train, x='Family_size', hue='Survived')

### 다음 시간에는 해당 내용 특징별로 범주화 할 예정, 
