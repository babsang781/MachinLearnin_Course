import pandas as pd
dog_list = ['비숑','포메라니안','셔틀랜드 쉽독','믹스:스피츠+포메']

dog_dic =  pd.DataFrame(['비숑','포메라니안','셔틀랜드 쉽독','믹스:스피츠+포메'],
                        index= range(1,len(dog_list)+1),
                        columns = ['견종'])
display(dog_dic)

case=[['믹스',49,35,25],['비숑',47,35,40],['비숑',45,33,37],['비숑',42,32,35],['비숑',46,35,38],['셔틀랜드 쉽독',64,44,40],['포메라니안',43,34,30]]

# 목둘레 3 cm 추가해서 작성함
data =  pd.DataFrame([['믹스',49,35,25],['비숑',47,35,40],['비숑',45,33,37],['비숑',42,32,35],['비숑',46,35,38],['셔틀랜드 쉽독',64,44,40],['포메라니안',43,34,30]],
                    index = range(1,len(case)+1),
                    columns = ['견종','가슴 둘레','목 둘레','등 길이'])
data

----
# 목 길이 기억으로 테스트 데이터 생성
## linear  / sgd_model 경사하강법

case1 = pd.DataFrame([[11.3, 35],[10.9,33]],index=[1,2], columns =['사진길이','실측'])
case1

# 수학 공식을 이용한 해석적 방법
### Linear Regression 모델 

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

### 문제는 2차원, 정답은 1차원 데이터 [][] 수가 다름
### 시리즈 데이터는 1차원으로 볼 수 있고, 시간부분은 2차원으로 표시해야 에러라 생기지 않음.
linear_model.fit(case1[['사진길이']],case1['실측'])

print(linear_model.coef_)  # 가중치
print(linear_model.intercept_)  # 절편


# 경사하강법 : SGDRegressor

from sklearn.linear_model import SGDRegressor

sgd_model = SGDRegressor(max_iter = 200, # 가중치 업데이트 반복 횟수 :총 5천번까지 학습
                        eta0 = 0.001, # 학습률 (learning rate) 오차를 얼마나 허락할 것인가
                        verbose = 1) # 학습과정 확인

sgd_model.fit(case1[['사진길이']],case1['실측'])   # Epoch 에포크: 중요한 사건·변화들이 일어남. 학습하는 횟수
### loss를 봐야함. 오차가 가장 작은 값으로 찾아가는 과정에서 처음 오차찾기 시작 지점 정도라고 생각하면 됨.

### y = 3.03271575x + 0.27289728
print(sgd_model.coef_)
print(sgd_model.intercept_)

sgd_model.predict([[12]])

### 분류 score: 정확도
### 회귀 score: R2 score
sgd_model.score(case1[['사진길이']],case1['실측'])


