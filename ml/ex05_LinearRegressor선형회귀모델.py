import pandas as pd

data =  pd.DataFrame([[2,20],[4,40],[8,80],[9,90]],
                    index = ['해도','병관','명훈','동원'],
                    columns = ['시간','성적'])

data

# 수학 공식을 이용한 해석적 방법
### Linear Regression 모델 

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

### 문제는 2차원, 정답은 1차원 데이터 [][] 수가 다름
### 시리즈 데이터는 1차원으로 볼 수 있고, 시간부분은 2차원으로 표시해야 에러라 생기지 않음.
linear_model.fit(data[['시간']],data['성적'])

print(linear_model.coef_)  # 가중치
print(linear_model.intercept_)  # 절편

# 경사하강법
### SGDRegressor

from sklearn.linear_model import SGDRegressor

sgd_model = SGDRegressor(max_iter = 5000, # 가중치 업데이트 반복 횟수 :총 5천번까지 학습
                        eta0 = 0.01, # 학습률 (learning rate) 오차를 얼마나 허락할 것인가
                        verbose = 1) # 학습과정 확인

sgd_model.fit(data[['시간']], data['성적'])   # Epoch 에포크: 중요한 사건·변화들이 일어남. 학습하는 횟수
### loss를 봐야함. 오차가 가장 작은 값으로 찾아가는 과정에서 처음 오차찾기 시작 지점 정도라고 생각하면 됨.

### y = 9.8485x + 1.1195
print(sgd_model.coef_)
print(sgd_model.intercept_)

sgd_model.predict([[7]])

### 분류 score: 정확도
### 회귀 score: R2 score
sgd_model.score(data[['시간']], data['성적'])





