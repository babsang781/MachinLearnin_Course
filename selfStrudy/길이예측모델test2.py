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

neck_round = [32,31,29]
neck_pic = [13.5,12.4,10.6]
dic = {'목 실측': neck_round, '목 사진길이': neck_pic}
neck_data = pd.DataFrame(dic)
neck_data

back_len = [40, 38]
back_pic = [41.3, 38]
dic = {'등 실측': back_len, '등 사진길이': back_pic}
back_data = pd.DataFrame(dic)
back_data

chest_round = [47,45,43,42,46]
chest_pic = [16.2,14.1,15.6,15.1,17]
dic2 = {'가슴 실측': chest_round, '가슴 사진길이': chest_pic}
chest_data = pd.DataFrame(dic2)
chest_data

# 수학 공식을 이용한 해석적 방법
### Linear Regression 모델 

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model2 = LinearRegression()
linear_model3 = LinearRegression()

### 문제는 2차원, 정답은 1차원 데이터 [][] 수가 다름
### 시리즈 데이터는 1차원으로 볼 수 있고, 시간부분은 2차원으로 표시해야 에러라 생기지 않음.
linear_model.fit(neck_data[['목 사진길이']],neck_data['목 실측']) 
linear_model2.fit(back_data[['등 사진길이']],back_data['등 실측']) 
linear_model3.fit(chest_data[['가슴 사진길이']],chest_data['가슴 실측']) 

print(linear_model.coef_)  # 가중치
print(linear_model.intercept_)  # 절편
print(linear_model2.coef_)  # 가중치
print(linear_model2.intercept_)  # 절편
print(linear_model3.coef_)  # 가중치
print(linear_model3.intercept_)  # 절편


# 경사하강법 : SGDRegressor

from sklearn.linear_model import SGDRegressor

sgd_model_neck = SGDRegressor(max_iter = 300, # 가중치 업데이트 반복 횟수 :총 5천번까지 학습
                        eta0 = 0.001, # 학습률 (learning rate) 오차를 얼마나 허락할 것인가
                        verbose = 1) # 학습과정 확인
sgd_model_back = SGDRegressor(max_iter = 300, eta0 = 0.001, verbose = 1) 
sgd_model_chest = SGDRegressor(max_iter = 300, eta0 = 0.001, verbose = 1) 


sgd_model_neck.fit(neck_data[['목 사진길이']],neck_data['목 실측'])   # Epoch 에포크: 중요한 사건·변화들이 일어남. 학습하는 횟수
### loss를 봐야함. 오차가 가장 작은 값으로 찾아가는 과정에서 처음 오차찾기 시작 지점 정도라고 생각하면 됨.
print('''

---------------------------------------

''')
sgd_model_back.fit(back_data[['등 사진길이']],back_data['등 실측']) 
print('''

---------------------------------------

''')
sgd_model_chest.fit(chest_data[['가슴 사진길이']],chest_data['가슴 실측']) 


### y = 3.03271575x + 0.27289728
print('목',end=' ')
print(sgd_model_neck.coef_,end=' ')
print(sgd_model_neck.intercept_)
print('가슴',end=' ')
print(sgd_model_chest.coef_,end=' ')
print(sgd_model_chest.intercept_)
print('등',end=' ')
print(sgd_model_back.coef_,end=' ')
print(sgd_model_back.intercept_)


### 분류 score: 정확도
### 회귀 score: R2 score
display(sgd_model_neck.score(neck_data[['목 사진길이']],neck_data['목 실측']))
display(sgd_model_chest.score(chest_data[['가슴 사진길이']],chest_data['가슴 실측']))
sgd_model_back.score(back_data[['등 사진길이']],back_data['등 실측'])




sgd_model_neck.predict([[13.5]]), sgd_model_back.predict([[30]]), sgd_model_chest.predict([[16]])

