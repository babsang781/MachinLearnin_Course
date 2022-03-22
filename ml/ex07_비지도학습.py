# 비지도학습
### 정답이 없는 데이터를 학습해서 패턴, 특징을 파악
### 데이터를 새롭게 표현해서 머신러닝이 원래 데이터보다 쉽게 해석할 수 있도록 만듦
### 
### 차원축소 (Dimension Reduction) : 특성이 너무 많아서 생기는 과대적합에 적합
    #### 특성( 차원)을 줄여서 정보를 축약: 하지만 최대한 많은 정보값을 보존하도록 하는 기술
### 군집(Clustering)
    #### 라벨링에 도움이 됨
    #### 비슷한 데이터들끼리 묶어줌

# 차원축소

!pip install mglearn

import mglearn

### 차원축소가 어덯게 이루어지는지 볼 수 있음
#### 아래는 pca 라고하여 주성분 분석임.Principal Component Analysis
mglearn.plots.plot_pca_illustration()

#### 데이터가 가장 많이 분포되어 잇는 방향이 주성분
#### 2 차원의 경우 두 번째 성분은 주성분의 직각 방향 중 분산이 가장 큰 것을 두 번째로 함.

#### 평균을 모두에 대해 빼서 0으로 맞추고, 주성분만 남김 -> 1차원 변경
#### 다시 2차원으로 되돌림

#### 노이즈 제거 등에 쓰임

### 장단점 : 
    #### 장점: 머신러닝이 판단을 잘 하게 만들어줄 수 잇음
    #### 단점 : 축이 어디 있는지 판단하기 힘듦, 축을 해석하기 힘듬

## iris 데이터

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y= iris.target

X

### PCA 
#### 주성분 분석을 사용해서 4차원 데이터를 2차원으로 축소

from sklearn.decomposition import PCA

#### 전체 데이터를 2차원으로 축소
pca = PCA(n_components = 2)  

#### 학습 : 차원축소를 위해 필요한 정보를 학습하는 단계
pca.fit(X)

X_trans = pca.transform(X)
X_trans

### 축소된 데이터 시각화
plt.figure(figsize=(10,10))
# 산점도
plt.scatter(X_trans[:,0], # 축소된 0번 컬럼
            X_trans[:,1], # 축소된 1번 컬럼
            c = y) 
plt.show()

#### 이상으로 차원축소 배우는 샘플 데이터

### 사람 데이터

from sklearn.datasets import fetch_lfw_people  # 미국 유명 텔런트 사진 데이터

#### 125*94 (세로 * 가로)
#### 87*65
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)  

people.keys()
#### target 이미지 라벨이 숫자로
#### target_names 라벨의 텍스트 이름


people.data.shape 

people.target

people.target_names

plt.imshow(people.images[0], cmap='gray')
plt.show()









