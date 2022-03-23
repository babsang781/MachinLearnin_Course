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
#### 87*65 = 5655 픽셀수 
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)  

people.keys()
#### target 이미지 라벨이 숫자로
#### target_names 라벨의 텍스트 이름

people.data.shape 

people.target

people.target_names

plt.imshow(people.images[0], cmap='gray')
plt.show()

X = people['data']
y = people['target']

### X 안에 하나의 데이터가 너무 크기 때문에 값을 범위를 줄여줌
X[0]

### 픽셀 하나는 가장 큰 것이 255이기 떄문에 0-1로 표준화
X = X/255

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

### knn모델 학습
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =3)
knn.fit(X_train, y_train)

knn.score(X_train, y_train)

knn.score(X_test, y_test)

### 머신러닝 이미지는 효율이 좋지 않음 .
### knn이 그나마 머신러닝 중에서 좋은데도 이정도 test 값만 나옴

### 차원 축소에서 상위 100개와 whiten 이라는 속성을 추가 
pca = PCA(n_components = 100, whiten = True)

### pca 새로 학습
pca.fit(X_train)

### X_train 데이터와 test 데이터에도 입력
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn.fit(X_train_pca, y_train)

knn.score(X_train_pca, y_train)

knn.score(X_test_pca, y_test)

## 주성분 데이터 확인
pca.components_[0]
### 이미지데이터와 같은 크기를 가지고 있지만
### 차원축소를 했기 떄문에 실제 데이터와 같게 나타나지는 않음

### 100개가 주성분으로 있음
plt.imshow(pca.components_[0].reshape(87, 65))
plt.show()

plt.imshow(pca.components_[99].reshape(87, 65))
plt.show()

## 주성분으로 이미지 확인하기
mglearn.plots.plot_pca_faces(X_train, X_test , (87,65))

## 비음수행렬 분해( NMF = NON- NEGATIVE MATRIX FACTORIZATION)
### PCA와 방법은 같고, 
### 사용해본는 것도 좋을 것

# 군집
### 그룹화되지 않은데이터를 그룹화



#### 대표적인 그룹화 방법: kmeans 실행 순서
#### 1. 그룹으로 나눌 수를 정함.
#### 2. 초기화, 랜덤 포인트 지정 빨 초 파 
#### 3. 포인트를 기준으로 가까운 데이터를 할당(1), 군집화
#### 4. 할당된 케이스의 중심 위치로 recompute(중심 포인트 다시 계산)(1)
#### 5. 재할당 군집화(2)
#### 6. 할당된 케이스의 중심 위치로 recompute(중심 포인트 다시 계산)(2)
#### 7. 재할당 군집화(3...)
#### 8. 할당된 케이스의 중심 위치로 recompute(중심 포인트 다시 계산)(3...)

#### 중심이 바뀌지 않을 때까지 반복.
mglearn.plots.plot_kmeans_algorithm()

from sklearn.datasets import make_blobs   ##2차원 데이터를 생성할 수 있음
from sklearn.cluster import KMeans

### 인위적으로 2차원 데이터를 생성 X : data / y: label
X, y = make_blobs(random_state=2)

### 군집 모델 생성
kmeans = KMeans(n_clusters =3 )
kmeans.fit(X)

X.shape

### kmeans 로 구분한 내용
### y는 label된 값이 있음
pred = kmeans.predict(X)  

pred

## 중심포인트의 좌표
kmeans.cluster_centers_

for n_cluster in range(3):
    data = X[y==n_cluster]
    plt.scatter(data[:, 0], data[: , 1], label = n_cluster)
    
plt.legend()
plt.show()

center = kmeans.cluster_centers_  ### 중심좌표를 center 로 저장
for n_cluster in range(3):
    data = X[pred==n_cluster]
    plt.scatter(data[:, 0], data[: , 1], label = n_cluster)
    plt.plot(center[n_cluster , 0] , center[n_cluster , 1] , marker = '^' , c = 'black')
    # plt.plot 또는 scatter 를 통해서 center를 찍어줌
    
plt.legend()
plt.show()

### 데이터는 구했는데 라벨을 하지 못한 경우, 참고용으로 사용 가능
### 단점: 군집의 중요도가 모든 방향에 동일하게 적용됨.
### 데이터가 특정한 모양을 가지고 있는 경우, 그룹을 잘 만들지 못함.


## 반달 데이터 셋

from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 200, random_state = 0 , noise = 0.05 )
### 반달 데이터 셋 만들기

for n_cluster in range(2):
    data = X[y==n_cluster]
    plt.scatter(data[:, 0], data[: , 1], label = n_cluster)
    
plt.legend()
plt.show()

kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)
pred = kmeans.predict(X)

### 모든 방향을 동일하게 보는 kmeans 가 제대로 나오지 않는 것을 확인해보기 
center = kmeans.cluster_centers_  ### 중심좌표를 center 로 저장
for n_cluster in range(2):
    data = X[pred==n_cluster]
    
    plt.scatter(data[:, 0], data[: , 1], label = n_cluster)
    plt.plot(center[n_cluster , 0] , center[n_cluster , 1] , marker = '^' , c = 'black')
    # plt.plot 또는 scatter 를 통해서 center를 찍어줌
    
plt.legend()
plt.show()

## 관련해서 책으로 이미지 군집 확인하기!!!

## 병합 군집

### 기본적인 메커니즘: 가까이 있는 것을 그룹화하기
### kmeans 보다 조금 더 나은 결과 왜?
#### 1. 가장 가까운 데이터를 잡고, 이 둘은 하나처럼 간주
#### 2. 그 다음 가장 가까운 데이터를 하나씩 하나의 데이터로 묶음 
#### 
mglearn.plots.plot_agglomerative_algorithm()

from sklearn.cluster import AgglomerativeClustering
agg= AgglomerativeClustering(n_clusters = 2)
pred = agg.fit_predict(X)

### 학습하는 데이터로 예측하여, 새로운데이터에 대한 예측 불가능
### fit_predict 만 있고, predict 메서드가 없음

for n_cluster in range(2):
  data = X[pred == n_cluster]
  plt.scatter(data[: , 0] , data[: , 1] , label = n_cluster)

plt.legend()
plt.show()

# DBSCAN 군집 알고리즘
### 밀도기반 클러스터링 ( DBSCAN: Density Based Spatial Clustering of Applications with noise)

### 클러스터의 개수를 미리 정하지 않지만 
### 복잡한 형상도 찾을 수 있으며 어떤 클래스에도 속하지 않는 포인트를 구분할 수 있음

### 두 매개 변수 eps : 거리 변수 / min_samples 가 있음
#### eps 거리 안에 데이터가 min_samples 개수만큼 있으면(본인 포함) 핵심 샘플로 분류

#### 데이터를 표시하는 세 가지 방법


mglearn.plots.plot_dbscan()

#### 큰 포인트는 core point 
#### 작은 포인트는 border point 
#### 무채색은 moise point 로 분류 됨.

### DBSCAN을 제대로 사용하기 위해서는 scaling 을 해주는 것이 좋음.
#### 값이 큰 것들이 과도한 영향을 주는 것을 방지하기 위함

#### scaler 를 통해 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정.
from sklearn.preprocessing import StandardScaler
st_scaler = StandardScaler()
st_scaler.fit(X)
X_scaler = st_scaler.transform(X)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN( eps = 0.3 , min_samples = 5)
pred = dbscan.fit_predict(X_scaler)

pred

# 데이터들이 모여있는 방향을 알 수 있음
for n_cluster in range(2):
  data = X[pred == n_cluster]
  plt.scatter(data[: , 0] , data[: , 1] , label = n_cluster)

plt.legend()
plt.show()

