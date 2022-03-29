# 영상필터
#### 필터는 원하는 값만 걸러내려고 할 때 사용하는 것
#### 흐릿 또는 또렷하게 만들기도 하고, edge를 검출하고 edge 의 방향을 알아내는 등 
#### 객체 인식과 분리의 기본이 되는 정보를 계산하기도 함.
### 공간영역(spacial domain) 필터 / 주파수 영역 ( frequency domain )

## 컨볼루션과 블러링
#### convolution 연산은 공간 영역 필터의 핵심, 블러링을 사례로 컨볼루션 연산을 확인 가능
#### n x n 크기의 커널로 주변 요소의 값에 따라 입력값을 영향을 주는 것으로 블러링 필터 작용
#### 모든 칸에 반복

### cv2.dilter2D() 함수 사용

### 예제, 평균 필터를 생성해서 블러 적용
import cv2
import numpy as np
img = cv2.imread('../img/girl.jpg')

### 5 x 5 평균필터 커널 생성
kernel = np.array( [[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04]])

### 5 x 5 평균필터 커널 생성(2)
kernel = np.ones(( 5, 5 ))/ 5**2

### 필터 적용,
blured = cv2.filter2D(img, -1, kernel)

###  결과 출력
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)

### 키가 입력될 때 까지 대기
cv2.waitKey()
cv2.destroyAllWindows()

