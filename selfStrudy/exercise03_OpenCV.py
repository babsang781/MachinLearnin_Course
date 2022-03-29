# < OpenCV >
### OpenCV는 Open Source Computer Vision Library를 줄여 쓴 말
### 영상 처리와 컴퓨터 비전 프로그래밍 분야의 가장 대표적인 라이브러리

### 예전에는 매우 복잡한 알고리즘을 C와 C++ 을 이용해 구현해야했으므로 비교적 전문 분야였으나, 
### 이 라이브러리 덕분에 기초 지식만으로 손쉽게 사용 가능해졌음

### OpenCV  https://opencv.org/
### 오픈 cv 는 처음에는 c 언어로 작성되었지만 지금은 C++공식, 바인딩언어**로 자바와 파이썬을 공식 채택,
#### 맥os, 윈도우, 리눅스, 안드로이드, ios 가능

### OpenCV는 소스를 두 개의 저장소에 나누어 관리
#### https://github.com/opencv/opencv
#### https://github.com/opencv/opencv_contribe

    #### main 은 연구와 상업 무관하게 사용 가능. 소스 코드 오픈 의무도 없음
    #### extra 는 시험중인 것, 특허권이 있는 것 등이 섞여 있음

#### **컴퓨터 프로그래밍에서 기본 단위(예- 변수)가 갖는 속성이 확정되어 더 이상 변경할 수 없는 구속(bind) 상태가 되는 것(상태이상?! 확정cc) 
    #### 프로그램 내에서 변수, 배열, 라벨, 절차 등의 명칭, 즉 식별자(identifier)가 
    #### 그 대상인 메모리 주소, 데이터형 또는 실제값으로 확정(배정)되는 것
        
### 변수(variables): 데이터를 저장하는 상자, 값이 변할 수 있는 객체 
    #### 일반적으로 식별자(이름) + 자료의 속성 + 하나 이상의 주소(참조), 자료값 으로 구성 
    # Scanner scanner = new Scanner();
    # int i = 0;        
    
    #### 정적 바인딩[Static binding; 명시적(explicit), 암시적(implicit) ]: 
        ##### 원시 프로그램의 컴파일링 이전 또는 링크 시에 확정되는 바인딩, 이후 변경 X
        #### 패턴이 정확하다. / 다양한 대처가 어렵다.
        
    #### 동적 바인딩(dynamic binding) : 프로그램의 실행되는 과정에서 바인딩, 실행중 Type 이 변경 가능.
        #### 유연하다. / 고비용으로 성능 저하가 생길 수 있고, 타입 에러 발견이 어렵다.
    #### 프로그래밍에서는 바인딩을 가급적 뒤로 미루도록 권고하고 있다고 함. 
    #### 각종 값이 확정되어 더 이상 변경할 수 없어짐으로써 발생할 수 있는 문제가 있는 것 같다.

    
    ### 바인딩 1. 속성이 부여된다. 2. 바인딩된 언어가 있으면 그것으로도 쓸 수 있다.
    
### 참고 출처
#### https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=okkam76&logNo=221347019465
#### [용어] 바인딩( Binding ) https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=ljc8808&logNo=220298473228
####  Binding 은 무었인가 https://hankkuu.tistory.com/17   

### 오픈cv 의 역사
    #### 인텔의 러시아 팀에서 cpu 집약적인 응용 프로그램의 성능을 향상시키기 위한 연구의 일부로 시작
    #### 1999년 IPL : image Process Library 를 기반으로 C 언어로 작성
    #### 다섯번의 베타 출시 이후, 2006년에 정식 버전1.0 을 출시, 이때 파이썬 모듈이 포함
    #### 여러차례 버전 배포 후 2010년에 2.2 버전에서 패키지가 현재의 구조르 갖춤
        ##### core, imgproc, fearures2d, ml, contrib 등
    #### 현재 4.5.5 버전 
    
#### 권장하는 버전은 3.4.1.15
#### pip3 install opencv-contrib-python==3.4.1.15 라고 함
    ##### 이후 버전부터 특허가 생긴 구현 내용을 포함하지 않고 배포하기 때문.
    ##### SIFT : 키포인트 검출- 기술하는 알고리즘
    ##### SURF : SIFT 의 가속화 버전
   
### 참고 출처
### https://throwexception.tistory.com/830?category=873104

# 주피터 노트북 open cv instal
# https://pypi.org/project/opencv-python/
# pip install --upgrade pip
!pip install opencv-python

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
img = cv2.imread('./img/dog.jpg')

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



# 예제2, 흑백 IMREAD_GRAYSCALE 읽기 
# 이미지 파일을 흑백으로 읽기
img_file ='./img/dog.jpg'
img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

#  결과 출력 cv2.imshow(title, img)
cv2.imshow('grayscale', img_gray)

# 키가 입력될 때 까지 대기
cv2.waitKey()
cv2.destroyAllWindows()

# 예제3 이미지 저장하기 imwrite

save_file = './img/dog_gray.jpg'  # 경로 변수 저장
cv2.imwrite(save_file, img_gray)  # cv2.imwrite( file_path, img )

cv2.imwrite('./img/dog_blurr.jpg', blured)

# 카메라 웹캠 제어
## 카메라, 웹캠 프레임 읽기



