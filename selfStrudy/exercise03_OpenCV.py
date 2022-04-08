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

# 이미지 영상 출력



## 예제1, file 읽기
### cv2 가 읽고 read - 보여줌 show ! 

import cv2
import numpy as np

img_file ='./img/dog.pg'   # 파일을 표시할 경로
img= cv2.imread(img_file)    # 이미지 경로(img_file)를 cv2.imread() 로 읽어서 img 에 할당!

if img is not None:                 # 경로에 문제가 있을 경우 떄문에 if 조건문으로 잡아줌
    cv2.imshow('title_ex01', img)    # 읽은 이미지를 화면에 표시
    cv2.waitKey()                    # 키가 입력될 때까지 대기 [default : 0 -> 무한대]
    cv2.destroyAllWindows()          # 창 모두 닫기 
    
else:
    print('No image file.')



## 흑백 IMREAD_GRAYSCALE 읽기 
### cv2.imread로 파일을 읽어서 담아주는데, 읽기 모드 바꾸기! 
### cv2.imread(경로, 읽기모드)

import cv2
import numpy as np

img_file ='./img/dog.jpg'
img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) 

#  결과 출력 cv2.imshow(title, img)
cv2.imshow('grayscale', img_gray)

# 키가 입력될 때 까지 대기 millisecond 단위, 2초
cv2.waitKey(2000)
cv2.destroyAllWindows()

## cv2.destroyAllWindows() 작업을 해주지 않으면 찐한 회색으로 이미지가 나오면서 제대로 실행되지 않음.

#### 읽기 모드의 종류
cv2.IMREAD_UNCHANGED : 이미지 그대로 출력(원본)
cv2.IMREAD_GRAYSCALE : 1 채널, 회색조 이미지로 변환
cv2.IMREAD_COLOR : 3채널, BGR 이미지로 변환
cv2.IMREAD_ANYDEPTH : 이미지에 따라 16,32bit 또는 8비트로 변환
cv2.IMREAD_ANYCOLOR : 이미지 모든 색상 형식으로 읽기
cv2.IMREAD_REDUCED_GRAYSCALE_2 : GRAYSCALE + 이미지 크기 1/2
cv2.IMREAD_REDUCED_COLOR_2 : COLOR + 이미지 크기 1/2
cv2.IMREAD_REDUCED_GRAYSCALE_4 : GRAYSCALE + 이미지 크기 1/4
cv2.IMREAD_REDUCED_COLOR_4 : COLOR + 이미지 크기 1/4
cv2.IMREAD_REDUCED_GRAYSCALE_8 : GRAYSCALE + 이미지 크기 1/8
cv2.IMREAD_REDUCED_COLOR_8 : COLOR + 이미지 크기 1/8
배경이 투명인 이미지(채널이 4개)를 불러올 경우엔 UNCHANGED를 사용(나중에 따로 다루겠습니다.) 

## 예제3 이미지 저장하기!
### cv2 가 써서 저장해줌: imwrite ! 

save_file = './img/dog_gray.jpg'  # 경로 변수 저장
cv2.imwrite(save_file, img_gray)  # cv2.imwrite( file_path, file 변수 )

# 예제4 동영상 및 카메라 프레임 읽기
## 동영상 파일이나 연결한 카메라로부터 연속된 이미지 프레임을 읽을 수 있는 API를 제공해줌

## cv2.VideoCapture( 경로 file_path 또는 index) : 비디오 캡쳐 객체 생성자 
### index 는 연결된 카메라 장치번호 인덱스를 말함)
### 이를 사용하는 객체로 주로 cap을 씀. 
#### cap = cv2.VideoCapture( index )

## 상기 객체 초기화 여부 확인 isOpened() : 객체 초기화 확인 True / False 반환
### ret = cap.isOpened()

## cap 객체. 영상 프레임 읽기 .read() , img 는 프레임 이미지, numpy 배열, None
### img = cap.read() / ret = cap.read()

## cap.set (id, value) : 프로퍼티 변경
### cap.get(id) : 프로퍼티 확인
### cap.release() : 캡쳐차원 반납



# 캠 연결 코드 

# VideoCapture(0) 함수로 0번 카메라 장치와 연결 - 비디오 객체 cap을 생성
cap = cv2.VideoCapture(0) 

if cap.isOpened():   # 연결된 캠이 있으면, 
    while True:
        
        # ret 객체 초기화 여부 True / False
        ret, img = cap.read()    # 카메라 프레임 읽기
        if ret:
            cv2.imshow('camera', img)    # 프레임 이미지 표시 imshow(title, 읽은 프레임)
            if cv2.waitKey(1) != -1:    # 아무키나  입력하면 꺼짐  / if c == 27: break   ESC 로 끄기
                 break
        else:
            print('no frame')
            break
else:             # 연결된 캠이 없으면,  
    print("can't open camera.")  
    
cap.release()
cv2.destroyAllWindows()
            
            

 cap = cv2.VideoCapture(0)  # VideoCapture(0) 함수로 0번 카메라 장치와 연결 - 비디오 객체 cap을 생성
if cap.isOpened():

    while True:
        # ret 객체 초기화 여부 True / False
        ret, frame = cap.read()   # 카메라 프레임 읽기 : img 에 비디오 객체 cap을 읽어서 read() 넣음 =
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)

        cv2.imshow('camera', frame)
        c = cv2.waitKey(1)  # if c == 27: break   ESC 로 끄기
        if c == 27:
            break

else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()

# 카메라로 사진 찍기

cap = cv2.VideoCapture(0) 

if cap.isOpened():
    while True

























# 영상필터
#### 필터는 원하는 값만 걸러내려고 할 때 사용하는 것
#### 흐릿 또는 또렷하게 만들기도 하고, edge를 검출하고 edge 의 방향을 알아내는 등 
#### 객체 인식과 분리의 기본이 되는 정보를 계산하기도 함.
### 공간영역(spacial domain) 필터 / 주파수 영역 ( frequency domain )



## 비디오 재생하기  cv2.imshow
### 프레임 폭 값 읽기 width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
### 프레임 폭 수정하기  img = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)  
### 초당 프레임 수 읽기  fps = cap.get(cv2.CAP_PROP_FPS)

import cv2 

video = "./img/dog.mp4"    #비디오 경로 저장

cap = cv2.VideoCapture(video)    # 비디오 객체 cap(capture) 생성, cv2.VideoCapture ( 경로 )

# 프레임 폭 값 확인하기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'original size: {width}, {height}')

# 프레임 폭 값 수정하기 ...???
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)    # 변경할 값으로 set
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)   # 변경할 값으로 set
#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#print(f'revised size: {width}, {height}')
## 왜 프레임 값 수정이 안 되냥?? 읽기는 되는데 설정이 안 되네

if cap.isOpened():               # 캡쳐 객체 초기화(연결) 확인
    fps = cap.get(cv2.CAP_PROP_FPS)    # 프레임 수 구하기 
    delay = int(1000/fps)
    print("FPS: %f, Delay: %dms" %(fps, delay))
    
    while True:
        ret, img = cap.read()    # 프레임 읽기 cap.read()
        
        # 화면 프레임 크기 수정  ## 비율중심 -상대 변경INTER_AREA : ORIGINAL의 50% 사이즈로 수정
        img = cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)   

        if ret:
            cv2.imshow(video, img)    # 화면에 표시
            cv2.waitKey(delay)
            
            c = cv2.waitKey(1)  # if c == 27: break   ESC 로 끄기
            if c == 27:
                break
        else:
            break
else:
    print("can't open video.")
cap.release()                    # cap 객체 반환
cv2.destroyAllWindows()


## 캠으로 사진 찍기
### 특정 키 눌리면 photo.jpg 로 저장! : imwrite

cap = cv2.VideoCapture(0)  # VideoCapture(0) 함수로 0번 카메라 장치와 연결 - 비디오 객체 cap을 생성
if cap.isOpened():

    while True:
        # ret 객체 초기화 여부 True / False
        ret, frame = cap.read()   # 카메라 프레임 읽기 : img 에 비디오 객체 cap을 읽어서 read() 넣음 =
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
        
        # 캠 화면 보여주기 imshow !
        cv2.imshow('camera', frame)
        
        # 제어하기 1. 현재 이미지 저장: 사진찍기!  2. 종료하기! 
        c = cv2.waitKey(1)
        if c == 46 :           # 아스키 코드 c == 46: 저장  ' .' 마침표로 이미지 저장
            cv2.imwrite('photo.jpg', frame)
            
        if c == 27:       # 아스키 코드 if c == 27: break   ESC 로 끄기
            break

            
else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()

## 캠으로 녹화하기: VideoWriter 객체
### 초당 프레임: fps // 인코딩 포맷: fourcc = cv2.VideoWriter_fourcc(*'DIVX') // 사이즈

import cv2 

cap = cv2.VideoCapture(0)    # 비디오 객체 cap 생성, cv2.VideoCapture ( 경로 )
if cap.isOpened():
    
    # VideoWriter( 경로, 인코딩, fps, size)
    file_path = "./img/cam_video.mp4"           # 1.비디오 경로 저장
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')    # 2. 인코딩 포맷
    fps = 25.40                                 # 3. fps 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))            # 4. size 지정
    out = cv2.VideoWriter(file_path, fourcc, fps, size )  # ** VideoWriter 객체 생성! 
    
    # frame 저장 촬영은 역시 반복문으로! 
    while True:
        ret, frame = cap.read()
        
        if ret:
            cv2.imshow('camera-recording', frame)
            out.wirte(frame)
            if cv2.waitKey(int(1000/fps)1) != -1 :
                break
        else:
            print("no frame")
            break
    
           # c = cv2.waitKey(1)
           #     if c == 27:       # 아스키 코드 if c == 27: break   ESC 로 끄기
           #         break
    out.release()  # VideoWriter 객체 out 종료 
    
else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()





















# 스켈레톤 이전 기본 시도 -> 실패



import cv2
import numpy as np

# 이미지를 읽어서 바이너리 스케일로 변환
img = cv2.imread('./img/silhouette1.png', cv2.IMREAD_GRAYSCALE)
_, biimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)
dst = img.astype(np.float32)
dst_norm = (dst-dst.min())*(255) / (dst.max()-dst.min())
skeleton = cv2.adaptiveThreshold(dst_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)


cv2.imshow('origin', img)
cv2.imshow('dist', dst_norm)
cv2.imshow('skeleton', skeleton)

cv2.waitKey(3000)
cv2.destroyAllWindows()

import cv2
import numpy as np

# 이미지를 읽어서 바이너리 스케일로 변환
img = cv2.imread('./img/silhouette1.png', cv2.IMREAD_GRAYSCALE)
_, biimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 거리변환 --(1)
dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)

# 거리 값을 0-255 범위로 정규화 --(2)
dst = img.astype(np.float32)
dst_norm = (dst-dst.min())*(255) / (dst.max()-dst.min())

# 거리 값에 스레시홀드로 완전한 뼈대 찾기 --(3)
skeleton = cv2.adaptiveThreshold(dst_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 7, -3)

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('dist', dst_norm)
cv2.imshow('skeleton', skeleton)
cv2.waitKey(3000)
cv2.destroyAllWindows()



## 컨볼루션과 블러링 
### cv2.dilter2D() 함수 사용
### kernel = np.array
### blured3 = cv2.filter2D(img, -1, kernel)
#### convolution 연산은 공간 영역 필터의 핵심, 블러링을 사례로 컨볼루션 연산을 확인 가능n x n 크기의 커널로 주변 요소의 값에 따라 입력값을 영향을 주는 것으로 블러링 필터 작용모든 칸에 반복 => n이 클수록 더 흐릿해짐 

### 예제, 평균 필터를 생성해서 블러 적용
import cv2
import numpy as np

img = cv2.imread('./img/dog.jpg')

### 5 x 5 평균필터 커널 생성  25개 셀이라서 0.04를 해주면 합쳐서 1 => 평균 필터
kernel = np.array( [[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04]])

### 5 x 5 평균필터 커널 생성(2)
kernel = np.ones(( 5, 5 ))/ 5**2

### 필터 적용,
blured = cv2.filter2D(img, -1, kernel)


### 3 x 3 평균필터 커널 생성, 필터 적용 
kernel = np.array( [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
kernel = np.ones(( 3, 3 ))/ 3**2
blured2 = cv2.filter2D(img, -1, kernel)

### 10 x 10 가로 평균필터 커널 생성, 필터 적용 
kernel = np.array( [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]])
kernel = np.ones(( 10, 10))/ 10**2
blured3 = cv2.filter2D(img, -1, kernel)


###  결과 출력 
cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)
cv2.imshow('avrg blur2', blured2)
cv2.imshow('avrg blur3', blured3)


### 키가 입력될 때 까지 대기
cv2.waitKey()
cv2.destroyAllWindows()



### 여러 이미지 동시 출력 matplotlib.pyplot
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./img/dog.jpg')

### 5 x 5 평균필터 커널 생성  25개 셀이라서 0.04를 해주면 합쳐서 1 => 평균 필터
kernel = np.array( [[0.04, 0.04, 0.04, 0.04, 0.04],[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04]])
kernel = np.ones(( 5, 5 ))/ 5**2
blured = cv2.filter2D(img, -1, kernel)

### 10 x 10 가로 평균필터 커널 생성, 필터 적용 
kernel = np.array( [[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                    [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]])
kernel = np.ones(( 10, 10))/ 10**2
blured3 = cv2.filter2D(img, -1, kernel)

###  matplotlib.pyplot : 결과 이미지 여러개 하나의 출력 페이지로 출력  

#### 각 이미지 읽기 , img 원본은 맨 처음 읽었음 : 총 4장 
img5x = blured
img10x = blured3 

### 원본과 blurr 종류별로 4장을 가로로 나열함
plt.subplot(1,3,1)              # 1행 4열 중 (1)첫 번째, 
plt.imshow(img[:,:,::-1])       # 컬러 채널 순서 변경
plt.xticks([]);plt.yticks([])   # plt 기본 좌표값을 없애주기 위해
plt.title('original')           # 이미지 타이틀 

plt.subplot(1,3,2)
plt.imshow(img5x[:,:,::-1])
plt.xticks([]);plt.yticks([])
plt.title('blurr5')

plt.subplot(1,3,3)
plt.imshow(img10x[:,:,::-1])
plt.xticks([]);plt.yticks([])
plt.title('blurr10')


### 키가 입력될 때 까지 대기
cv2.waitKey()
cv2.destroyAllWindows()



# 히스토그램 정규화 (histo_normalize.py)*

import cv2
import numpy as np
import matplotlib.pylab as plt

#--① 그레이 스케일로 영상 읽기*

img = cv2.imread('./img/dog.jpg', cv2.IMREAD_GRAYSCALE)

#--② 직접 연산한 정규화*

img_f = img.astype(np.float32)
img_norm = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
img_norm = img_norm.astype(np.uint8)

#--③ OpenCV API를 이용한 정규화*

img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

#--④ 히스토그램 계산*

hist = cv2.calcHist([img], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([img_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([img_norm2], [0], None, [256], [0, 255])

cv2.imshow('Before', img)
cv2.imshow('Manual', img_norm)
cv2.imshow('cv2.normalize()', img_norm2)

hists = {'Before' : hist, 'Manual':hist_norm, 'cv2.normalize()':hist_norm2}
for i, (k, v) in enumerate(hists.items()):
    plt.subplot(1,3,i+1)
    plt.title(k)
    plt.plot(v)
plt.show()
