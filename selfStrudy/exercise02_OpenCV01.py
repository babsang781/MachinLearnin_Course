# OpenCV 사용법과 , 이미지 처리 방법
## 특히 파이썬 라이브러리로 굉장히 사용하기 쉽게 특화!

# openCV + python 개요
## 원래는 c++로 인텔에서 만든 것이지만, 현재는 다양하게 지원
## c++로 된 OpenCV 라이브러리들을 파이썬 래퍼로 감싸고, 모듈을 추가한 것( 잘 못알아듣겠네)
## OpenCV 의 배열은 Numpy 배열로 변환되어 내부 처리를 수행, Numpy, SciPy, Matplotlib 호환

# 아나콘다3 네비게이터에 OpenCV 설치하기

import cv2

# imread() : 외부 이미지를 읽어옴
# cv2.IMREAD_COLOR : 이미지를 칼라 이미지로 설정
lion = cv2.imread("./image/lion.jpg", cv2.IMREAD_COLOR)

import matplotlib.pyplot as plt

# 눈금삭제 
plt.xticks([])
plt.yticks([])

plt.imshow(lion)

lion = cv2.cvtColor(lion, cv2.COLOR_RGB2BGR)
plt.xticks([])
plt.yticks([])

plt.imshow(lion)

# 이미지 처리에서 사용하는 색상
## Color 이미지: RGB로 구성된 이미지
### Red, Green, Blue 각각 8비트로 구성(0-255)
### 트루 칼라 : 24비트
### 참고: ARGB : Alpha 투명도 추가된 것
### RGB 칼라의 단점 : 복잡하고, 빛에 의해 영향을 받음
## 흑백(gray) 이미지: RGB 채널의 평균을 내서 하나의 채널로 통합 0-255
## 이진 (binary) 이미지 : 0 과 255(1) 로만 구성된 이미지

# 흑백이미지로 변환하기
## cv2.IMREAD_GRAYSCALE  
lion_gray = cv2.imread("./image/lion.jpg", cv2.IMREAD_COLOR)
lion_gray = cv2.cvtColor(lion_gray, cv2.COLOR_BGR2GRAY)

plt.xticks([])
plt.yticks([])
# cmap="gray" : 출력 모드 : 흑백
plt.imshow(lion_gray, cmap="gray")

# OCR ( 광학 문자 판독 : OPTICAL CHARACTER R...)
## 이미지로 된 문자를 텍스트로 변환하는 작업
### Tessaract API , 구글 OCR API, Naver OCR API 등을 사용할 수 있음
#### Tessaract API 다운로드 : https://github.com/UB-Mannheim/tesseract/wiki
##### tesseract-ocr-w64-setup-v5.0.0-alpha.20201127.exe

# Tessaract API 설치
!pip install pytesseract

## 설치위치를 지정
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import cv2
import matplotlib.pyplot as plt

# 이미지 불러오기
img = cv2.imread("./image/.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    

plt.xticks([])
plt.yticks([])
plt.imshow(img)

# 이미지에서 문자열을 추출
# lang="eng" : 언어를 영어로 설정 ( 한글: kor)
result = pytesseract.image_to_string(img, lang="eng")
print(result)

## tesseract API를 사용할 때는 이진 이미지로 변경한 후에 사용하는 것을 권장

img = cv2.imread("./image/.jpg") 
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    

## 흑백 이미지로 변경
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## 이진 이미지로 변경
_,binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

result = pytesseract.image_to_string(img, lang="eng")
print(result)



