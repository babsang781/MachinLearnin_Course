# 전체 실행 하지 않는 파일

# flask 인스톨 

!pip install flask

# 임포트

from flask import Flask # 플라스크 클래스 임포트

# 플라스크 객체를 생성해서 사용할 프로젝트에서 사용하는 앱(기능) 관련 변수를 생성
app = Flask(__name__) 

## "현재 모듈의 이름을 담고있는 내장 변수": __name__ 
## 파이썬을 일반 실행프로그램으로 만들면 해당하는 파일의 이름이 name 의 속성으로 들어가게 됨. 내장함수.

### 이 작업이 있어야 서버를 구동시키는 Flask 객체가 생성됨.

# 이제 app 한테 서버 구동 요청하면 됨.

# 서버 구동


## interpreter 에서 .py 파일을 직접실행할 경우, 
## 내장변수에 __main__이 담기게 되고, 이 경우 실행하라는 뜻

#    app.run() 을 통한 웹 주소 및 포트 제어
if __name__ == '__main__' : # .py 파일에서 main 함수 역할
    app.run(host='localhost', port='5000')

## app.run(host='127.0.0.1', port='5000') 이런 식으로 
## host 와 port 번호를 통해 웹 주소를 변경할 수 있음.
## 주로 사용하는 포트가 5000 , vs 코드도 그랬던듯, 중복되면 9000번도 사용가능

## 지금 혼자 사용하는 경우 localhost로 해서 사용가능하지만, 
## 각각 다른 개발자가 개발하면 하나의 url을 만들어서 사용하면 됨.





# : 1차 모음 테스트

from flask import Flask # 플라스크 클래스 임포트

# 플라스크 객체를 생성해서 사용할 프로젝트에서 사용하는 앱(기능) 관련 변수를 생성
app = Flask(__name__) 

# 여기서는 함수별로 url mappeing 을 한다고 생각하면됨

@app.route('/')  # @app.route('/route') 로 들어오면 실행하는 함수 작성
def first() : 
    return "어서오세요"  # http://localhost:5000/ 실행 후 일로 가면 return만 보임

@app.route('/model_b')  # url 다르게 해서 다른 모델 실행하도록 가능
def second() : 
    return "model_b실행 페이지" 


if __name__ == '__main__' : # .py 파일에서 main 함수 역할
    app.run(host='localhost', port='5000')


# 연습 페이지 만들어보기

from flask import Flask  # import 
from flask import request, redirect
import pickle    # 데이터 로딩 / 저장을 위한 모듈 library  import
import pandas as pd
import numpy as np

app = Flask(__name__)  # 객체 생성

# url 이동 전, 객체 생성과 같이 모델 불러오기, 같은위치 파일 자동완성 가능..
## rb read binary
with open('knn_model.pkl', 'rb') as f:   # iris 학습한 knn 모델을 불러옴
    model = pickle.load(f)


@app.route('/iris_test/', methods=['POST'])    # url mapping 주소 및 함수 생성
def iris_test() : 
    if request.method == 'POST':
        display(request.form)  # 주피터 노트북에서 마지막 print 없이 실행 보여주는 함수 , df 잘 나오는 것은 데이터 형태 그대로 보여주기 때문 // print 는 값만 보여주는 것
        
        # get 방식은 .args 를 사용 : num1 = request.args['num1'], num2 = request.args['num2'] 이런 형태 변수 입력도 가능
        # 넘어온 값을 전처리 preprocessing(requset.form)
        df = preprocessing(request.form)
        pre = model.predict(df)
        print(pre)
        return "done~"
        # 서버 간 이동이 있기 때문에 redirect 방식으로 querystring 을 통해 결과 데이터 전송
        # return redirect("http://localhost:8081/anisize/result.do?predict="+ str(pre[0])) 

if __name__ == '__main__' :    # 서버 구동 및 host 지정
    app.run(host='localhost', port='5000')



# 넘어온 값을 전처리하는 함수 작성 
# 딕셔너리 데이터로 값 저장된 상태
def preprocessing(data_dic):
    with open('iris_columns.pkl', 'rb') as f :
        iris_columns = pickle.load(f)
    
    # np.zeros((1,4)) 1행 4열 짜리 0으로 채워진 df 생성, 컬럼 입력
    df = pd.DataFrame(np.zeros((1, 4)), columns=iris_columns)   
    
    df['sepal length (cm)'] = data_dic['sepal_len']
    df['sepal width (cm)'] = data_dic['sepal_wid']
    df['petal length (cm)'] = data_dic['petal_len']
    df['petal width (cm)'] = data_dic['petal_wid']
    
    return df






