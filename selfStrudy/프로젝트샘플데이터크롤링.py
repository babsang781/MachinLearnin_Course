# 학습데이터: 손 같이 나온 카드 크롤링

driver = wb.Chrome()
driver.get('https://www.google.com/search?q=%EC%86%90,+%EC%B9%B4%EB%93%9C&newwindow=1&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjQy7OqrY33AhWTLqYKHQmwBt8Q_AUoAXoECAEQAw&biw=1274&bih=1021&dpr=0.9')
time.sleep(0.7)

os.mkdir('C:/Users/smhrd/Desktop/손-카드')

#bs 컴퓨터가 알아들을 수 있게 변환 해주는 애
soup = bs(driver.page_source,"lxml")
img = soup.select(".islrc img")
time.sleep(0.7)

imgSrc = [] 
for i in img:
    try:
        imgSrc.append(i['src'])
    except:
        imgSrc.append(i['data-src'])
        
        
for i in range(len(imgSrc)):
    urlretrieve(imgSrc[i], f'C:/Users/smhrd/Desktop/손-카드/{i}.jpg')
    time.sleep(1.2)

# 폴더 삭제
import shutil
shutil.rmtree('C:/Users/smhrd/Desktop/손-카드')

driver = wb.Chrome()
driver.get('https://www.google.com/search?q=%EC%86%90,+%EC%B9%B4%EB%93%9C&newwindow=1&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjQy7OqrY33AhWTLqYKHQmwBt8Q_AUoAXoECAEQAw&biw=1274&bih=1021&dpr=0.9')
time.sleep(0.7)

os.mkdir('C:/Users/smhrd/Desktop/손-카드')

#bs 컴퓨터가 알아들을 수 있게 변환 해주는 애
soup = bs(driver.page_source,"lxml")
img = soup.select("#islrg > div.islrc > div.isv-r > a.islib > div.islir > img.rg_i")
img

imgUrl = [] 
imgUrl = img[0]['src']
imgUrl

os.mkdir('C:/Users/smhrd/Desktop/손-카드')

soup = bs(driver.page_source,"lxml")
img = soup.select(".islrc img")
time.sleep(0.7)

imgSrc = [] 
for i in img:
    try:
        imgSrc.append(i['src'])
    except:
        imgSrc.append(i['data-src'])
        
        
for i in range(len(imgSrc)):
    urlretrieve(imgSrc[i], f'C:/Users/smhrd/Desktop/손-카드/{i}.jpg')
    time.sleep(1.2)

for j in range(len(imgUrl)):
    urlretrieve(imgUrl[j], f'C:/Users/smhrd/Desktop/손-카드/{j}.jpg')
    time.sleep(1.2)





# 샘플 데이터 크롤링

import requests as req
from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys 
import time
import pandas as pd
from bs4 import BeautifulSoup as bs
import os    # 파일 시스템을 위한 라이브러리
from urllib.request import urlretrieve    # urlretrieve: 이미지의 경로를 파일로 저장시켜주는 라이브러리
from urllib import parse

dic={"상품명": [], "가격": [], "브랜드": [], "사이즈":[]}
cocomapet=pd.DataFrame(dic)
somedata = {"상품명": 1, "가격": 1, "브랜드": 1, "사이즈": 1}
cocomapet = cocomapet.append(somedata, ignore_index=True)
cocomapet

driver = wb.Chrome()
driver.get('http://cocomapet.com/product/detail.html?product_no=7229&cate_no=374&display_group=1')
time.sleep(0.7)

driver.execute_script("window.scrollTo(0, 700)")
time.sleep(0.7)
driver.execute_script("window.scrollTo(0, 0)")
driver.quit()

# clip board
import shutil
shutil.rmtree('C:/Users/smhrd/Desktop/이미지')

driver = wb.Chrome()
driver.get('http://cocomapet.com/product/detail.html?product_no=7229&cate_no=374&display_group=1')
time.sleep(1)
num =0

os.mkdir('C:/Users/smhrd/Desktop/이미지')
os.mkdir(f'C:/Users/smhrd/Desktop/이미지/{num}')
soup = bs(driver.page_source,'lxml')

# thumnail 저장
thumnail = soup.select('.BigImage')
urlretrieve('http:'+ thumnail[0]['src'], f'C:/Users/smhrd/Desktop/이미지/{num}/thumnail.jpg')
time.sleep(1)

# 상세 설명 이미지 저장
img = soup.select('#detailarea p > img')
for k in range(len(img)):
    parse_src = parse.quote(img[k]['src'])
    urlretrieve('http://cocomapet.com'+parse_src, f'C:/Users/smhrd/Desktop/이미지/{num}/{k}.jpg')
    time.sleep(1)

driver.quit()

import requests as req
from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys 
import time
import pandas as pd
import shutil    # 삭제 라이브러리
from bs4 import BeautifulSoup as bs
import os    # 파일 시스템을 위한 라이브러리
from urllib.request import urlretrieve    # urlretrieve: 이미지의 경로를 파일로 저장시켜주는 라이브러리
from urllib import parse


os.mkdir('C:/Users/smhrd/Desktop/이미지')
os.mkdir('C:/Users/smhrd/Desktop/csv')

# 크롤링 페이지 이동
driver = wb.Chrome()
driver.get('http://cocomapet.com/product/list.html?cate_no=496')
time.sleep(1)

# 크롤링 담아줄 df 생성
dic={"num": [], "category": [], "title": [], "price":[],"brand": [], "option": [], "picture_count":[]}
cocomapet = pd.DataFrame(dic)
num=88

# 카테고리별 이동
for i in range(12, len(cate_list)+1):
    driver.execute_script("window.scrollTo(0, 0)")
    time.sleep(1)
    category = driver.find_element_by_css_selector(f'#contents > div.xans-element-.xans-product.xans-product-menupackage > ul > li:nth-child({i}) > a').text
    driver.find_element_by_css_selector(f'#contents > div.xans-element-.xans-product.xans-product-menupackage > ul > li:nth-child({i}) > a').click()
    time.sleep(1.5)
    
    # 에러 없이 반복하기 -> 반복 횟수 구하기 
    count = int(driver.find_element_by_css_selector('#prdTotal > strong').text)
    if count > 8:
        count = 8
    
    # 세부 아이템 페이지 이동
    for j in range(count):
        num = num+1
        img = driver.find_elements_by_css_selector("img.thumb")
        img[j].click()
        time.sleep(1.5)

        # 세부 페이지 크롤링 제품명, 가격, 브랜드
        title = driver.find_element_by_css_selector(".detail-fixed-wrap > h3").text
        price = driver.find_element_by_css_selector(".product_price_css > td").text
        price = re.sub('원|,', "", price)
        brand = driver.find_element_by_css_selector(".prd_brand_css > td").text
        print(title)

        # 옵션이 있으면 -> 옵션 없이 바로 사이즈가 나오지 않으면
        try:
            temp = driver.find_element_by_css_selector(".detail-fixed-wrap > table > tbody > tr:first-child > th").text
            if temp != '사이즈' :
                options = driver.find_elements_by_css_selector(".ProductOption0 > option:nth-child(n+3)")
                for k in range(len(options)):
                    if k == len(options)-1:
                        option += options[k].text
                    else :
                        option += options[k].text + ','
            else :
                option = ''
        except:
            option = ''
            pass
        
        
        # 이미지 저장 : 폴더 생성 및 썸네일 저장, 
        os.mkdir(f'C:/Users/smhrd/Desktop/이미지/{num}')
        soup = bs(driver.page_source,'lxml')

        # thumnail 저장
        thumnail = soup.select('.BigImage')
        parse_src = parse.quote(thumnail[0]['src'])
        try:
            urlretrieve('http:'+ parse_src, f'C:/Users/smhrd/Desktop/이미지/{num}/thumnail.jpg')
            time.sleep(1.5)
        except:
            pass


        # 상세 설명 이미지 저장
        img = soup.select('#detailarea img')
        for k in range(len(img)):
            parse_src = parse.quote(img[k]['src'])
            try:
                urlretrieve('http://cocomapet.com' + parse_src, f'C:/Users/smhrd/Desktop/이미지/{num}/{k}.jpg')
                time.sleep(1.5)
            except:
                pass             
        picture_count = len(img)

        # df 저장
        one_row={"num": num, "category": category, "title": title, "price": price,"brand": brand, "option": option, "picture_count": picture_count}
        cocomapet = cocomapet.append(one_row, ignore_index = True)
       
        driver.back()
        time.sleep(1.5)
        
    cocomapet.to_csv(f'C:/Users/smhrd/Desktop/csv/cocomapet{num}.csv',encoding='euc-kr')
        
driver.quit()

# 견종 리스트

# import 별도

# 크롤링 페이지 이동 및 페이지 작업
driver = wb.Chrome()
driver.get('https://alpet.co.kr/common/dog_info_create_1/')
time.sleep(1)

# 크롤링 담아줄 df 생성
dic={"num": [], "name": [],}
dogBreed_list = pd.DataFrame(dic)


soup = bs(driver.page_source,'lxml')
dogBreed = driver.find_elements_by_css_selector("button.clickshow_dog_name")

dogBreed[10].text

dogBreeds=[]
for i in range(len(dogBreed)): 
    dogBreeds.append(dogBreed[i].text)

dic={"num": range(1, len(dogBreed)+1), "name": dogBreeds,}
dogBreed_list = pd.DataFrame(dic)
dogBreed_list

dogBreed_list.to_csv(f'C:/Users/smhrd/Desktop/dogBreed.csv',encoding='euc-kr')

# database 연결하기 - 1차시 실패

!pip install pymysql
!pip install sqlalchemy
