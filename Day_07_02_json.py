import requests
import json
import re

# d = {"ip":"8.8.8.8"}
# print(d)
# print(d['ip'])
#
# d2= json.dumps(d)
# print(d2)
# print(type(d2))
#
# d3= json.loads(d2)
# print(d3)
# print(type(d3))
# print('-'*50)

# dt 변수로부터 값만 뽑아서 출력하기
# dt='{"time": "03:53:25 AM", "milliseconds_since_epoch": 1362196405309, "date": "03-02-2013"}'
# print (json.loads(dt)['time'],json.loads(dt)['milliseconds_since_epoch'],json.loads(dt)['date'])
# print('-'*50)

# 문자열로 만들어짐 -> json
url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
received = requests.get(url)
# print(received)
# print(received.text)
origin=received.content.decode()
# print(origin)
# text=bytes.decode(origin, encoding='utf-8') #euc-kr
# print(text)

# code와 value에 들어있는 값만 출력하기
# text=json.loads(origin)
# for i in text:
#     print(i['code'], i['value'])
# print([(i['code'], i['value']) for i in text])
# 정규표현식을 사용해서 똑같은 결과 출력하기
# {"code":"11","value":"서울특별시"}
# print(re.findall(r'[^\"][0-9가-힣]+[^\"]', origin))
# codes=re.findall(r'[0-9]+', origin)
# values=re.findall(r'[가-힣]+', origin)
# print(codes)
# print(values)

# binds = zip(codes, values)
# 괄호 씌우면 해당 값만 반환됨
print(re.findall(r'"code":"([0-9]+)"', origin))
print(re.findall(r'"value":"([가-힣]+)"', origin))
# code와 value를 한번에 찾기
print(re.findall(r'"code":"([0-9]+)","value":"([가-힣]+)"', origin))
