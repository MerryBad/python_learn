import requests
import re
# 기상청
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
received = requests.get(url)
# print(received)
# print(received.text)

# province 를 찾기
# print(re.findall(r"<province>(['가-힣']+)",received.text))
# print(re.findall(r"<province>(.+)</province>",received.text))

# location 검색
# print(re.findall(r'<location wl_ver="3">(.+)</location>', received.text))
# DOTALL = 개행문자 무시 ( 찾으려고 하는 것이 여러 줄에 있을 때 )
# .+ : 탐욕적 (greedy)
# .+? : 비탐욕적 (non-greedy)
locations=re.findall(r'<location wl_ver="3">(.+?)</location>', received.text, re.DOTALL)
# print(len(locations))
# province와 city 찾기
# for loc in locations:
#     # print(loc)
#     prov = re.findall(r"<province>(.+)</province>", loc)
#     city = re.findall(r"<city>(.+)</city>", loc)
#     print (prov[0], city[0])

# data 찾기
# <data>
# <mode>A02</mode>
# <tmEf>2020-07-18 00:00</tmEf>
# <wf>구름많음</wf>
# <tmn>22</tmn>
# <tmx>28</tmx>
# <reliability/>
# <rnSt>30</rnSt>
# </data>
for loc in locations:
    dates = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    for data in dates:
        mode=re.findall(r'<mode>(.+?)</mode>', data)
        tmEf=re.findall(r'<tmEf>(.+?)</tmEf>', data)
        wf=re.findall(r'<wf>(.+?)</wf>', data)
        tmn=re.findall(r'<tmn>(.+?)</tmn>', data)
        tmx=re.findall(r'<tmx>(.+?)</tmx>', data)
        rnSt=re.findall(r'<rnSt>(.+?)</rnSt>', data)
        print(mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])