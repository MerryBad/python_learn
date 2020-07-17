import requests
import re
import json
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
#     print(loc)
    # prov = re.findall(r"<province>(.+)</province>", loc)
    # city = re.findall(r"<city>(.+)</city>", loc)
    # print (prov[0], city[0])

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
# for loc in locations:
#     dates = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
#     for data in dates:
#         mode=re.findall(r'<mode>(.+?)</mode>', data)
#         tmEf=re.findall(r'<tmEf>(.+?)</tmEf>', data)
#         wf=re.findall(r'<wf>(.+?)</wf>', data)
#         tmn=re.findall(r'<tmn>(.+?)</tmn>', data)
#         tmx=re.findall(r'<tmx>(.+?)</tmx>', data)
#         rnSt=re.findall(r'<rnSt>(.+?)</rnSt>', data)
#         print(mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])

# 서울ㆍ인천ㆍ경기도
# A02 2020-07-19 00:00 흐림 22 27 40
# A02 2020-07-19 12:00 흐리고 비 22 27 80
# ==>
# 서울ㆍ인천ㆍ경기도 A02 2020-07-19 00:00 흐림 22 27 40

# for loc in locations:
#     test = re.findall(r"<province>(.+?)</province>.+<city>(.+?)</city>", loc, re.DOTALL)
#     # print(test[0])
#     prov, city = test[0]
#     dates = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
#     for data in dates:
#         # result=re.findall(r'<mode>(.+?)</mode>.+<tmEf>(.+?)</tmEf>.+<wf>(.+?)</wf>.+<tmn>(.+?)</tmn>.+<tmx>(.+?)</tmx>.+<rnSt>(.+?)</rnSt>', data, re.DOTALL)
#         # mode, tmEf, wf, tmn, tmx, rnSt = result[0]
#         result=re.findall(r'<.+>(.+?)</.+>', data)
#         mode, tmEf, wf, tmn, tmx, rnSt = result               # unpacking
#         print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
#         print(prov, city, *result)                            # packing

# prov와 city를 한번에 찾기
# for loc in locations:
#     test = re.findall(r"<province>(.+?)</province>\r\n\t\t\t\t<city>(.+?)</city>", loc, re.DOTALL)
#     print(test[0])
# < province > 서울ㆍ인천ㆍ경기도 < / province >
# < city > 서울 < / city >
# for loc in locations:
#     test = re.findall(r"<province>(.+?)</province>.+<city>(.+?)</city>", loc, re.DOTALL)
#     # print(test[0])
#     prov, city = test[0]
#     print(prov, city)#####################다중치환######################
#
# 파싱한 결과를 weather.csv 파일에 저장하기

# 파일로 저장하면서 , 로 구분한다
# print(prov,city,mode,tmEf,wf,tmn,tmx,rnSt, file=f, sep=',')
#
# f = open("./data/weather.csv", 'w', encoding='utf-8')
# for loc in locations:
#     test = re.findall(r"<province>(.+?)</province>.+<city>(.+?)</city>", loc, re.DOTALL)
#     prov, city = test[0]
#     dates = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
#     for data in dates:
#         result=re.findall(r'<.+>(.+?)</.+>', data)
#         mode, tmEf, wf, tmn, tmx, rnSt = result
#         f.write(prov + "," + city + ",")
#         for i in result:
#             f.write(i+ ",")
#         f.write("\n")
# f.close()

f = open("./data/weather.csv", 'w', encoding='utf-8')
for loc in locations:
    test = re.findall(r"<province>(.+?)</province>.+<city>(.+?)</city>", loc, re.DOTALL)
    prov, city = test[0]
    dates = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    for data in dates:
        result=re.findall(r'<.+>(.+?)</.+>', data)
        mode, tmEf, wf, tmn, tmx, rnSt = result
        #문자열로 변환
        line='{},{},{},{},{},{},{},{}\n'.format(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        f.write(line)
f.close()
