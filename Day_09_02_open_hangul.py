import requests
import re

# 오픈한글에 있는 API를 사용해서 한글 단어에 일치하는 영문 자판을 출력하세요
word = input("한글을 입력하세요 : ")
url="https://openhangul.com/nlp_ko2en?q="
received = requests.get(url+word)
# print(received.text)
result=re.findall(r'<img src="images/cursor.gif"><br>(.+)', received.text)
print(result[0].strip())