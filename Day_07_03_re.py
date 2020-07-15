import re
db= '''3412    [Bob] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''

#print(db)
# 검색할때 findall(r'(찾을거)', obj)
# 숫자 찾기
finds = re.findall(r'[0-9]+',db)
print(finds)

# 이름만 찾아보기
print ( re.findall(r'[A-Z][a-z]+',db))
print ( re.findall(r'[A-Za-z]+',db))

###########################################

# T로 시작하는 이름을 찾기
print(re.findall(r'[T][a-z]+', db))
# T로 시작하지 않는 이름 찾기
print(re.findall(r'[A-SU-Z][a-z]+', db))