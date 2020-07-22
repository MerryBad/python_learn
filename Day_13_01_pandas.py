import pandas as pd

# s=pd.Series([2,1,5,9])
# print(s)
# print(type(s))
#
# print(s.index)
# print(s.values)
# print(type(s.values))
#
# # print(s[-1]) # error
# print(s[0], s[3])

# s2=pd.Series([2,1,5,9], index=['a','b','c','d'])
# print(s2)

# print(s2[0], s2[3])
# print(s2[-1]) # 인덱스를 숫자로 안하면 가능

# # 2와 9를 출력하는 다른 코드
# print(s2['a'], s2['d'])

# 마지막 3개의 행을 출력하기
# print(s2[1:])
# print(type(s2[1:]))
#
# print(s2['b':])
# print(s2['b':'d'])
#
# print(s2.values)
# print(s2.values[1:])

df=pd.DataFrame({
    'year':[2018,2019,2020,2018,2019,2020],
    'city':['ochang','ochang','ochang','sejong','sejong','sejong'],
    'rain':[130, 150, 160, 150, 145, 155],
})
# print(df)
# print(type(df))
#
# print(df.head(), end='\n\n') # 상위 5개
# print(df.tail(), end='\n\n') # 하위 5개
#
# print(df.head(2), end='\n\n') # 상위 2개
# print(df.tail(2), end='\n\n') # 하위 2개
#
# df.info()
#
# print(df.index)
# print(df.columns)
# print(df.values)
# print(df.values.dtype)
#
# print(df['year'])
# print(df.year)
# print(type(df['year']))
df.index=['a','b','c','d','e','f']

# print(df.iloc[0])       # 숫자 사용할 때 iloc
# print(df.iloc[-1])
#
# print(df.loc['a'])      # 그 외에는 loc
# print(df.loc['f'])

# 데이터프레임에 대해 슬라이싱 문법을 확인하기
# print(df.iloc[1:4],end='\n\n')
# print(df.loc['b':'d'],end='\n\n')
# print(df.iloc[::-1])
# print(df[::-1])
# print(df[1:4])

# print(df.pivot('year','city','rain'))
# print(df.pivot('city','year','rain'))

# print(df.ix['a']) #error