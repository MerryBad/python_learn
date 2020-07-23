import pandas as pd
import numpy as np

def get_data():
    # ratings UserID::MovieID::Rating::Timestamp
    # users   UserID::Gender::Age::Occupation::Zip-code
    # movies  MovieID::Title::Genres

    users = pd.read_csv('ml-1m/users.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=['UserID','Gender','Age','Occupation','Zip-code']
                        )
    movies = pd.read_csv('ml-1m/movies.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=['MovieID','Title','Genres']
                        )
    ratings = pd.read_csv('ml-1m/ratings.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=['UserID','MovieID','Rating','Timestamp']
                        )

    # print(users)
    # print(movies)
    # print(ratings)

    data = pd.merge(pd.merge(ratings, users), movies)
    # print (data)
    return data

def pivot_basic():
    df=get_data()
    # pv1 = df.pivot_table(index='Age', values='Rating')
    # print(pv1.head(), end='\n\n')
    # pv2 = df.pivot_table(index='Age', columns='Gender', values='Rating')
    # print(pv2.head(), end='\n\n')

    # # 18세의 여성 데이터만 출력하기
    # print(pv2.head().values[1, 0])
    # # print(pv2.F, end='\n\n')
    # print(pv2.F[18], end='\n\n')
    # # print(pv2.loc[18], end='\n\n')
    # print(pv2.loc[18]['F'], end='\n\n')

    # pv3 = df.pivot_table(index=['Age','Gender'], values='Rating')
    # print(pv3.head(), end='\n\n')
    # # print(pv3.unstack().head(), end='\n\n')
    # # 18세의 여성 데이터만 출력하기
    # print(pv3.loc[18,'F'].values, end='\n\n')
    # # print(pv3['Rating'], end='\n\n')
    # print(pv3.Rating[18,'F'], end='\n\n')

    # pv4 = df.pivot_table(index=['Age'], columns=['Occupation', 'Gender'], values='Rating')
    # print(pv4.head(), end='\n\n')

    # pv5 = df.pivot_table(index=['Age'], columns=['Occupation', 'Gender'], values='Rating', fill_value='0')
    # pv5.index=["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    # print(pv5.head(), end='\n\n')

    # pv6 = df.pivot_table(index='Age', columns='Gender', values='Rating', aggfunc=[np.mean, np.sum])
    # print(pv6.head(), end='\n\n')

    pv7 = df.pivot_table(index='Age', columns='Gender', values='Rating', aggfunc=np.mean)

    pv8 = df.pivot_table(index='Age', columns='Gender', values='Rating', aggfunc=np.sum)

    pv9 = pd.concat([pv7,pv8], axis=0)
    print(pv9, end='\n\n')

def over500():
    df = get_data()
    # 1/ 영화 제목별 평점
    by_Title=df.pivot_table(index='Title', columns='Gender', values='Rating')
    print(by_Title.head())
    # 2/ 최소 500번 이상 평가한 영화 목록
    by_count=df.groupby(by='Title').size()
    # print(by_count)
    bool_500=(by_count.values >= 500)
    # print(bool_500)
    # print(type(bool_500))

    over_500=by_count[bool_500]
    # print(over_500)
    return by_Title, over_500
    # return by_Title.loc[over_500.index]
# 3/ 여성들이 선호하는 영화 검색
# by_Title에서 titles와 일치하는 영화만 추출하기
by_Title, titles= over500()
title_500=by_Title.loc[titles.index]
# print(title_500)
# # top 5 추출
# top_female = title_500.sort_values('F')
# # print(top_female)
# print(top_female.tail())
top_female = title_500.sort_values('F', ascending=False)
print(top_female.head(), end='\n\n')

# 성별 호불호가 갈리지 않는 영화 top 5
title_500['Diff']=abs(title_500.F-title_500.M)
# print(title_500.head())
diff_500=title_500.sort_values('Diff', ascending=True)
print(diff_500.head())
# get_data()
# pivot_basic()

