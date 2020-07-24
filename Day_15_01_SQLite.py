import sqlite3

def read_weather():
    f=open('data\weather.csv','r',encoding='utf-8')
    # 아래 코드를 컴프리헨션으로 바꾸기
    data=[row.strip().split(',') for row in f]
    # for row in f:
    #     data.append()
    f.close()
    return data

def create_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor() # 데이터 위치를 가리킴

    query = 'CREATE TABLE kma (prov TEXT, city TEXT, mode TEXT, tmEf TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT)'
    cur.execute(query)

    conn.commit()
    conn.close()

def insert_row():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor() # 데이터 위치를 가리킴

    base = 'INSERT INTO kma VALUES ("{}","{}","{}","{}","{}","{}","{}","{}")'
    query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
    cur.execute(query)

    conn.commit()
    conn.close()

# sqlite3을 검색해서 데이터 읽어오는 쿼리 추가하기
def show_db():
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor() # 데이터 위치를 가리킴

    query = 'SELECT * FROM kma'
    for row in cur.execute(query):
        print(row)

    # row = [row for row in cur.execute(query)]
    # return row
    conn.commit()
    conn.close()

# 한번에 넣는 함수 만들기
def insert_row(rows):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor() # 데이터 위치를 가리킴

    base = 'INSERT INTO kma VALUES ("{}","{}","{}","{}","{}","{}","{}","{}")'
    for row in rows:
        # query = base.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        query = base.format(*row)
        cur.execute(query)


    conn.commit()
    conn.close()
# 특정 도시의 데이터를 가져오는 함수 만들기
def search_city(city):
    conn = sqlite3.connect('data/weather.sqlite3')
    cur = conn.cursor() # 데이터 위치를 가리킴

    query = 'SELECT * FROM kma WHERE city="{}"'.format(city) # 문자열 넣을때 "{}"
    for row in cur.execute(query):
        print(row)

    conn.commit()
    conn.close()


# create_db()
#
data=read_weather()
# for row in data:
#     insert_row()
# insert_row(data)
#
# show_db()

search_city('청주')
search_city('부산')
