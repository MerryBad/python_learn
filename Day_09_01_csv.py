import csv
# weather.csv 파일을 필드별로 분리해서 출력하기
# => , 빼고 출력하기
def read_csv_1():
    f=open("./data/weather.csv","r",encoding="utf-8")
    # for text in f:
    #     word=text.strip().split(',')
    #     prov, city, mode, tmEf, wf, tmn, tmx, rnSt = word
    #     print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)

    # rows=[]
    # for row in f:
    #     # print(row.strip().split(','))
    #     rows.append(row.strip().split(','))

    ## 위 반복문을 컴프리헨션으로 바꾸기
    rows=[row.strip().split(',') for row in f]
    return rows
    f.close()

def read_us500():
    f=open("data/us-500.csv","r",encoding="utf-8")

    for row in csv.reader(f):
        print(row)

    f.close()
# read_csv_2()
#
# # rows에 들어있는 값을 이전 결과처럼 출력하세요
# text=read_csv_2()
# for line in text:
#     for col in line:
#         print(col,end=' ')
#     print('\b')
#     # split <-> join
#     print(','.join(line))
# read_us500()
def write_csv(rows):
    f = open("data/kma.csv", "w", encoding="utf-8", newline='')
    # for row in rows:
    #     f.write(''.join(row)+'\n')
    # writer ( delimiter 구분자  quoting 따옴표?)
    writer=csv.writer(f, delimiter='☆', quoting=csv.QUOTE_ALL)
    # for row in rows:
    #     writer.writerow(row)
    writer.writerows(rows)

    f.close()

rows=read_csv_1()
write_csv(rows)