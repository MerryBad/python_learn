import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc, colors # 한글깨짐 수정

# 여성 데이터 추가하기
def bar_1():
    men = [25, 17, 21, 23, 30]
    women = [27, 13, 24, 31, 29]

    indices = np.arange(len(men))
    plt.bar(indices,men,width=0.4, color='r')
    plt.bar(indices+0.4,women,width=0.4, color='g')
    plt.xticks(indices+0.2, ['A','B','C','D','E'])
    plt.show()

# gdp파일을 읽어서 상위 10개국의 데이터를 막대 그래프로 표시
# 순위:국가이름:데이터
def bar_2():
    f = open('./data/2016_GDP.txt', 'r', encoding='utf-8')
    f.readline() # 첫번째 줄 버리기

    names, dollors = [], []
    for row in f:
        # # print(row.strip())
        # items=row.strip().split(':')
        # # print(items)
        # names.append(items[1])
        # dollors.append(items[2])
        _, name, dollor = row.strip().split(':') # _ : place holder
        dollor= dollor.replace(',', '')
        # dollor=''.join(dollor.split(','))
        names.append(name)
        dollors.append(int(dollor))
    f.close()
    names_10=names[:10]
    dollors_10=dollors[:10]
    # print(names_10)
    # print(dollors_10)
    ttf='C:\Windows\Fonts\malgun.ttf'
    font_name= font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name) # 한글 폰트 추가

    indicies=range(10)
    # plt.bar(indicies, dollors_10)
    # plt.bar(indicies, dollors_10, color='r')
    # plt.bar(indicies, dollors_10, color='rgb')
    # plt.bar(indicies, dollors_10, color=colors.BASE_COLORS)
    # plt.bar(indicies, dollors_10, color=colors.TABLEAU_COLORS)
    # plt.bar(indicies, dollors_10, color=['r','g','b'])
    plt.bar(indicies, dollors_10, color=['red','green','blue'])

    # plt.xticks(indicies, names_10, rotation='vertical') # 한글깨짐
    plt.xticks(indicies, names_10, rotation=60)
    plt.title('2016 GDP')
    plt.subplots_adjust(bottom=0.15) # 차트 크기 조절
    plt.show()

# bar_1()
bar_2()

