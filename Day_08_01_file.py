#poem.txt

# def read_1():
#     f = open('./data/poem.txt', 'r', encoding='utf-8')
#     lines = f.readlines()
#     print(lines)
#     f.close()
#
# read_1()

# def read_2():
#     f = open('./data/poem.txt', 'r', encoding='utf-8')
#     while True:
#         line=f.readline()
#         print(line.strip())#문자열의 양쪽 공백 제거
#         #print(line, end='')#프린트의 개행문자 출력 x
#         if not line: # 비어있다면
#             break
#     f.close()
#
# read_2()


# def read_3():
#     f = open('./data/poem.txt', 'r', encoding='utf-8')
#     lines=[]
#     for line in f:
#         # print(line.strip())
#         lines.append(line.strip())
#
#     f.close()
#     return lines
# read_3()

# 파일에 들어있는 단어의 갯수는 몇개?
# import re
# count=0
# lines=read_3()
# for line in lines:
#     print(line)
#     words=re.findall(r'[가-힣]+',line)
#     print(words)
#     count +=len(words)
# print('words :', count)

# def read_4(): # 알아서 close 해줌
#     with open('./data/poem.txt', 'r', encoding='utf-8') as f:
#       for line in f:
#           print(line.strip())
#
# read_4()


def write():
    f = open('./data/sample.txt', 'w', encoding='utf-8')
    f.write('hello\n')
    f.write('python')


    f.close()

write()