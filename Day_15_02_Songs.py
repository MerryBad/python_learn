import requests
import re
def show_songs(code, page):
    # form data
    payload = {
        'S_PAGENUMBER': page,   # 1
        'S_MB_CD': code,        # W0726200
        # 'S_HNAB_GBN': 'I',
        # 'hanmb_nm': 'G-DRAGON',
        'sort_field': 'SORT_PBCTN_DAY'
    }

    url='https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp'
    received = requests.post(url, data=payload)
    # print(received)
    # print(received.text)

    # get방식
    # https://~/search?q= ####

    # post방식
    # get 방식을 사용할 수 없을 때 사용
    # 1. 암호화가 필요할 때
    # 2. 많은 데이터를 전달할 때
    # 3. 폼을 전송할 때

    # 데이터 출력하기

    tbody=re.findall(r'<tbody>(.+?)</tbody>', received.text, re.DOTALL)
    # print(len(tbody))
    # print(tbody[1])

    tbody_text=tbody[1]

    # imgs=re.findall(r'<img src="/images/common/control.gif"  alt="" />',
    #            tbody_text)
    # print(len(imgs))

    # imgs=re.findall(r'<img .+? />', tbody_text)
    # print(len(imgs))

    tbody_text=re.sub(r' <img .+? />', '', tbody_text)

    trs=re.findall(r'<tr>(.+?)</tr>',tbody_text,re.DOTALL)
    # print(len(trs))
    if not trs:
        return False
    for tr in trs:
        # print(tr)
        tr = re.sub(r'<br/>', ', ', tr)
        tds=re.findall(r'<td>(.+)</td>',tr)
        # tds = [td.strip() for td in tds]
        tds[0] = tds[0].strip()
        print(tds)
    return True
page=1
while show_songs('W0702500', page):
    print('---------------',page)
    page += 1