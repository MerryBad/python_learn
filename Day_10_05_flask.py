from flask import Flask, render_template
import random
# Flask 파이썬으로 웹사이트 만들기 가능
# static 과 templates 폴더 만들기
# 이미지      html ~
app = Flask(__name__)
@app.route('/')#루트 설정
def index():
    return '쉬는 시간입니다'
@app.route('/lotto')
def lotto():
    numbers=[random.randrange(45)+1 for _ in range(6)]
    return str(numbers)
@app.route('/html')
def html():
    numbers=[random.randrange(45)+1 for _ in range(6)]
    return render_template('randoms.html', numbers=numbers) # html과 함께 변수 전달





if __name__=='__main__':
    app.run(debug=True)#코드 수정시 바로 반영됨
