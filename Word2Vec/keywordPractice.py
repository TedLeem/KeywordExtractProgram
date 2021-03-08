from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.utils import pprint
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        # 중 KoNLPy 중 꼬꼬마(Kkma) 분석기 함수 중 문장을 추출하는 sentences()라는 함수를 이용하여 문장을 분리한다.
        self.okt = Okt()
        # 문장으로 분리 한 뒤 문장을 형태소 단위로 나눈 후 품사 태깅을 통해 명사들만 추출한다
        # KoNLPy 중 Twitter를 이용하여 명사를 추출해 준다.어디있었떤건데?
        self.stopwords = ['중인' ,'만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자"
             ,"아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가","ㅜ"]
    # 불 언어 
    def url2sentences(self, url):
        # url 주소를 받아 기사내용(article.text)을 추출하여 Kkma.sentences()를 이용하여 문장단위로 나누어 준 후 senteces를 return 해 준다.
        
        f = open(url,mode='r',encoding='utf-8')
        if f.readable:

            textStr = f.read()
            sentences = self.kkma.sentences(textStr)
            
            for idx in range(0, len(sentences)):
                if len(sentences[idx]) <= 10:
                    sentences[idx-1] += (' ' + sentences[idx])
                    sentences[idx] = ''
            f.close()    
            return sentences
        else:
            print('cannot open file : ',url)
            return None
        
  
    def text2sentences(self, text):
        # text(str)를 입력받아 Kkma.sentences()를 이용하여 문장단위로 나누어 준 후 senteces를 return 해 준다.
        sentences = self.kkma.sentences(text)      
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        
        return sentences

    def get_nouns(self, sentences):
        # sentences를 받아 Twitter.nouns()를 이용하여 명사를 추출한 뒤 nouns를 return해 준다.
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence)) 
                                       if noun not in self.stopwords and len(noun) > 1]))
        return nouns


okt = Okt()
url = 'C:\Users\user\Desktop\EnigmaPython\'
sentenceTokenizer = SentenceTokenizer()
sentences = sentenceTokenizer.url2sentences(url)
for sentence in sentences:
    print('/////////////////morphs')
    print(okt.morphs(sentence))
    print('/////////////////pos')
    print(okt.pos(sentence))

