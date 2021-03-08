import re
from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.utils import pprint
from konlpy.tag import Kkma
from pykospacing import spacing
from hanspell import spell_checker
from collections import Counter

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
  "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 
  'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 
  'pi'} 

# 형태소 기준 토근화를 하기위한 데이터전처리
# 영어 숫자는 포함시키고 나머지 기호들을 제외, 띄어쓰기 준수 불용어 제거 등장빈도가 적고 길이가 짧은 단어 제외 

def clean_punc(text, punct, mapping): 
    for p in mapping: 
          text = text.replace(p, mapping[p]) 
    for p in punct: 
          text = text.replace(p, f' {p} ') 
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''} 
    for s in specials: 
          text = text.replace(s, specials[s]) 
    return text.strip()
 
def clean_text(texts): 
    corpus = []
    for i in range(0, len(texts)): 
      review = re.sub(r'[^가-히]',' ',str(texts[i]))
      review = re.sub(r'\s+', ' ', review) #remove spaces 
      corpus.append(review)
      # 한글을 제외한 나머지 다 제거 
      # review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.+\:\;\!+\-\,\_\~\$\'\"]', '',str(texts[i]))
      #remove punctuation  
      # review = re.sub(r'\d+','', str(review))# remove number
      # review = review.lower() #  #lower case 
      # review = re.sub(r'\s+', ' ', review) #remove extra space 
      # review = re.sub(r'<[^>]+>','',review) #remove Html tags 
      # review = re.sub(r"^\s+", '', review) #remove space from start
      # review = re.sub(r'\s+$', '', review) #remove space from the end 
      
    return corpus


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
            # textStr = spacing(textStr) #띄어쓰기 잡아주는 라이브러리 
            sentences = self.kkma.sentences(textStr)
            # kss 모듈 하나 더 있긴한데 나중에 해보기

            # sentences = textStr.split('.')            
            for idx in range(0, len(sentences)):
                if len(sentences[idx]) <= 10:
                    # sentences[idx-1] += (' ' + sentences[idx])
                    sentences[idx-1] += sentences[idx]
                    sentences[idx] = ''
            f.close()    
            return sentences
        else:
            print('cannot open file : ',url)
            return None

def spellCheck(file):
  # file을 불러와서 총제적으로 맞춤법 검사하여 반환
  sentenceTokenier = SentenceTokenizer()
  sentences = sentenceTokenier.url2sentences(file)
  # 단위로 끊어서 문장들을 반환
  sentences = clean_text(sentences) 
  # 한글제외한 단어들을 제외시킴 
  # 문서내용을 가-히만 포함시키도록함
  txtFile = ''
  for sentence in sentences:
    # txtFile.join(spell_checker.check(sentence).checked)
    txtFile += spell_checker.check(sentence).checked
  return txtFile

def excludeOneWord(txt):
  nouns_tagger = Okt()
  count = Counter(nouns_tagger.nouns(txt))
  print(count)
  # remove_char_counter = Counter({x : count[x] for x in count if len(x) > 1})
  # print(remove_char_counter)


fl= 'C:/Users/user/Desktop/EnigmaPython/reviewPractice2.txt'
# txtFile = spellCheck(fl)

#//////////////////////////////////////////
# 데이터 연습 
f = open(fl, mode='r', encoding= 'utf-8')
txt = f.read()
okt = Okt()
nouns = okt.nouns(txt)
phrases = okt.phrases(txt)
morphs = okt.morphs(txt)
pattern = re.compile('[가-히]+(다|고|서|로|요)')
newMorphs = [morph for morph in morphs if morph not in nouns and len(morph) >1
                   and not pattern.match(morph)]
# newPhrases = [phrase for phrase in phrases if phrase not in nouns and len(phrase) > 1]
print(newMorphs)
print(newPhrases)
#//////////////////////////////////////////

# file2 = 'C:/Users/user/Desktop/EnigmaPython/reviewPracticeSpell.txt'
# f2 = open(file2,mode='w',encoding='utf-8')
# f2.write(txtFile)
