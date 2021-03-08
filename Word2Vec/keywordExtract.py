from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.utils import pprint
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import re


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        # 중 KoNLPy 중 꼬꼬마(Kkma) 분석기 함수 중 문장을 추출하는 sentences()라는 함수를 이용하여 문장을 분리한다.
        self.okt = Okt()
        # 문장으로 분리 한 뒤 문장을 형태소 단위로 나눈 후 품사 태깅을 통해 명사들만 추출한다
        # KoNLPy 중 Twitter를 이용하여 명사를 추출해 준다.
        url= 'C:/Users/user/Desktop/EnigmaPython/stopwords.txt'
        f = open(url,mode='r',encoding='utf-8')
        self.stopwords = f.readlines()
        self.stopwords = [stopword.strip() for stopword in self.stopwords]
        f.close()
        # 미리 다운받은 한국 불용어 모음을 stopwords 리스트에 저장한다
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
                # nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence)) 
                #                        if noun not in self.stopwords and len(noun) > 1]))
                morphs = self.okt.morphs(sentence)
                tempNouns = self.okt.nouns(sentence)
                pattern = re.compile('[가-히]+(다|고|서|로)')
                newMorphs = [morph for morph in morphs if morph not in tempNouns and len(morph) >1 
                            and not pattern.match(morph) ]
                            # 모든 형태소들중에서 명사들은 제외시키고 단어의 길이가 1인것들도 제외시키고 위의 정규식과 매칭되는 패턴도 제외시킨다.
                nouns.append(' '.join([noun for noun in newMorphs 
                        if noun not in self.stopwords and len(noun) > 1]))
                    # 불용어 들도 제외시킨다
                print('nouns :',end='')
                print( nouns)
        return nouns

class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
    def build_sent_graph(self, sentence):
        # 명사로 이루어진 문장을 입력받아 sklearn의 TfidfVectorizer.fit_transform을 이용하여
        #  tfidf matrix를 만든 후 Sentence graph를 return 한다.
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence
    def build_words_graph(self, sentence):
        # 명사로 이루어진 문장을 입력받아 sklearn의 CountVectorizer.fit_transform을 이용하여
        #  matrix를 만든 후 word graph와 {idx: word}형태의 dictionary를 return한다.
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}



class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.url2sentences(text)
            self.nouns = self.sent_tokenize.get_nouns(self.sentences)
            self.graph_matrix = GraphMatrix()
            self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
            self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
            self.rank = Rank()
            self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
            self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
            self.word_rank_idx = self.rank.get_ranks(self.words_graph)
            self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
            index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        return summary
    
    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        keywords = []
        index=[]
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        return keywords

url = 'C:/Users/user/Desktop/EnigmaPython/reviewPractice2.txt'

textRank = TextRank(url)
for row in textRank.summarize(10):
    print(row)
    print()
print('keywords:', textRank.keywords())
