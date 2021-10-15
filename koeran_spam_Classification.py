import math, sys
from konlpy.tag import Twitter

class BayesianFilter:
    def __init__(self):
        self.words = set()  # 출현한 단어 기록
        self.word_dict = {}  # 카테고리마다의 출현 횟수 기록
        self.category_dict = {}  # 카테고리 출현 횟수 기록

    # 1. 형태소 분석하기
    def split(self, text):
        # return text.split()
        results = []
        twitter = Twitter()  # 단어의 기본형 사용
        malist = twitter.pos(text, norm=True, stem=True)

        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                results.append(word[0])

        return results

        # 2. 단어와 카테고리의 출현 횟수 세기

    def inc_word(self, word, category):
        # 단어 횟수 세기
        # 단어를 카테고리에 추가하기
        if not category in self.word_dict:
            self.word_dict[category] = {}
        if not word in self.word_dict[category]:
            self.word_dict[category][word] = 0

        self.word_dict[category][word] += 1
        self.words.add(word)

    def inc_category(self, category):
        # 카테고리 횟수 세기
        # 카테고리 계산하기
        if not category in self.category_dict:
            self.category_dict[category] = 0

        self.category_dict[category] += 1

    # 3. 텍스트 학습하기
    def fit(self, text, category):
        # 텍스트를 형태소로 분할하고, 카테고리와 단어를 연결함
        word_list = self.split(text)
        for word in word_list:
            self.inc_word(word, category)
            self.inc_category(category)

    # 4. 단어 리스트에 점수 매기기
    def score(self, words, category):
        # 확률을 곱할 때 값이 너무 작으면 다운플로가 발생할 수 있어 log를 이용함
        score = math.log(self.category_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score

    # 5. 예측하기
    def predict(self, text):
        best_category = None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        for category in self.category_dict.keys():
            score = self.score(words, category)
            score_list.append((category, score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list

    # 6. 카테고리 내부의 단어 출현 비율 계산
    def word_prob(self, word, category):
        # 단어 출현률을 계산할 때 학습사전(word_dict)에 없는 단어가 나오면
        # 카테고리의 확률이 0이 되어버려 1을 더해 활용
        n = self.get_word_count(word, category) + 1
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d

    # 카테고리 내부의 단어 출현 횟수 구하기
    def get_word_count(self, word, category):
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0

    # 카테고리 계산
    def category_prob(self, category):
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories