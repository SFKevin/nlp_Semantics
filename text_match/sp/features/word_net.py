filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "E:\\CIKM2018\\w2v.model.bin"
fast_path = "E:\\CIKM2018\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
file_stop_word = "E:\\CIKM2018\\spanish_stop_word.txt"
from CIKM.datautils import datahelper
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import wordnet as wn
import datetime
from multiprocessing import Pool
from nltk.metrics import edit_distance

x_train1, x_train2, _, _, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
x_test1, x_test2 = datahelper.load_testdata(filepath_test)

train = pd.DataFrame()
test = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

test['question1'] = x_test1
test['question2'] = x_test2


def tokenize(q1, q2):
    return q1.split(" "), q2.split(" ")
    # return word_tokenize(q1), word_tokenize(q2)


def posTag(q1, q2):
    return nltk.pos_tag(q1), nltk.pos_tag(q2)


class Lesk(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.meanings = {}
        for word in sentence:
            self.meanings[word] = ''

    def getSenses(self, word):
        # print word
        return wn.synsets(word.lower())

    def getGloss(self, senses):
        gloss = {}
        for sense in senses:
            gloss[sense.name()] = []
        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())
        return gloss

    def getAll(self, word):
        senses = self.getSenses(word)
        if senses == []:
            return {word.lower(): senses}
        return self.getGloss(senses)

    def Score(self, set1, set2):
        overlap = 0
        for word in set1:
            if word in set2:
                overlap += 1
        return overlap

    def overlapScore(self, word1, word2):
        gloss_set1 = self.getAll(word1)
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        score = {}
        for i in gloss_set1.keys():
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        bestSense = None
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        maxOverlap = 0
        context = sentence
        word_sense = []
        meaning = {}

        senses = self.getSenses(word)

        for sense in senses:
            meaning[sense.name()] = 0

        for word_context in context:
            if not word == word_context:
                score = self.overlapScore(word, word_context)
                if score[0] == None:
                    continue
                meaning[score[0]] += score[1]

        if senses == []:
            return word, None, None

        self.meanings[word] = max(meaning.keys(), key=lambda x: meaning[x])

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()


def path(set1, set2):
    return wn.path_similarity(set1, set2)


def wup(set1, set2):
    return wn.wup_similarity(set1, set2)


def edit(word1, word2):
    if float(edit_distance(word1, word2)) == 0.0:
        return 0.0
    return 1.0 / float(edit_distance(word1, word2))


def computePath(q1, q2):
    R = np.zeros((len(q1), len(q2)))
    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = path(wn.synset(q1[i][1]), wn.synset(q2[j][1]))
            if sim == None:
                sim = edit(q1[i][0], q2[j][0])
            R[i, j] = sim
    return R


def computeWup(q1, q2):
    R = np.zeros((len(q1), len(q2)))
    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = wup(wn.synset(q1[i][1]), wn.synset(q2[j][1]))
            if sim == None:
                sim = edit(q1[i][0], q2[j][0])
            R[i, j] = sim
    return R


def overallSim(q1, q2, R):
    sum_X = 0.0
    sum_Y = 0.0
    for i in range(len(q1)):
        max_i = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_i:
                max_i = R[i, j]
        sum_X += max_i
    for i in range(len(q1)):
        max_j = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_j:
                max_j = R[i, j]
        sum_Y += max_j

    if (float(len(q1)) + float(len(q2))) == 0.0:
        return 0.0

    overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

    return overall


def semanticSimilarity(questions):
    q1, q2 = questions.split('_split_tag_')
    tokens_q1, tokens_q2 = tokenize(q1, q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    sentence = []
    for i, word in enumerate(tag_q1):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense1 = Lesk(sentence)
    sentence1Means = []
    for word in sentence:
        sentence1Means.append(sense1.lesk(word, sentence))

    sentence = []
    for i, word in enumerate(tag_q2):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense2 = Lesk(sentence)
    sentence2Means = []
    for word in sentence:
        sentence2Means.append(sense2.lesk(word, sentence))

    R1 = computePath(sentence1Means, sentence2Means)
    R2 = computeWup(sentence1Means, sentence2Means)

    R = (R1 + R2) / 2

    return overallSim(sentence1Means, sentence2Means, R)


def makeFeature(df_features):
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    pool = Pool(processes=16)
    df_features['z_wordnet'] = pool.map(semanticSimilarity, df_features['questions'])
    pool.close()
    pool.join()
    print('all done')
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    df_features.fillna(0.0)
    return df_features


if __name__ == "__main__":
    train['questions'] = train['question1'] + '_split_tag_' + train['question2']
    test['questions'] = test['question1'] + '_split_tag_' + test['question2']
    train = makeFeature(train)
    test = makeFeature(test)
    test.to_csv('test_wordnet.csv', index=False)
