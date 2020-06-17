import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
TaggededDocument = gensim.models.doc2vec.TaggedDocument

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from operator import itemgetter


import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from sklearn.manifold import TSNE


import nltk
from fuzzywuzzy import fuzz

from summa.summarizer import summarize

IMG_PATH = 'static/img/'

plt.switch_backend('agg')


class ref_category_desp(object):

    def __init__(
            self,
            input_DF
    ):
        self.input_DF = input_DF

    # select the sentences that can match with the topic words
    def sentence_selection(self, abs, topic_list,match_ratio=30):

        matched_sentences = []

        abs = abs.lower()
        sentences = nltk.tokenize.sent_tokenize(abs)

        for sentence in sentences:

            for topic_word in topic_list:
                # match the sentences in abs with topic_words
                #ratio = fuzz.partial_ratio(sentence, topic_word)
                ratio = fuzz.token_set_ratio(sentence, topic_word)
                if (ratio >= match_ratio):
                    matched_sentences.append([sentence,topic_word,ratio])
                    break

        return matched_sentences

    # concat all the selected sentences into a string for textrank
    def sentence_concat(self, abs_list):
        sentences = ''
        for sentences_list in abs_list:
            for sentence in sentences_list:
                single_sentence = str(sentence[0])
                if ((single_sentence.startswith('keywords'))== False):
                    sentences = sentences + single_sentence

                    #print(single_sentence)
        sentences = sentences.replace('.', '. ')
        sentences = sentences.replace('.  ', '. ')
        sentences = sentences.replace('\n', ' ')
        sentences = sentences.replace('  ', ' ')
        return sentences

    # use textrank to get the category description
    def textrank_summary(self, sentences, summary_len):

        summary = summarize(sentences, words=summary_len)

        return summary

    # 找出category description句对应的句子
    # 其实是为了找到category description对应的topic word (经过concat已无法通过序号找到)
    def summary_sentence_match(self, summary_sentence, abs_list):
        max_ratio = 0
        matched_sentence_list = []
        for sentences_list in abs_list:
            for sentence in sentences_list:
                single_sentence = str(sentence[0])
                ratio = fuzz.ratio(single_sentence, summary_sentence)
                if (ratio >= max_ratio):
                    max_ratio = ratio
                    matched_sentence_list = sentence
        summary_sent_info = matched_sentence_list
        return summary_sent_info

    # 按固定格式写 category_desp
    def rewrite_category_desp(self, num, category_summary, topic_word):
        num_list = ['first','sencond','third','fourth','fifth','sixth','seventh','eighth','ninth']
        topic_word = topic_word.replace('-',' ')
        text = "The "+num_list[num] +" category is about the "+topic_word+'. '
        category_desp = text + category_summary
        category_desp = category_desp.replace('  ',' ')
        category_desp = category_desp.replace('.', '. ')
        category_desp = category_desp.replace('.  ', '. ')
        category_desp = category_desp.replace('\n', ' ')

        return category_desp

    def desp_generator(self, match_ratio = 70, summary_len = 30, topic_selection = 'topic_bigram'):

        train_tsv=self.input_DF
        data_without_NaN = train_tsv.dropna(axis=0)
        type_info = data_without_NaN['label'].value_counts()
        type_list = list(data_without_NaN['label'].unique())
        desp_list = []

        for type_item in type_list:
            desp_dict = {}
            train_data = data_without_NaN[data_without_NaN['label'] == type_item]

            topic_word_list = train_data['topic_word'].reset_index(drop=True)
            topic_word_list =topic_word_list[0]
            topic_bigram_list = train_data['topic_bigram'].reset_index(drop=True)[0]
            topic_trigram_list = train_data['topic_trigram'].reset_index(drop=True)[0]
            abstract_list = train_data['intro'].reset_index(drop=True)

            # select the sentences matched with the topic_words/topic_bigrams/topic_trigrams
            # from the abstracts in ref papers by the function sentence_selection
            if topic_selection == 'topic_word_list':
                topic_str = topic_word_list
            elif topic_selection == 'topic_trigram_list':
                topic_str = topic_trigram_list
            else:
                topic_str = topic_bigram_list

            topic_str = [i.replace('[', '') for i in topic_str]
            topic_str = [i.replace(']', '') for i in topic_str]
            topic_str = [i.replace('\\', '') for i in topic_str]
            topic_str = [i.replace('\'', '') for i in topic_str]
            topic_str = [i.replace('_', '-') for i in topic_str]
            topic_list = topic_str

            matched_sent_list = []
            for abs in abstract_list:
                matched_sentences = self.sentence_selection(abs, topic_list, match_ratio)
                matched_sent_list.append(matched_sentences)

            #print('___________________________')
            #print('type_item: ' + str(type_item) + '  fuzzy matched sentences: ')

            # concat the sentences in matched_sent_list and replace some symbols
            concat_sents = self.sentence_concat(matched_sent_list)


            # use textrank to summarize the selected sentences
            category_summary = self.textrank_summary(concat_sents, summary_len)

            summary_sent_info = self.summary_sentence_match(category_summary,matched_sent_list)

            desp_dict['category'] = type_item
            desp_dict['category_desp'] = self.rewrite_category_desp(type_item,category_summary,summary_sent_info[1])
            desp_dict['topic_word'] = summary_sent_info[1]

            desp_list.append(desp_dict)

        return desp_list


def clustering(df, n_cluster=3):
    text = df['abstract']
    wordstest_model = text
    test_model = [
        [wordnet_lemmatizer.lemmatize(word.lower()) for word in remove_stopwords(strip_punctuation(words)).split()] for
        words in wordstest_model]
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    top_words = []
    for testword in text:
        test_bow = dictionary.doc2bow([wordnet_lemmatizer.lemmatize(word.lower()) for word in
                                       remove_stopwords(strip_punctuation(testword)).split()])
        test_tfidf = tfidf_model[test_bow]
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]  # [:len(test_tfidf)]
        top_words.append([(dictionary[i[0]]) for i in top_n_words])

    x_train = []
    cnt = 0
    for i, text in enumerate(text):
        word_list = top_words[cnt]
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
        cnt += 1

    model_dm = Doc2Vec(x_train, min_count=2, size=100, sample=1e-3, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=500)
    model_dm.save('model_dm')
    infered_vectors_list = []
    # print("load doc2vec model...")
    model_dm = Doc2Vec.load("model_dm")
    # print("load train vectors...")
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    sim_matrix = cosine_similarity(infered_vectors_list)
    labels = SpectralClustering(n_clusters=n_cluster).fit_predict(sim_matrix)
    df['label'] = labels

    tfidf_top_words = []
    for i in range(len(x_train)):
        string = ""
        text = x_train[i][0]
        for word in text:
            string = string + word + ' '
        tfidf_top_words.append(string)
    df['top_n_words'] = tfidf_top_words

    ## ------ get unigram ------
    n = df['label'].nunique()
    top_dicts = []
    for i in range(n):
        tmp_dict = {}
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['top_n_words']
                for word in [wordnet_lemmatizer.lemmatize(word.lower()) for word in
                             remove_stopwords(strip_punctuation(testword)).split()]:
                    try:
                        tmp_dict[word] += 1
                    except:
                        tmp_dict[word] = 1
        top_dicts.append(tmp_dict)

    top_list = []
    for d in top_dicts:
        tmp = sorted(d.items(), key=itemgetter(1), reverse=True)
        tmp_list = []
        for i in [0, 1, 2]:
            if tmp[i][1] >= 2:
                tmp_list.append(tmp[i][0])
        top_list.append(tmp_list)

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_word'] = topic_words

    ## ------ get bigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [word.lower() for word in remove_stopwords(strip_punctuation(testword)).split()]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    bigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 1):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1])

        bigram_docs.append(tmp)

    test_model = bigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(bigram):
        for i in range(n):
            for j in range(len(docs[i]) - 1):
                if bigram == docs[i][j] + "_" + docs[i][j + 1]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
        # print([(get_source(dictionary[i[0]]), i[1]) for i in top_n_words])
        top_list.append([(get_source(dictionary[i[0]])) for i in top_n_words])
        # print()

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_bigram'] = topic_words

    ## ------ get trigram ------
    n = df['label'].nunique()
    docs = []
    for i in range(n):
        tmp = []
        docs.append(tmp)

    source_docs = []
    for i in range(n):
        tmp = []
        source_docs.append(tmp)

    for i in range(n):
        for j, r in df.iterrows():
            if r['label'] == i:
                testword = r['abstract']
                for word in [word.lower() for word in remove_stopwords(strip_punctuation(testword)).split()]:
                    docs[i].append(lancaster_stemmer.stem(wordnet_lemmatizer.lemmatize(word)))
                    source_docs[i].append(word)

    trigram_docs = []
    for i in range(n):
        tmp = []
        for j in range(len(docs[i]) - 2):
            tmp.append(docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2])

        trigram_docs.append(tmp)

    test_model = trigram_docs
    dictionary = corpora.Dictionary(test_model, prune_at=2000000)
    corpus_model = [dictionary.doc2bow(test) for test in test_model]
    tfidf_model = models.TfidfModel(corpus_model)
    corpus_tfidf = tfidf_model[corpus_model]

    def get_source(trigram):
        for i in range(n):
            for j in range(len(docs[i]) - 2):
                if trigram == docs[i][j] + "_" + docs[i][j + 1] + "_" + docs[i][j + 2]:
                    return (source_docs[i][j] + "_" + source_docs[i][j + 1] + "_" + source_docs[i][j + 2])

    top_list = []
    for testword in test_model:
        test_bow = dictionary.doc2bow(testword)
        test_tfidf = tfidf_model[test_bow]
        top_n_words = sorted(test_tfidf, key=lambda x: x[1], reverse=True)[:5]
        top_list.append([(get_source(dictionary[i[0]])) for i in top_n_words])
        # print([(get_source(dictionary[i[0]]), i[1]) for i in top_n_words])

    topic_words = []
    for j, r in df.iterrows():
        topic_words.append(top_list[r['label']])

    df['topic_trigram'] = topic_words


    ## get tsne fig
    tsne = TSNE(n_components=2, init='pca', perplexity=12)
    X_tsne = tsne.fit_transform(np.array(infered_vectors_list))
    colors = scatter(X_tsne, df['label'])

    plt.savefig(IMG_PATH + 'tsne_result' + '.png', dpi=800)
    plt.close()
    return df, colors


def scatter(x, colors):
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("muted", 10))
    color_hex = sns.color_palette("muted", 10).as_hex()
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=1,
                    c=palette[colors.astype(np.int)])
    c = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in colors]
    for i in range(x.shape[0]):
        ax.text(x[i, 0], x[i, 1], '[' + str(i) + ']', fontsize=15, color=c[i], weight='1000')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    return color_hex[:colors.nunique()]


def get_cluster_description(df):
    match_ratio = 70  # threshold for fuzzy matching
    summary_len = 30  # length of generated summary
    topic_selection = 'topic_bigram_list'  # topic_bigram_list #topic_word_list #topic_trigram_list
    input_DF = df
    category_desp = ref_category_desp(input_DF)
    description_list = category_desp.desp_generator(match_ratio=match_ratio, summary_len=summary_len, topic_selection='topic_bigram_list')
    return description_list