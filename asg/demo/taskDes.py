#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:28:09 2020

@author: yangruosong
"""
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from summa.summarizer import summarize


def readReferenceTsv(fileName):
    refs = []
    with open(fileName, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split("\t")
            if len(line) == 6:
                refs.append([line[4], line[5]])

    return refs


def cleanText(text):
    texts = text.strip().split("\n")
    sents = []
    for t in texts:
        sents += sent_tokenize(t)
    return sents


def sentText(refs):
    newRefs = []
    aiRefs = []
    abslist = []
    introlist = []
    for ref in refs:
        newAbs = [word_tokenize(sent) for sent in cleanText(ref[0])]
        newIntro = [word_tokenize(sent) for sent in cleanText(ref[1])]
        newRefs.append([newAbs, newIntro])
        aiRefs.append(newAbs + newIntro)
        abslist.append(newAbs)
        introlist.append(newIntro)

    return newRefs, aiRefs, abslist, introlist


def testTopicSentence(topic, sent):
    flag = 0
    for word in topic:
        if word in topic:
            flag += 1
    if flag == len(topic):
        return True
    else:
        return False


def testContainTopic(topic, abstract):
    for sent in abstract:
        if testTopicSentence(topic, sent):
            return True

    return False


def extractTopicRef(topic, abslist, introlist):
    newabs = []
    newintro = []
    for al, il in zip(abslist, introlist):
        if testContainTopic(topic, al):
            newabs.append(al)
            newintro.append(il)
    return newabs, newintro


def extractTopicSent(abslist, introlist):
    newabs = []
    newintro = []
    for al, il in zip(abslist, introlist):
        newal = []
        for a in al:
            if "we" not in a and "paper" not in a:
                newal.append(a)
            else:
                if len(newal) > 0:
                    newabs.append(newal)
                    newintro.append(il)
                    newal = []
                break
    return newabs, newintro


def combineSentence(sents):
    if len(sents) == 1:
        text = " ".join(sents[0])
    else:
        sentences = []
        for sent in sents:
            s = " ".join(sent)
            sentences.append(s)
        # sentences = [" ".join(sent) for sent in sents]
        text = " ".join(sentences)
    return text


'''----------------------------sentence similarity-----------------------------------'''


def JaccardSim(sent1, sent2):
    sent1 = set(sent1)
    sent2 = set(sent2)
    sim = float(len(sent1 & sent2)) / len(sent1 | sent2)
    return sim


def testSim(sents):
    similarity = []
    for i in range(len(sents) - 1):
        for j in range(i + 1, len(sents), 1):
            sim = JaccardSim(sents[i], sents[j])
            similarity.append((i, j, sim))
    return similarity


def cleanSummary(summ):
    summary = [word_tokenize(sent) for sent in sent_tokenize(summ)]
    similarity = testSim(summary)
    clean_ids = []
    for i, j, sim in similarity:
        if sim > 0.3:
            clean_ids.append(j)
    clean_ids = set(clean_ids)
    newSum = []
    for i in range(len(summary)):
        if i not in clean_ids:
            newSum.append(summary[i])

    if len(newSum) == 1:
        text = " ".join(newSum[0])
    else:
        sent = [" ".join(s) for s in newSum]
        text = " ".join(sent)
    return text


'''--------------------------------------summary--------------------------------------------'''


def sumAllSent(topic, fileName):
    summ = None
    # refs = readReference()
    refs = readReferenceTsv(fileName)
    newRefs, aiRefs, abslist, introlist = sentText(refs)
    # print(len(abslist))
    newabs, newintro = extractTopicRef(topic, abslist, introlist)
    # print(len(newabs))
    newabs, newintro = extractTopicSent(newabs, newintro)
    # print(len(newabs))
    sents = []
    for abstext in newabs:
        sents += abstext
    # sents = topicStartSentence(topic, sents)
    # print(sents)
    text = combineSentence(sents)
    summ = summarize(text, words=100)
    # print(summ)
    clean_sum = cleanSummary(summ)
    # print("\n")
    # print(clean_sum)
    return clean_sum


if __name__ == "__main__":
    # sumAllSent(["energy"],"tsvs/2742488.tsv")
    sumAllSent(["cache"], "tsvs/2830555.tsv")
    # sumAllSent(["imbalanced"],"tsvs/2907070.tsv")
    # introGen(["energy"],"tsvs/2742488.tsv")