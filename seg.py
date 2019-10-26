# -*- coding:utf-8 -*-
import jieba
import json
import jieba.analyse
import jieba
import jieba.posseg as pseg
from gensim import corpora, models, similarities
import gensim
import re
import logging
import csv

dic={}
csv_reader=csv.reader(open('train.csv', encoding='utf-8'))
for row in csv_reader:
    dic[row[0]]=row[1]

r1 = u'[a-zA-Z0-9’#$%&()*+-<=>@★▲\d+…【】[\\]^_{|}~]+'
i=0
with open('train.json', encoding='utf-8') as f:
    for line in f:
        result = ""
        i = i + 1
        if i%1000==0:
            print(i)

        d = json.loads(line)
        id=d['id']
        fcon=d['content']

        if id in dic.keys():
             sco = dic[id]
        else:
             continue
        tem = re.sub(r1, '', fcon)

        sen=jieba.cut(tem)
        for word in sen:
            result=result+" "+word
        path="train_se/"+sco+"/"+id+".txt"
        with open(path, 'a', encoding='utf-8') as fp:
            fp.write(result)
