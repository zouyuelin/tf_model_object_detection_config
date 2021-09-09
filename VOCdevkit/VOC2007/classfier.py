# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:15:53 2019

@author: Lenovo
"""

import os
import random

trainval_percent = 0.08
train_percent = 0.92
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets/Main'
if not os.path.exists("ImageSets"):
	os.mkdir("ImageSets")

if not os.path.exists('ImageSets/Main'):
	os.mkdir('ImageSets/Main')

total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)

train_num = int(num*train_percent)
train = random.sample(list,train_num )

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
