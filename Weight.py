# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:57:46 2016

@author: ssits
"""

import numpy as np
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def my_tok(s):
    return s.split()

'''
Example of using TfidfVectorizer
Ref: http://stackoverflow.com/questions/23792781/tf-idf-feature-weights-using-sklearn-feature-extraction-text-tfidfvectorizer
'''
str_text1='ทั้งนี้ ท่า ทา ท้า ปกติ แล้ว เวลา ขาย เพชร หรือ เอา สินค้า ให้ ลูกค้า ดู นั้น เรา จะ ดู อยู่ แล้ว ว่า เอา สินค้า ให้ ดู กี่ ชิ้น แต่ ใน กรณี นี้ คนร้าย ไม่ ได้ ขโมย สินค้า ที่ เอา มา ให้ ดู แต่ กลับ เอื้อม มือ ไป หยิบ สินค้า ที่ อยู่ ใน ตู้ แทน'.decode('utf8')
str_text2='ถ้า ท้อ เป็น เพียง ถ่าน ถ้า ผ่าน จึง เป็น เพชร'.decode('utf8')

#myset = set(str_text1.split() + str_text2.split())

vect = TfidfVectorizer(
                        preprocessor=None,
                        analyzer="word",
                        use_idf=True, # utiliza o idf como peso, fazendo tf*idf
                        norm=None, # normaliza os vetores
                        smooth_idf=False, #soma 1 ao N e ao ni => idf = ln(N+1 / ni+1)
                        sublinear_tf=False, #tf = 1+ln(tf)
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        strip_accents=None, # retira os acentos
                        vocabulary=None,
                        ngram_range=(1,1),               
                        tokenizer=my_tok
             )
             
             
corpus_tf_idf = vect.fit_transform([str_text1,str_text2]) 
idf = vect.idf_
vol=vect.vocabulary_
dict_out= dict(zip(vect.get_feature_names(), idf))

print "--- Start tf-idf ----"
for i in dict_out:
    print i,dict_out[i]
    
print "--- End ----"     
    
    
    
    
'''
Example of using CountVectorizer
Ref: http://stackoverflow.com/questions/22920801/can-i-use-countvectorizer-in-scikit-learn-to-count-frequency-of-documents-that-w
''' 
cv = CountVectorizer(         
                        analyzer="word", 
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        strip_accents=None,
                        vocabulary=None,
                        ngram_range=(1,1), 
                        preprocessor=None,              
                        tokenizer=my_tok
             )
             
lst_values=cv.fit_transform([str_text1,str_text2]).toarray()
vol=cv.vocabulary_
lst_keys=cv.get_feature_names()

myarray = np.asarray(lst_keys)
myarray = np.transpose(myarray)

lst_count = [np.transpose(myarray),lst_values[0],lst_values[1]]
lst_count = np.asarray(lst_count)

file = codecs.open('/home/ssits/SM_nltk_git/count_word.csv', "w", "utf-8")

for i in range(0,len(lst_count)):
    text_line=""
    for n in range(0,len(lst_count[i])):
        print lst_count[i][n]
        text_line=text_line + lst_count[i][n] + " "
    file.write(text_line.strip() + "\n")
    
file.close()

