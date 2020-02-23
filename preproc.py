'''
本文件包含了预处理训练和测试数据的代码
为了提高效率，我们会将部分中间数据存储至本地
WRITE为True时，我们执行存储和读取功能，
否则，只执行读取功能
'''
import jieba
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from gensim.models import Word2Vec

train_data_path = './dataset/train.data'
train_target_path = './dataset/train.solution'
train_target_transform_path = './intermediate_data/train.solution.T'
train_tfidf_path = './intermediate_data/train_tfidf.npz' 
test_tfidf_path = './intermediate_data/test_tfidf.npz'
train_vec_path = './intermediate_data/train_vec.npy' 
test_vec_path = './intermediate_data/test_vec.npy'
#train_data_path = './little_dataset/train.data'
#train_target_path = './little_dataset/train.solution'
#train_target_transform_path = './little_intermediate_data/train.solution.T'
#train_tfidf_path = './little_intermediate_data/train_tfidf.npz' 
#test_tfidf_path = './little_intermediate_data/test_tfidf.npz'
#train_vec_path = './little_intermediate_data/train_vec.npy' 
#test_vec_path = './little_intermediate_data/test_vec.npy'
test_data_path = './dataset/test.data'
#stop_words_path = './stopwords.data'
stop_words_path = './new_stopwords.data'
emoji_path = './dataset/emoji.data'

WRITE = False
MAX_FEATURE = 80000
DIMENTION = 200

#根据分词结果生成tfidf矩阵，存入另一个文件
def tfidf_word_list(write):
    if write == True:
        eng_num = re.compile('[0-9a-zA-Z]+')
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_word_list = [' '.join(jieba.cut(eng_num.sub('', line).strip())) for line in f] 
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_word_list = [' '.join(jieba.cut(eng_num.sub('', line).strip())) for line in f]
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            stop_words = [line.strip() for line in f] 
#        vectorizer = TfidfVectorizer(
#                token_pattern=r"(?u)\b\w+\b", max_features=MAX_FEATURE, stop_words=stop_words)  
        vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b", stop_words=stop_words)  
        train_X = vectorizer.fit_transform(train_word_list)
        test_X = vectorizer.transform(test_word_list)
        scipy.sparse.save_npz(train_tfidf_path, train_X)
        scipy.sparse.save_npz(test_tfidf_path, test_X)
    train_X = scipy.sparse.load_npz(train_tfidf_path)
    test_X = scipy.sparse.load_npz(test_tfidf_path)
    return train_X, test_X

#根据分词结果生成词向量矩阵，存入另一个文件
def word_vec_list(write):
    if write == True:
        with open(train_data_path, 'r', encoding='utf-8') as f:
           train_word_list = [' '.join(jieba.cut(line)).strip() for line in f]  
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_word_list = [' '.join(jieba.cut(line)).strip() for line in f]  
        train_word_list = [line.split() for line in train_word_list]    
        test_word_list = [line.split() for line in test_word_list] 
        model = Word2Vec(train_word_list+test_word_list, min_count=1, size=DIMENTION)
        train_X = np.zeros((len(train_word_list),DIMENTION))
        test_X = np.zeros((len(test_word_list),DIMENTION))        
        for i in range(len(train_word_list)):
            train_X[i] = np.mean(model[train_word_list[i]],axis=0)
        for i in range(len(test_word_list)):
            test_X[i] = np.mean(model[test_word_list[i]],axis=0)    
        model.save("word2vec.model")
        np.save(train_vec_path,train_X)
        np.save(test_vec_path,test_X)
    train_X = np.load(train_vec_path)
    test_X = np.load(test_vec_path)
    return train_X, test_X


#将emoji的汉字形式替换成序号并写入文件中,若文件已存在，则do nothing
def transform_emoji_file(write):
    if write == True:
        with open(train_target_path,'r',encoding='utf-8') as train_target_file:
            with open(train_target_transform_path,'w+',encoding='utf-8') as target_transform_file:
               emoji = pd.read_table(emoji_path,header=None).values
               for line in train_target_file:
                   for e in emoji:
                       pattern = '{'+e[1]+'}'
                       searchObj = re.search(pattern, line, re.M|re.I)
                       if searchObj:
                           target_transform_file.write(str(e[0])+'\n')
                           break 
    with open(train_target_transform_path, 'r', encoding='utf-8') as f:
        y = [int(line.strip()) for line in f]  
    return np.array(y)

#预处理
def ret_data(write = WRITE):
#    train_X, test_X = tfidf_word_list(write)
    train_X, test_X = word_vec_list(write)
    train_y = transform_emoji_file(write)
#    #卡方检验
#    chi = SelectKBest(chi2, k=MAX_FEATURE)
#    train_X = chi.fit_transform(train_X, train_y)
#    test_X = chi.transform(test_X)
    return train_X, test_X, train_y