'''
本文件包含了训练分类器和利用分类器进行预测的代码
同时，将预测结果保存至本地文件中也在这里进行
'''

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import preproc

result_path = './submission.csv' 

#由于ovo在预测时需要占用大量内存，笔记本吃不消，故要分批预测
OVO_PRED_BATCH_SIZE = 50000
SPLIT = 700000

def predict_with_batch(X_test, clf, batch_size = OVO_PRED_BATCH_SIZE):
    n = X_test.shape[0]
    y_pred = np.zeros(n)
    iter_times = n//batch_size
    for i in range(iter_times):
        y_pred[i*batch_size:(i+1)*batch_size] = clf.predict(
                X_test[i*batch_size:(i+1)*batch_size])
    if n % batch_size != 0:
        y_pred[iter_times*batch_size:] = clf.predict(X_test[iter_times*batch_size:])
    return y_pred

def validation_by_clf(clf,X_train,y_train,split=SPLIT):
    X_valid, y_valid = X_train[split:], y_train[split:]
    X_s_train, y_s_train = X_train[:split], y_train[:split]
    clf.fit(X_s_train,y_s_train)
    y_pred = predict_with_batch(X_valid, clf)
    print(f1_score(y_valid,y_pred,average='micro'))
    
def solve_by_clf(clf, X_train,y_train):
    clf.fit(X_train,y_train)
    return clf

#使用朴素贝叶斯进行分类
#当max_feature为:
#不设置时：f1_score:0.16015
#10000时，f1_score:0.15274
#30000时，f1_score:0.16099
#50000时，f1_score:0.16432
#100000时，f1_score:0.16687
#120000时，f1_score:0.16666
#修改停词表后
#当max_feature为20000时，f1_score:0.16664726‬
#修改alpha为0.15后
#当max_feature为20000时，f1_score:0.16707126
def solve_by_NB(X_train,y_train):
    clf = MultinomialNB(alpha = 0.15)
    scores = np.mean(cross_val_score(clf, X_train, y_train, scoring='f1_micro', cv=5))
    print(scores)
    return solve_by_clf(clf, X_train,y_train)
 
#使用k最近邻进行分类(数据集太大，跑不动)
def solve_by_KNN(X_train,y_train):
    clf = KNeighborsClassifier()
    return solve_by_clf(clf, X_train,y_train)

#使用支持向量机进行分类
#当max_feature为:
#20000时，f1_score:0.16862
#50000时，f1_score:0.17199
#80000时, f1_score:0.17145
def solve_by_SVM(X_train,y_train):
#    clf = OneVsOneClassifier(LinearSVC(C=1,random_state=0))
    clf = OneVsOneClassifier(LinearSVC())
    validation_by_clf(clf,X_train,y_train)
    return solve_by_clf(clf, X_train,y_train)

#使用多层感知机进行分类
#改用词向量的算术平均值作为特征，因为神经网络适合训练稠密矩阵
#当词向量维度为50时，test_score:0.17064
#当词向量维度为100时，test_score:0.17340
#当词向量维度为200时，test_score:0.17470
def solve_by_MPL(X_train,y_train):
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 100), batch_size=256, max_iter=200,
        learning_rate_init=1e-3, learning_rate='adaptive',
        early_stopping=True, verbose=True)
    clf.fit(X_train, y_train)
    return clf

#使用多层感知机的BAGGING方法进行分类
#当词向量维度为200且基学习器个数为10时，test_score:0.17860
def solve_by_Bagging_MPL(X_train,y_train):
    base_clf = MLPClassifier(
        hidden_layer_sizes=(100, 100), batch_size=256, max_iter=200,
        learning_rate_init=1e-3, learning_rate='adaptive',
        early_stopping=True, verbose=True)
    clf = BaggingClassifier(
            base_estimator=base_clf, n_estimators=10,
            n_jobs=1, verbose=True)
    clf.fit(X_train, y_train)
    return clf

if __name__ == '__main__':
    X_train, X_test, y_train = preproc.ret_data()
#    clf = solve_by_NB(X_train,y_train)
#    clf = solve_by_SVM(X_train,y_train)
#    clf = solve_by_MPL(X_train,y_train)
#    clf = solve_by_Bagging_MPL(X_train,y_train)
#    joblib.dump(clf,'BagMpl.pkl')
    clf = joblib.load('BagMpl.pkl')
    y_pred = predict_with_batch(X_test, clf).astype(int)
    print(y_pred)
    y_pred = y_pred.reshape(len(y_pred),1)
    y_pred = np.insert(y_pred,0,values=np.arange(len(y_pred)).astype(int),axis=1)
    df = pd.DataFrame(y_pred, columns=['ID','Expected'])
    df.to_csv(result_path, index=False)