import pandas as pd
from sklearn import metrics

t1=pd.read_csv('test.csv')
t2=pd.read_csv('test2.csv')


print(t1.shape)


X=t1.to_numpy()
Y=t2.to_numpy()


print(sum(sum(metrics.pairwise.cosine_similarity(X[:,1:],Y[:,1:]))))


