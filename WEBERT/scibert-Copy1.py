import pandas as pd
import argparse

from utils import create_fold
import csv
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from WEBERT import BERT, BETO, SciBERT
from sklearn import metrics
import numpy


#bert_model='SciBert'
#bert_model='Beto'
bert_model='Bert'
df_project=pd.read_csv('public_fiverr.csv',encoding = "ISO-8859-1")
df_coompany=pd.read_csv('private_fiverr.csv',encoding = "ISO-8859-1")
df_service=pd.read_csv('services.csv',encoding = "ISO-8859-1")
neurons=768
stopwords=False
dynamic=True
static=False
folder_path='bert_embeddings/Dynamic/'
cased=False
cuda=False
names=['abstract','services','Sci-BERT']
model='base'
resultsDF = pd.DataFrame(columns=names)

for index,project in df_project.iterrows():
     
    data_input=[]
    project=(str(project['abstract_text']))
    data_input.append(project)
    file_name='zarnish.txt'
    if bert_model=='SciBert':
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
    elif bert_model=='Bert':
        bert=BERT(data_input,file_name, language='english', stopwords=stopwords, 
                                      model=model, cased=cased, cuda=cuda)
    elif bert_model=='Beto':
        bert=BETO(data_input,file_name, stopwords=stopwords, model=model, cased=cased, cuda=cuda)
    
    emb_project=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
    
    X=numpy.array(emb_project)
    X=X[:,1:]
    
    for ind,service in df_service.iterrows():
        if ind>100:
            break
            print(ind)
        data_input=[]
        service=str(service['services'])
        data_input.append(service)
        
         
         
        file_name="mustafa.txt"
        
        
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
        emb_service=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
        
        
        Y=numpy.array(emb_service)
        if len(Y)!=0:
            Y=Y[:,1:]
        
            resultsDF.loc[len(resultsDF.index)]=[project,service,sum(sum(metrics.pairwise.cosine_similarity(X,Y))) ]
        
    break
print('One project done successfully')
resultsDF.to_csv('serv-abstract.csv')


      
      
