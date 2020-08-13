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



parser = argparse.ArgumentParser(add_help=True)
 
parser.add_argument('-s','--services_file_path', default='./services.csv',help='Services file path')
parser.add_argument('-p','--projects_file_path', default='./public_test.csv',help='Projects file path')
parser.add_argument('-c','--companies_file_path', default='./private_test.csv',help='Companies file path')
parser.add_argument('-o','--output_file_path', default='./results.csv',help='Output file path')
args = parser.parse_args()




cuda=True



bert_model='SciBert'
 
#df_project=pd.read_csv('public_fiverr.csv',encoding = "ISO-8859-1")
#df_coompany=pd.read_csv('private_fiverr.csv',encoding = "ISO-8859-1")
#df_service=pd.read_csv('services.csv',encoding = "ISO-8859-1")

df_project=pd.read_csv(args.projects_file_path,encoding = "ISO-8859-1") 
df_company=pd.read_csv(args.companies_file_path,encoding = "ISO-8859-1") 
df_service=pd.read_csv(args.services_file_path,encoding = "ISO-8859-1") 



neurons=768
stopwords=False
dynamic=True
static=False
folder_path='results/'
cased=False
cuda=False
names=['company_id','application_id','abstract_or_companyDescription','services','Sci-BERT_score']
model='base'
resultsDF = pd.DataFrame(columns=names)

for index,project in df_project.iterrows():
     
    data_input=[]
    project_desc=(str(project['abstract_text']))
    project_id= project['application_id']
    data_input.append(project_desc)
    file_name='nreq.txt'
     
    bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
     
    
    emb_project=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
    
    X=numpy.array(emb_project)
    X=X[:,1:]
    
    for ind,service in df_service.iterrows():
       
        data_input=[]
        service=str(service['services'])
        data_input.append(service)
        
         
         
        file_name="noreq.txt"
        
        
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
        emb_service=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
        
        
        Y=numpy.array(emb_service)
        if len(Y)!=0:
            Y=Y[:,1:]
        
            resultsDF.loc[len(resultsDF.index)]=['None',project_id,project_desc,service,sum(sum(metrics.pairwise.cosine_similarity(X,Y))) ]
print('Calculations of abstract done....')        
for index,comp in df_company.iterrows():
     
    data_input=[]
    comp_desc=(str(comp['description']))
    comp_id=(str(comp['company_id']))
    data_input.append(comp_desc)
    file_name='nreq.txt'
     
    bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
     
    
    emb_project=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
    
    X=numpy.array(emb_project)
    X=X[:,1:]
    
    for ind,service in df_service.iterrows():
       
        data_input=[]
        service=str(service['services'])
        data_input.append(service)
        
         
         
        file_name="noreq.txt"
        
        
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
        emb_service=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
        
        
        Y=numpy.array(emb_service)
        if len(Y)!=0:
            Y=Y[:,1:]
        
            resultsDF.loc[len(resultsDF.index)]=[comp_id,'None',comp_desc,service,sum(sum(metrics.pairwise.cosine_similarity(X,Y))) ]
        


        

print('Calculations of Companies done....') 
print('Writing results file')
resultsDF=resultsDF.sort_values(by='Sci-BERT_score', ascending=False)     
 
resultsDF.to_csv(args.output_file_path,index=False)


      
      
