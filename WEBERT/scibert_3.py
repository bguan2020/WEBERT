import pandas as pd
import argparse
import re
import unicodedata

from utils import create_fold
import csv
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from WEBERT import BERT, BETO, SciBERT
from sklearn import metrics
import numpy
import pickle
import timeit



def remove_accents(text):
    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
    no_accents_text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
    return no_accents_text.replace('đ', 'd')


def clean_text(text):
    pattern = pattern = re.compile(r'[^áàảãạâấầẩẫậăẵẳắằặđéèẻẽẹêếềểễệíìịỉĩóòõỏọôốồổộỗơớờởỡợúùũủụưứừửữựýỳỷỹỵ\sa-z_]')
    return re.sub(pattern, ' ', text)





start = timeit.default_timer()

parser = argparse.ArgumentParser(add_help=True)
 
parser.add_argument('-s','--services_file_path', default='./services.csv',help='Services file path')
parser.add_argument('-p','--projects_file_path', default='./public_fiverr.csv',help='Projects file path')
parser.add_argument('-c','--companies_file_path', default='./private_fiverr.csv',help='Companies file path')
parser.add_argument('-o','--output_file_path', default='./results.csv',help='Output file path')
args = parser.parse_args()




cuda=True



bert_model='SciBert'
 
#df_project=pd.read_csv('public_fiverr.csv',encoding = "ISO-8859-1")
#df_coompany=pd.read_csv('private_fiverr.csv',encoding = "ISO-8859-1")
#df_service=pd.read_csv('services.csv',encoding = "ISO-8859-1")

df_project=pd.read_csv(args.projects_file_path,encoding = "ISO-8859-1").head(1000)
df_company=pd.read_csv(args.companies_file_path,encoding = "ISO-8859-1").head(1)
df_service=pd.read_csv(args.services_file_path,encoding = "ISO-8859-1")

proj_embs_file = os.path.join(os.getcwd(), 'db', 'proj_embs.db')
comp_embs_file = os.path.join(os.getcwd(), 'db', 'comp_embs.db')



proj_computed_embs = {}
if os.path.isfile(proj_embs_file):
    with open(proj_embs_file , mode='rb') as f:
        proj_computed_embs= pickle.load(f)


comp_computed_embs = {}
if os.path.isfile(comp_embs_file):
    with open(comp_embs_file , mode='rb') as f:
        comp_computed_embs= pickle.load(f)



neurons=768
stopwords=False
dynamic=True
static=False
folder_path='results/'
cased=False
cuda=False
names=['company_id','application_id','abstract_or_companyDescription','keywords','proj_title/company_name','services','Sci-BERT_score']
model='base'
resultsDF = pd.DataFrame(columns=names)
counter=0
for index,project in df_project.iterrows():
     
    data_input=[]
    project_desc=(str(project['abstract_text']))
    project_desc=clean_text(project_desc)
    project_desc=remove_accents(project_desc)
    project_id= project['application_id']
    data_input.append(project_desc)
    file_name='nreq.txt'
    counter+=1
    print('Calculating embeddings for abstract number ',counter)
    if project_id in proj_computed_embs:

        X=proj_computed_embs[project_id]
        #print('||||||||||||||||||||||||||||||||||||')
    else:

     
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
     
    
        emb_project=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
    
        X=numpy.array(emb_project)
        X=X[:,1:]
        
        proj_computed_embs[project_id]=X
    
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
        
            resultsDF.loc[len(resultsDF.index)]=['None',project_id,project_desc,'','',service,sum(sum(metrics.pairwise.cosine_similarity(X,Y))) ]
     
print('Calculations of abstract done....')
with open(proj_embs_file, mode='wb') as f:
    pickle.dump(proj_computed_embs, f)
print(df_company.head())
for index,comp in df_company.iterrows():
     
    data_input=[]
    comp_desc=(str(comp['description']))
    comp_desc=clean_text(comp_desc)
    comp_desc=remove_accents(comp_desc)
    comp_id=(str(comp['company_id']))
    data_input.append(comp_desc)
    file_name='nreq.txt'
    if comp_id in comp_computed_embs:
        X=comp_computed_embs[comp_id]
    else:
     
        bert=SciBERT(data_input,file_name, stopwords=stopwords,cased=cased, cuda=cuda)
     
    
        emb_project=bert.get_bert_embeddings(folder_path, dynamic=dynamic, static=static)
    
        X=numpy.array(emb_project)
        X=X[:,1:]
        
        comp_computed_embs[comp_id]=X
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
        
            resultsDF.loc[len(resultsDF.index)]=[comp_id,'None',comp_desc,'','',service,sum(sum(metrics.pairwise.cosine_similarity(X,Y))) ]
        


        

print('Calculations of Companies done....')
with open(comp_embs_file, mode='wb') as f:
    pickle.dump(comp_computed_embs, f) 
print('Writing results file')
resultsDF=resultsDF.sort_values(by='Sci-BERT_score', ascending=False)     
 
resultsDF.to_csv(args.output_file_path,index=False)
stop = timeit.default_timer()

print('Total execution Time: ', stop - start) 

      
      
