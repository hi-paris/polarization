import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Data
df_programmes = pd.read_csv('programmes_processed.csv', sep=';', encoding='utf-8')
df_articles = pd.read_csv('articles_processed.csv', sep=';', encoding='utf-8')
print(df_programmes.columns)
print(df_articles.columns)

df_bert= df_articles

programmes_corr = {
0 : 'EDU-UDF (Federal Democratic Union of Switzerland, right-wing)',
1: 'PEV (Evangelical People\'s Party , center)',
2 : 'SP (Social Democratic Party, left-wing)',
3 : 'UDC-SVP (Swiss People\'s Party, right-wing)',
4 : 'Verts (Green Party, left-wing)',
5 :'Die Mitte (The Centre, center)',
6 :'PLR-FDP (The Liberals, center-right)'
}

# GPUs
os.environ["CUDA_VISIBLE_DEVICES"] ="1,2"

# Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
model.to('cuda')
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Bert_inf(model,tokenizer,text,device):
  """
  vectorize text with Bert CLS
  """
  
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,max_length=100, add_special_tokens = True).to(device)
  outputs = model(**inputs)
  with torch.no_grad():
    last_hidden_states = outputs.last_hidden_state.cpu()
    del outputs
  return last_hidden_states[:,0]


def to_bert_vec_programmes(dataframe):
  """
  return the list of vectors
  """
  l=list()
  for i in tqdm(range(len(dataframe))):
    torch.cuda.empty_cache()
    l.append(Bert_inf(model,tokenizer,dataframe.iloc[:, 2].values.tolist()[i],device))
    
  return l

def to_bert_vec_articles(text):
  """
  return the list of vectors
  """
  l=list()
  torch.cuda.empty_cache()
  l.append(Bert_inf(model,tokenizer,text,device))
    
  return l

# programmes
l_prog = to_bert_vec_programmes(df_programmes)
document_embeddings_prog=torch.cat(l_prog,dim=0)

# function
for i in tqdm(range(len(df_bert['text_processed']))):
  vec = Bert_inf(model,tokenizer,df_bert['text_processed'][i],device)
  sim_matrix = cosine_similarity(vec, document_embeddings_prog)
  similar_ix=np.argsort(cosine_similarity(vec, document_embeddings_prog))[0][::-1][:7]
  l=list()
  for ix in similar_ix:
    l.append([programmes_corr[ix], cosine_similarity(vec,document_embeddings_prog)[::-1][0][ix]])
  
  df_bert.loc[i,'bert_pred1']=l[0][0]
  df_bert.loc[i,'bert_sim1']=l[0][1]

  df_bert.loc[i,'bert_pred2']=l[1][0]
  df_bert.loc[i,'bert_sim2']=l[1][1]

  df_bert.loc[i,'bert_pred3']=l[2][0]
  df_bert.loc[i,'bert_sim3']=l[2][1]

  df_bert.loc[i,'bert_pred4']=l[3][0]
  df_bert.loc[i,'bert_sim4']=l[3][1]

  df_bert.loc[i,'bert_pred5']=l[4][0]
  df_bert.loc[i,'bert_sim5']=l[4][1]

  df_bert.loc[i,'bert_pred6']=l[5][0]
  df_bert.loc[i,'bert_sim6']=l[5][1]

  df_bert.loc[i,'bert_pred7']=l[6][0]
  df_bert.loc[i,'bert_sim7']=l[6][1]

df_bert.to_csv('/home/infres/ext-1227/articles_bert.csv',sep=';',encoding='utf-8')