from PyPDF2 import PdfFileReader
import pandas as pd
import numpy as np
import os
from itertools import repeat

base_path = r'C:\Personal\BASF\CodeChallenge'
dirs = os.listdir(base_path)
dirs.pop(0)
paths=[]
for dir in dirs:
    paths.append(os.listdir(base_path +'/'+ dir))

# select only Rheovis
paths=paths[0]
dirs=dirs[0]


def get_content(path):
    pdfobj = open(os.path.join(base_path, dirs,path), 'rb')
    pdf = PdfFileReader(pdfobj)
    l=[]
    for i in range(0, pdf.getNumPages()):
        l.append(pdf.getPage(i).extractText())
    return ''.join(l)

# load qas dat
rep =[4,3,1,8]
qas_dat = pd.read_csv(r'C:\Personal\BASF\CodeChallenge\Code\dat_pd_ctxt.txt', sep=";")
ans_list=[]
for i in range(qas_dat.shape[0]):
    ans=dict(text=qas_dat.iloc[i,2], answer_start= qas_dat.iloc[i, 3])
    ans_list.append(ans)

qas_list=[]
for i in range(qas_dat.shape[0]):
    qas=dict(id = qas_dat.iloc[i,4], question= qas_dat.iloc[i,1], answer=[ans_list[i]])
    qas_list.append(qas)
# elements are added to qas_fin [:4], [4:7]
qas_fin=[]
qas_fin.append(qas_list[:4])
qas_fin.append(qas_list[4:7])
qas_fin.append(qas_list[7])
qas_fin.append(qas_list[8:16])

#ctxt_list=[]
#for path in paths:
#    ctxt= get_content(path)
#    ctxt_list.append(ctxt)

ctxt_list=[]
for i in range(qas_dat.shape[0]):
    ctxt_list.append(qas_dat['context'][i])

context=[]
rep =[4,3,1,8]

train= []
for i in range(len(ctxt_list)):
    train.append(dict(context=ctxt_list[i], qas= qas_list[i]))

# import bert from transformers
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# training
# Apply the tokenizer to the input text, treating them as a text-pair.
def train_bert(ques,answ):
    # tokenize questions and answer texts
    input_ids= tokenizer.encode(ques,answ,padding=True,truncation=True)
    # first SEP token in input_ids
    sep_index= input_ids.index(tokenizer.sep_token_id)
    # number of segment A tokens including SEP tokens itself
    num_seg_a = sep_index+1 # +1 is for the [SEP] token
    # remaing ones which are seg B
    num_seg_b= len(input_ids)- num_seg_a
    # Construcing a list of 0's and 1's
    segment_ids= [0]*num_seg_a +[1]*num_seg_b
    #segment_id for every input token
    assert len(segment_ids)== len(input_ids) # make sure that the divided segment ids are qual to toal num of input segments

    start_score, end_score= model(torch.tensor([input_ids]),  # tokens representing input text
                                  token_type_ids= torch.tensor([segment_ids]))  # seg ids differentiating question from answer

    # finding tokens with highest start and end score
    ans_start= torch.argmax(start_score)
    ans_end= torch.argmax(end_score)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # combining tokens in the answer and printing
    answer= ' '.join(tokens[ans_start:ans_end+1])

    # cleaning the answers
    answer= tokens[ans_start]

    # select remaining ans tokens and join the with whitespaces
    for i in range(ans_start+1, ans_end+1):
        # If subword token, combine with prev
        if tokens[i][0:2]== '##':
            answer+= tokens[i][2:]
        else:
            answer+= ' '+tokens[i]
    return start_score, end_score,answer

for i in range(len(train)):
        print('Question {}'.format(train[i]['qas']['question']))
        print('Answer {}'.format(train_bert(train[i]['qas']['question'], train[i]['context'])[2]))





