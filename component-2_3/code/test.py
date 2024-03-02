# -- coding:UTF-8 
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import time
import data_utils 
from numpy import loadtxt

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))

herb_num=563
gene_num=2106
factor_num=64
batch_size=2048*512
top_k=10
run_id="1"
print(run_id)

path_save_model_base='./model/'+'/s'+run_id
training_herb_set,training_gene_set,training_set_count = np.load('./data/dataset/training_set.npy',allow_pickle=True)
testing_herb_set,testing_gene_set,testing_set_count = np.load('./data/dataset/testing_set.npy',allow_pickle=True)   
herb_rating_set_all = np.load('./data/dataset/set_all.npy',allow_pickle=True).item()

h_d=data_utils.readD(training_herb_set,herb_num)
g_d=data_utils.readD(training_gene_set,gene_num)
d_i_train=h_d
d_j_train=g_d
sparse_u_i=data_utils.readTrainSparseMatrix(training_herb_set,h_d,g_d,True)
sparse_i_u=data_utils.readTrainSparseMatrix(training_gene_set,h_d,g_d,False)

class BPR(nn.Module):
    def __init__(self, herb_num, gene_num, factor_num,herb_gene_matrix,gene_herb_matrix,d_i_train,d_j_train):
        super(BPR, self).__init__()
        self.herb_gene_matrix = herb_gene_matrix
        self.gene_herb_matrix = gene_herb_matrix
        self.embed_herb = nn.Embedding(herb_num, factor_num)
        self.embed_gene = nn.Embedding(gene_num, factor_num) 

        # load the embeddings of the component 1
        # self.embed_herb.weight.data.copy_(torch.from_numpy(loadtxt("./data/embedding/herb_LINE.txt")))
        # self.embed_gene.weight.data.copy_(torch.from_numpy(loadtxt("./data/embedding/gene_LINE.txt")))

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_herb.weight, std=0.01)
        nn.init.normal_(self.embed_gene.weight, std=0.01)
        
    def forward(self, herb, gene_i, gene_j):    
        herbs_embedding=self.embed_herb.weight
        genes_embedding=self.embed_gene.weight  

        gcn1_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, genes_embedding) + herbs_embedding.mul(self.d_i_train))
        gcn1_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, herbs_embedding) + genes_embedding.mul(self.d_j_train))

        gcn2_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn1_genes_embedding) + gcn1_herbs_embedding.mul(self.d_i_train))
        gcn2_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn1_herbs_embedding) + gcn1_genes_embedding.mul(self.d_j_train))

        gcn3_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn2_genes_embedding) + gcn2_herbs_embedding.mul(self.d_i_train))
        gcn3_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn2_herbs_embedding) + gcn2_genes_embedding.mul(self.d_j_train))
       
        gcn_herbs_embedding= torch.cat((herbs_embedding,gcn1_herbs_embedding,gcn2_herbs_embedding,gcn3_herbs_embedding),-1)
        gcn_genes_embedding= torch.cat((genes_embedding,gcn1_genes_embedding,gcn2_genes_embedding,gcn3_genes_embedding),-1)
        
        return gcn_herbs_embedding, gcn_genes_embedding

test_batch=52
testing_dataset = data_utils.resData(train_dict=testing_herb_set, batch_size=test_batch,num_gene=gene_num,all_pos=training_herb_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0) 

model = BPR(herb_num, gene_num, factor_num,sparse_u_i,sparse_i_u,d_i_train,d_j_train)
model=model.to('cuda')
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)

print('--------testing--------')
count, best_hr = 0, 0
HR_best = 0
NDCG_best = 0

for epoch in range(0,350,1):
    model.train()   
    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    model.load_state_dict(torch.load(PATH_model)) 
    model.eval()     
    gcn_herbs_embedding, gcn_genes_embedding= model(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]))
    herb_e=gcn_herbs_embedding.cpu().detach().numpy()
    gene_e=gcn_genes_embedding.cpu().detach().numpy()
    all_pre=np.matmul(herb_e,gene_e.T) 
    HR, NDCG = [], [] 
    set_all=set(range(gene_num))  
    test_start_time = time.time()
    for u_i in testing_herb_set: 
        gene_i_list = list(testing_herb_set[u_i])
        index_end_i=len(gene_i_list)
        gene_j_list = list(set_all-training_herb_set[u_i]-testing_herb_set[u_i])
        gene_i_list.extend(gene_j_list) 
        pre_one=all_pre[u_i][gene_i_list] 
        indices=data_utils.largest_indices(pre_one, top_k)
        indices=list(indices[0])   
        hr_t,ndcg_t=data_utils.hr_ndcg(indices,index_end_i,top_k) 
        elapsed_time = time.time() - test_start_time 
        HR.append(hr_t)
        NDCG.append(ndcg_t)    
    hr_test=round(np.mean(HR),4)
    ndcg_test=round(np.mean(NDCG),4)           
    str_print_evl='epoch:'+str(epoch)+' time:'+str(round(elapsed_time,2))+'\t test'+' hr@'+str(top_k)+':'+str(hr_test)+' ndcg@'+str(top_k)+':'+str(ndcg_test) 
    print(str_print_evl)   

    if hr_test > HR_best:
        HR_best = hr_test
    if ndcg_test > NDCG_best:
        NDCG_best = ndcg_test

print("HR_best "+str(HR_best))
print("NDCG_best "+str(NDCG_best))