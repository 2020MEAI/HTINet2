# -- coding:UTF-8
import numpy as np 
import torch.utils.data as data
import torch
import math

def readD(set_matrix,num_):
    herb_d=[] 
    for i in range(num_): 
        len_set=1.0/(len(set_matrix[i])+1)
        herb_d.append(len_set)
    return herb_d

def readTrainSparseMatrix(set_matrix,h_d,g_d,is_herb):
    herb_genes_matrix_i=[]
    herb_genes_matrix_v=[]
    if is_herb:
        d_i=h_d
        d_j=g_d
    else:
        d_i=g_d
        d_j=h_d
    for i in set_matrix:
        len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            herb_genes_matrix_i.append([i,j]) 
            d_i_j=np.sqrt(d_i[i]*d_j[j]) 
            herb_genes_matrix_v.append(d_i_j)
    herb_genes_matrix_i=torch.cuda.LongTensor(herb_genes_matrix_i)
    herb_genes_matrix_v=torch.cuda.FloatTensor(herb_genes_matrix_v)
    return torch.sparse.FloatTensor(herb_genes_matrix_i.t(), herb_genes_matrix_v)

def metrics_loss(model, test_val_loader_loss, batch_size): 
    loss_sum=[]
    for herb, gene_i, gene_j in test_val_loader_loss:
        herb = herb.cuda()
        gene_i = gene_i.cuda()
        gene_j = gene_j.cuda()
        prediction_i, prediction_j,loss = model(herb, gene_i, gene_j) 
        loss_sum.append(loss.item())  
    test_val_loss=round(np.mean(loss_sum),4)
    return test_val_loss

def hr_ndcg(indices_sort_top,index_end_i,top_k): 
    hr_topK=0
    ndcg_topK=0

    ndcg_max=[0]*top_k
    temp_max_ndcg=0
    for i_topK in range(top_k):
        temp_max_ndcg+=1.0/math.log(i_topK+2)
        ndcg_max[i_topK]=temp_max_ndcg

    max_hr=top_k
    max_ndcg=ndcg_max[top_k-1]
    if index_end_i<top_k:
        max_hr=(index_end_i)*1.0
        max_ndcg=ndcg_max[index_end_i-1] 
    count=0
    for gene_id in indices_sort_top:
        if gene_id < index_end_i:
            hr_topK+=1.0
            ndcg_topK+=1.0/math.log(count+2) 
        count+=1
        if count==top_k:
            break

    hr_t=hr_topK/max_hr
    ndcg_t=ndcg_topK/max_ndcg  
    return hr_t,ndcg_t

def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_gene=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):
        super(BPRData, self).__init__()

        self.num_gene = num_gene
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_gene=set(range(num_gene)) 

    def ng_sample(self):
        self.features_fill = []
        a = list()
        for herb_id in self.train_dict: 
            positive_list=self.train_dict[herb_id] 
            all_positive_list=self.all_rating[herb_id] 
            a.append(len(positive_list)) 
            for gene_i in positive_list:
                for t in range(self.num_ng): 
                    gene_j=np.random.randint(self.num_gene)
                    while gene_j in all_positive_list:
                        gene_j=np.random.randint(self.num_gene)
                    self.features_fill.append([herb_id,gene_i,gene_j])

    def __len__(self):  
        return self.num_ng*self.data_set_count

    def __getitem__(self, idx):
        features = self.features_fill  
        # print("idx")
        # print(idx)
        # print(len(features))
        herb = features[idx][0]
        # print("herb")
        # print(herb)
        gene_i = features[idx][1]
        gene_j = features[idx][2]
        return herb, gene_i, gene_j

  
class resData(data.Dataset):
    def __init__(self,train_dict=None,batch_size=0,num_gene=0,all_pos=None):
        super(resData, self).__init__() 
        self.train_dict = train_dict 
        self.batch_size = batch_size
        self.all_pos_train=all_pos 

        self.features_fill = []
        for herb_id in self.train_dict:
            self.features_fill.append(herb_id)
        self.set_all=set(range(num_gene))
   
    def __len__(self):  
        return math.ceil(len(self.train_dict)*1.0/self.batch_size)
         

    def __getitem__(self, idx): 
        
        herb_test=[]
        gene_test=[]
        split_test=[]
        for i in range(self.batch_size):
            index_my=self.batch_size*idx+i 
            if index_my == len(self.train_dict):
                break   
            herb = self.features_fill[index_my]
            gene_i_list = list(self.train_dict[herb])
            gene_j_list = list(self.set_all-self.all_pos_train[herb])
            u_i=[herb]*(len(gene_i_list)+len(gene_j_list))
            herb_test.extend(u_i)
            gene_test.extend(gene_i_list)
            gene_test.extend(gene_j_list)  
            split_test.append([(len(gene_i_list)+len(gene_j_list)),len(gene_j_list)]) 
           
        return torch.from_numpy(np.array(herb_test)), torch.from_numpy(np.array(gene_test)), split_test
    
    