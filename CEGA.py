# coding=gbk
import copy
import math
import torch
torch.cuda.empty_cache()
import networkx as nx
import numpy as np
from scipy.sparse import *
from tools import structing, doubly_stochastic, \
    get_adj_matrix, cal_degree_dict, evaluate,get_top_k_acc,compute_average_mnc,sparse_sinkhorn_sims_trainable,evaluate
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
import time
from scipy import sparse 
import scipy
import os
import pickle
from sparse_search import test
import gc
from torch_scatter import scatter_max,scatter_add
import tensorly as tl
from numba import jit
import pandas as pd
import torch_scatter
import random
tl.set_backend('pytorch')
use_amp = True

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)






def get_hits(sim, N, top_k=(1, 5, 10)):
  sim_o = -sim[:,:]
  sim = sim_o.argsort(-1)
  top_lr = [0] * len(top_k)
  MRR_lr = 0
  for i in range(sim.shape[0]):
    rank = sim[i, :]
    rank_index = np.where(rank == i)[0][0]
    MRR_lr += 1/(rank_index + 1)
    for j in range(len(top_k)):
      if rank_index < top_k[j]:
        top_lr[j] += 1
  top_rl = [0] * len(top_k)
  MRR_rl = 0
  sim = sim_o.argsort(0)
  for i in range(sim.shape[1]):
    rank = sim[:,i]
    rank_index = np.where(rank == i)[0][0]
    MRR_rl += 1/(rank_index + 1)
    for j in range(len(top_k)):
      if rank_index < top_k[j]:
        top_rl[j] += 1
  print('For each left:')
  for i in range(len(top_lr)):
    print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / N * 100))
  print('MRR: %.3f' % (MRR_lr / sim.shape[0]))  
  print('For each right:')
  for i in range(len(top_rl)):
    print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / N * 100))
  print('MRR: %.3f' % (MRR_rl /sim.shape[-1]))
  





def logsoftmax(src,index,num_nodes):
  # 计算logsumexp
  out_max = scatter_max(src, index=index, dim=-1, dim_size=num_nodes)[0][index]
  src = src- torch.log(scatter_add((src-out_max).exp(), index, dim=-1, dim_size=num_nodes)[index] + 1e-16) - out_max
  return src


def sp_sinkhorn(src, row_index, col_index, size, tao=0.5):
     num_nodes = len(src)
     src = src/tao
     for i in range(10):
       src = logsoftmax(src,row_index,num_nodes)
       src = logsoftmax(src,col_index,num_nodes)
     src=src.exp()
     return  torch.sparse.FloatTensor(torch.stack([row_index,col_index]),
                                       src, torch.Size([size,size]))


def getIntraTransMatrix(g1,g2):
  N = len(g1) + len(g2)

  adj1 = nx.to_scipy_sparse_matrix(g1, nodelist=list(range(len(g1))))
  adj2 = nx.to_scipy_sparse_matrix(g2, nodelist=list(range(len(g2))))
  D1 = np.sum(adj1, axis=1).reshape(-1, 1)
  D2 = np.sum(adj2, axis=1).reshape(-1, 1)
  adj1_hat = adj1 / D1
  adj2_hat = adj2 / D2
  
  rows, cols = adj1_hat.nonzero() 
  
  data1 = adj1_hat[rows,cols].squeeze(axis=0)
  
  rows2,cols2 = adj2_hat.nonzero()
  data2 = adj2_hat[rows2,cols2].squeeze(axis=0)
  rows = np.hstack([rows, rows2+len(g1)])
  cols = np.hstack([cols, cols2+len(g1)])

  data = np.array(np.hstack([data1,data2])).flatten()

  N = len(g1)+len(g2)  
  P_TRANS = torch.sparse.FloatTensor(torch.LongTensor(np.stack([rows, cols])),
                              torch.FloatTensor(data), torch.Size([N,N]))
  adj1 = torch.sparse.LongTensor(torch.LongTensor(np.stack(adj1.nonzero())),
                          torch.FloatTensor(adj1.data))
  adj2 = torch.sparse.LongTensor(torch.LongTensor(np.stack(adj2.nonzero())),
                              torch.FloatTensor(adj2.data))
  return P_TRANS, adj1,adj2


def getInterTransMatrix(G1,G2,attribute,alpha,c,M,layers):
  N1 = len(G1)
  N = len(G1) +len(G2)
  
  G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, layers)
  G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, layers)
  struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2, degree_sim = \
        structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c, M)


  row2 = []
  col2 = []
  data2= []
  for key in struc_neighbor_sim1.keys():

      for index, neighbor in enumerate(struc_neighbor1[key]):
          row2.append(key)
          col2.append(neighbor + N1)
          data2.append(struc_neighbor_sim1[key][index])


  for key in struc_neighbor_sim2.keys():

      for index, neighbor in enumerate(struc_neighbor2[key]):
          row2.append(key + N1)
          col2.append(neighbor)

          data2.append(struc_neighbor_sim2[key][index])


  P_TRANS2 = torch.sparse.FloatTensor(torch.LongTensor([row2,col2]),
                                     torch.FloatTensor(data2), torch.Size([N,N]))
  return P_TRANS2


def cal_degree_dict(G, layers):
  percentiles = [0,25,50,75,100]
  G_list = [i for i in range(len(G))]
  Mat = []
  G_degree = G.degree()
  degree_dict = {}
  degree_dict[0] = {}
  for node in G_list:
    degree_dict[0][node] = {node}
  for i in range(1, layers + 1):
    degree_dict[i] = {}
    for node in G_list:
      neighbor_set = []
      for neighbor in degree_dict[i - 1][node]:
        neighbor_set += nx.neighbors(G, neighbor)
      neighbor_set = set(neighbor_set)
      for j in range(i - 1, -1, -1):
        neighbor_set -= degree_dict[j][node]
      degree_dict[i][node] = neighbor_set
  for i in range(layers + 1):
    sub_mat = []
    for node in sorted(G_list):
      if len(degree_dict[i][node]) == 0:
        degree_dict[i][node] = [0]
      sub_mat.append(np.log(np.percentile([G_degree[j] for j in degree_dict[i][node]], percentiles) + 1))
    
    sub_mat = np.array(sub_mat).reshape((-1,len(percentiles)))
    Mat.extend([sub_mat[:,layer].reshape((-1,1)) for layer in range(len(percentiles))])
  return Mat
  
  
def compute_struct_sim(G1,G2,attribute,alpha,c,layers):
  N1 = len(G1)
  N = len(G1) +len(G2)
  MatList1 = cal_degree_dict(G1, layers)
  MatList2 = cal_degree_dict(G2, layers)
  SIM = 0
  for mat1, mat2 in zip(MatList1,MatList2):
    SIM = SIM + manhattan_distances(mat1,mat2)
  if attribute is not None:
    SIM = c * SIM + attribute * (1 - c)
  SIM = np.exp(-alpha * SIM)
  return SIM
  
def getInterTransMatrixFast(G1,G2,alpha,c,M,layers,SIM):
  
  N = len(G1) + len(G2)

  G1_nodes = [i for i in range(len(G1))]
  G2_nodes = [i for i in range(len(G2))]
  
  
  j1 = np.argpartition(SIM, -M, axis=-1)[:,-M:].flatten()
  i1 = np.array([[i]*M for i in range(len(G1))]).flatten()
  values1 = SIM[i1,j1]
  
  SIM_inverse = SIM.T
  j2 = np.argpartition(SIM_inverse, -M, axis=-1)[:,-M:].flatten()
  i2 = np.array([[i]*M for i in range(len(G2))]).flatten()
  values2 = SIM_inverse[i2,j2]
  
  row = np.hstack([i1, i2+len(G1)])
  col = np.hstack([j1+len(G1), j2 ])
  indices = np.stack([row,col])
  values = np.hstack([values1,values2])
  
  out_sum = torch_scatter.scatter_sum(torch.FloatTensor(values),dim=-1,index=torch.LongTensor(row))
  src = torch.FloatTensor(values)/out_sum.gather(dim=-1,index=torch.LongTensor(row))
  
  


  P_TRANS2 = torch.sparse.FloatTensor(torch.LongTensor(indices),
                                    src, torch.Size([N,N]))

  
  return P_TRANS2
  
  
def cross_graph_MF(switching_prob, P_TRANS_INTRA,P_TRANS_INTER,N,device,window_size,b,size):
  s_prob = torch.sigmoid(switching_prob)
  switch_diag_intra = torch.diag(s_prob)
  switch_diag_inter = torch.diag(1 - s_prob)
  
  
  P_TRANS =  torch.sparse.mm(torch.transpose(P_TRANS_INTRA,0,1),switch_diag_intra) + torch.sparse.mm(torch.transpose(P_TRANS_INTER,0,1),switch_diag_inter)
  
  # 计算分解矩阵
  I = torch.eye(N).to(device)
  
  P_TRANS = torch.transpose(P_TRANS, 0, 1)
  P_TRANS = P_TRANS.to_sparse()
  for r in range(window_size):
      if r == 0:
         S1 = I
      else:
         S1 = S1 + I
      S1 = torch.sparse.mm(P_TRANS, S1)
  
  
  D_R = P_TRANS.detach()/0.5
  D_R = D_R.to_sparse()
  D_R.values = torch.exp(D_R.values())
  D_R = torch.sparse.sum(D_R,dim=-1)/torch.sparse.sum(D_R)
  list1 = [i for i in range(N)]
  D_R = torch.sparse.FloatTensor(torch.LongTensor([list1,
                    list1]).to(device), D_R.values().to(device),torch.Size([N,N]))
  
  
  
  S1 =  torch.sparse.mm(D_R, S1.T)
  S = S1 + S1.T
  S = S / (2 * window_size * b)
  S = torch.clamp(S, min=1e-12) 
  S = torch.log(S) # 不能+1
  
  U, sigma, V = torch.svd_lowrank(S.float(),q=size,niter=1) # faster
  return U,sigma,V
  
#def alignment_loss(A1_th_sparse,A2_th_sparse,DS,device,N1,N):
#  
#  A1DS = torch.sparse.mm(A1_th_sparse,DS)
#  A2DST = torch.sparse.mm(A2_th_sparse,DS.T)
#  
#  loss = F.mse_loss(A1DS, A2DST.T,reduction="sum") +\
#  F.mse_loss( DS@ DS.T, torch.eye(N1).float().to(device),reduction="sum")*2
#  loss = loss / N
#  return loss


def alignment_loss(A1_th_sparse,A2_th_sparse,DS,device,N1,N):
  
  A1 = A1_th_sparse.to_dense()
  A2 = A2_th_sparse.to_dense()
  loss = F.mse_loss(DS.T @ A1 @ DS, A2,reduction="sum") +\
                F.mse_loss(DS @ A2 @ DS.T, A1,reduction="sum")
  
  loss = loss / N
  return loss

def train(G1, G2,attr_sim,  lr = 0.5, layers = 3, alpha = 5 , c = 0.5 , window_size = 5 , b = 5 ,size =128, M = 10, steps = 50, device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")):

    s_time = time.time()
    N1 = nx.number_of_nodes(G1)
    N2 = nx.number_of_nodes(G2)
    N = N1 + N2
    print(device)


    switching_prob = [0.0 for i in range(N)]
    switching_prob = torch.tensor(switching_prob).float().to(device) 
    switching_prob.requires_grad = True

    optimizer = torch.optim.Adam([switching_prob], lr=lr)


    P_TRANS_INTRA, A1,A2 = getIntraTransMatrix(G1,G2)
    P_TRANS_INTRA=P_TRANS_INTRA.to(device)
    A1_th_sparse = A1.to(device)
    A2_th_sparse= A2.to(device)
    SIM = compute_struct_sim(G1,G2,attr_sim,alpha,c,layers)
    P_TRANS_INTER = getInterTransMatrixFast(G1,G2,alpha,c,M,layers,SIM).to(device)




    for step in range(steps):
 
      U,sigma,V = cross_graph_MF(switching_prob, P_TRANS_INTRA,P_TRANS_INTER,N,device,window_size,b,size)

      P = U[:N1,:] @ torch.diag(sigma) @ U[N1:,:].T
      DS = doubly_stochastic(P,0.5,10)
      
      loss = alignment_loss(A1_th_sparse,A2_th_sparse,DS,device,N1,N)
      
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      if (step+1) % 10== 0 or step ==0: # 20
        
      
        print("use_time:{}".format(time.time()-s_time))
        e = U @ torch.sqrt(torch.diag(sigma))
        e = e.detach().cpu().numpy()
      
        pair = [i for i in range(N1)] + [i+N1 for i in range(N1)]
      
        test_pair = np.array(pair).reshape((2,-1)).T
      
        test(test_pair,e,top_k=500,iteration=15)
    P = U[:N1,:] @ torch.diag(sigma) @ U[N1:,:].T

    DS = doubly_stochastic(P,0.5,10).detach().cpu().numpy()

    return DS
  
  
  
def batch_train(G1, G2, SIM, lr = 0.5, layers = 3, alpha = 5 , c = 0.5 , window_size = 5 , b = 5 ,size =128, M = 10, steps = 50, device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")):

    s_time = time.time()
    N1 = nx.number_of_nodes(G1)
    N2 = nx.number_of_nodes(G2)
    N = N1 + N2
    print(device)


    switching_prob = [0.0 for i in range(N)]
    switching_prob = torch.tensor(switching_prob).float().to(device) 
    switching_prob.requires_grad = True

    optimizer = torch.optim.Adam([switching_prob], lr=lr)


    P_TRANS_INTRA, A1,A2 = getIntraTransMatrix(G1,G2)
    P_TRANS_INTRA=P_TRANS_INTRA.to(device)
    A1_th_sparse = A1.to(device)
    A2_th_sparse= A2.to(device)

    P_TRANS_INTER = getInterTransMatrixFast(G1,G2,alpha,c,M,layers,SIM).to(device)



    for step in range(steps):
 
      U,sigma,V = cross_graph_MF(switching_prob, P_TRANS_INTRA,P_TRANS_INTER,N,device,window_size,b,size)

      P = U[:N1,:] @ torch.diag(sigma) @ U[N1:,:].T
      DS = doubly_stochastic(P,0.5,10)
      
      loss = alignment_loss(A1_th_sparse,A2_th_sparse,DS,device,N1,N)
      
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
     
    E = U @ torch.sqrt(torch.diag(sigma))
    E1 = E[:N1]
    E2 = E[N1:]
    P = E1@E2.T
    DS = doubly_stochastic(P,0.5,10).detach().cpu().numpy()

    return DS
  
  
def cal_rdd(G1,G2):
  D1 = np.array([G1.degree(i) for i in range(len(G1))])
  D2 = np.array([G2.degree(i) for i in range(len(G2))])
  D1 = D1.reshape((-1,1))
  D2 = D1.reshape((1,-1))
  #列向量
  y_cols = len(D2.flatten())
  x_rows = len(D1.flatten())
  
  x = D1.repeat(y_cols, axis=-1)
  y = D2.repeat(x_rows,axis= 0)
  rdd = (2*np.abs(x-y)/(x + y))+1
  rdd = 1.0/ rdd
  return rdd
  
def select_candidate_pairs(e1,e2, rdd,sim=None):
  if sim is None:
    row,col,sims = sparse_sinkhorn_sims_trainable(e1,e2,top_k=1)
  else:
    col = np.argmax(sim, axis=-1)
    row = [i for i in range(sim.shape[0])]
    sims = sim[row,col]
    
  sims = np.array(sims)
  sims = sims.flatten()
  rdd_v = rdd[row,col].flatten()
  pred = [(i,j,k) for i,j,k in zip(row,col,rdd_v*sims)]
  pred1 = sorted(pred,key=lambda x:x[2], reverse=True)
  
  if sim is None:
    row,col,sims = sparse_sinkhorn_sims_trainable(e2,e1,top_k=1)
  else:
    col = np.argmax(sim.T, axis=-1)
    row = [i for i in range(sim.shape[1])]
    sims = (sim.T)[row,col]

  sims = np.array(sims)
  sims = sims.flatten()
  pred = [(i,j,k) for i,j,k in zip(row,col,rdd_v*sims)]
  pred2 = sorted(pred,key=lambda x:x[2], reverse=True)
  pairs1 = [(item[0],item[1]) for item in pred1]
  pairs2 = [(item[0],item[1]) for item in pred2]
  candidate_pairs = []
  for item in pairs1:
    x,y  = item
    if (y,x) in pairs2:
      candidate_pairs.append((x,y))
  return candidate_pairs
  
def initial_cega(G1, G2,attr_sim, lr = 0.5, layers = 3, alpha = 5 , c = 0.5 , window_size = 5 , b = 5 ,size =128, M=10,steps = 200, device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") ):
  device = "cpu"
  N = len(G1) + len(G2)
  N1 = len(G1)
  switching_prob = [0.0 for i in range(N)]
  switching_prob = torch.tensor(switching_prob).float().to(device) 
  P_TRANS_INTRA, A1,A2 = getIntraTransMatrix(G1,G2)
  P_TRANS_INTRA=P_TRANS_INTRA.to(device)
  A1_th_sparse = A1.to(device)
  A2_th_sparse= A2.to(device)
  SIM = compute_struct_sim(G1,G2,attr_sim,alpha,c,layers)
  P_TRANS_INTER = getInterTransMatrixFast(G1,G2,None,alpha,c,M,layers,SIM).to(device)

  U,sigma,V  = cross_graph_MF(switching_prob, P_TRANS_INTRA,P_TRANS_INTER,N,device,window_size,b,size)
  e = U @ torch.sqrt(torch.diag(sigma))
  e = e.numpy()
  e1 = e[:N1]
  e2 = e[N1:]
  return e1,e2
  

def get_ordered_graph(g):
  old_nodes = sorted(g.nodes()) # small to large
  new_nodes = [i for i in range(len(g))]
  old2new = {old_nodes[i]:i for i in new_nodes}
  new_edges = [(old2new[edge[0]], old2new[edge[1]]) for edge in g.edges()]
  new_g = nx.Graph()
  new_g.add_nodes_from(new_nodes)
  new_g.add_edges_from(new_edges)
  return new_g, {old2new[key]:key for key in old2new.keys()}
  


      
def partition_train(dataname,rate, G1,G2,attr_sim, lr = 0.5, layers = 3, alpha = 5 , c = 0.5 , window_size = 5 , b = 5 ,size =128, M=10, steps = 30, device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")):
 
  s_time = time.time()
  visited_pairs = []
  unqualified_pairs = []
  
  adj1 = nx.to_numpy_array(G1, nodelist=list(range(len(G1))))
  adj2 = nx.to_numpy_array(G2, nodelist=list(range(len(G2))))
  if dataname == "blog1-blog2":
    radius = 1
  elif dataname == "dblp1-dblp2":
    radius = 3
  else:
    radius = 2
    
  
 
  
 
  if dataname == "blog1-blog2" or dataname == "arxiv1-arxiv2":
    MAX_NUM = 3000 #arxiv 3000
    MIN_NUM = 300 # arxiv 300 #blog 500 #email 300 # twitter 100
    DIFF_BASE =50  # arxv 50 twotter:10, hops=2, dblp_hops=3
    MAX_ITERS = 100
  else:
    MAX_NUM = 3000
    MIN_NUM = 100 # arxiv 300 #blog 500 #email 300 # twitter 100
    DIFF_BASE =10 
    MAX_ITERS = 40
  iters = 0

  sim_path = "{}_{}_sim.npy".format(dataname,rate)
  if not os.path.exists(sim_path):
    SIM = compute_struct_sim(G1,G2,attr_sim,alpha,c,layers)
    print("initial time use {}".format(time.time()-s_time))
  
    np.save(sim_path,SIM)
  else:
    SIM = np.load(sim_path)
    get_hits(SIM, len(G1))
    #raise EOFError
  rdd = cal_rdd(G1,G2)
  candidate_pairs = select_candidate_pairs(None,None, np.ones_like(rdd),sim=SIM)
  #candidate_pairs = select_candidate_pairs(None,None, rdd,sim=SIM)
  COUNT_MATRIX = np.ones_like(SIM) 
  
  
  #candidate_pairs = select_candidate_pairs(e1,e2, rdd,None)
  
  
  ans_dict = {i:i for i in range(len(G1))}
  print("begin training~~~~~")
 
  while iters < MAX_ITERS:
    
    #random.shuffle(candidate_pairs)
    pair = candidate_pairs.pop()

    if pair in visited_pairs or pair in unqualified_pairs:
    
      continue
    node1, node2 = pair[0],pair[1]
    if iters ==MAX_ITERS:
      break


    sub_G1 = nx.ego_graph(G1,node1,radius=radius) 
    sub_G2 = nx.ego_graph(G2,node2,radius=radius)
 
    if len(sub_G1) > MAX_NUM or len(sub_G2) > MAX_NUM:
      continue
    elif len(sub_G1)<MIN_NUM and  len(sub_G2)<MIN_NUM:
      continue
    elif abs(len(sub_G1) -len(sub_G2))>DIFF_BASE:
      unqualified_pairs.append(pair)
      print("unqualified pair:", pair)
      continue
    else:
      visited_pairs.append(pair)
      print((len(sub_G1),len(sub_G2)))
      print(visited_pairs)
      
      iters += 1
      ordered_sub_G1, new2old_G1 = get_ordered_graph(sub_G1)
      ordered_sub_G2, new2old_G2 = get_ordered_graph(sub_G2)
      extracted_sim = SIM[sorted(sub_G1),:][:,sorted(sub_G2)]
      train_start_time = time.time()
      
      try:

        sim = batch_train(ordered_sub_G1, ordered_sub_G2, extracted_sim)
        print("finish the {}-th batch training, use_time :{}".format(iters, time.time()-train_start_time))
       
        tmp_start_time = time.time()
        SIM[np.ix_(sorted(sub_G1.nodes()),sorted(sub_G2.nodes()))] = sim + SIM[np.ix_(sorted(sub_G1.nodes()),sorted(sub_G2.nodes()))]
        COUNT_MATRIX[np.ix_(sorted(sub_G1.nodes()),sorted(sub_G2.nodes()))] =COUNT_MATRIX[np.ix_(sorted(sub_G1.nodes()),sorted(sub_G2.nodes()))] + np.ones_like(sim)
        print("assign value use time {}".format(time.time()-tmp_start_time))
        ss_time = time.time()
        
      
        if iters % 5 == 0:
          RESULT_SIM = SIM/COUNT_MATRIX
          #jaccad = adj1 @ RESULT_SIM @ adj2
          get_hits(RESULT_SIM, len(G1))
          print("evaluation using {}".format(time.time()-ss_time))
       
          candidate_pairs = select_candidate_pairs(None,None, np.ones_like(rdd),sim=RESULT_SIM)
      
          print("looking for candidates using {}".format(time.time()-ss_time))
          
          print("totao used time:{}".format(time.time()-s_time))
      except:
        print("unstable SVD factorization for this pair")
        pass 

        
   
     
 

        
        
        
        

    

    

  
  
          









