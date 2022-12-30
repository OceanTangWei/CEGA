

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import faiss
import pickle
import json
from tqdm import tqdm
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")


def sparse_sinkhorn_sims(left,right,features,top_k=500,iteration=15,mode = "test"):
    features_l = features[left]
    features_r = features[right]

    faiss.normalize_L2(features_l); faiss.normalize_L2(features_r)

    res = faiss.StandardGpuResources()
    dim, measure = features_l.shape[1], faiss.METRIC_INNER_PRODUCT
    if mode == "test":
        param = 'Flat'
        index = faiss.index_factory(dim, param, measure)
    else:
        param = 'IVF256(RCQ2x5),PQ32'
        index = faiss.index_factory(dim, param, measure)
        index.nprobe = 16
    if len(gpus):
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(features_r)
    index.add(features_r)
    sims, index = index.search(features_l, top_k)
    
    row_sims = K.exp(sims.flatten()/0.02)
    index = K.flatten(index.astype("int32"))

    size = len(left)
    row_index = K.transpose(([K.arange(size*top_k)//top_k,index,K.arange(size*top_k)]))
    col_index = tf.gather(row_index,tf.argsort(row_index[:,1]))
    covert_idx = tf.argsort(col_index[:,2])

    for _ in range(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:,0],params = tf.math.segment_sum(row_sims,row_index[:,0]))
        col_sims = tf.gather(row_sims,col_index[:,2])
        col_sims = col_sims / tf.gather(indices=col_index[:,1],params = tf.math.segment_sum(col_sims,col_index[:,1]))
        row_sims = tf.gather(col_sims,covert_idx)
        
    return K.reshape(row_index[:,1],(-1,top_k)), K.reshape(row_sims,(-1,top_k))

def test(test_pair,features,top_k=500,iteration=15):
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    index,sims = sparse_sinkhorn_sims(left, right,features,top_k,iteration,"test")
    ranks = tf.argsort(-sims,-1).numpy()
    index = index.numpy()
    
    wrong_list,right_list = [],[]
    h1,h10,mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1)
    pos[right] = np.arange(len(right))
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == index[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i],right[index[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n"%(h1/len(test_pair),h10/len(test_pair),mrr/len(test_pair)))
    
    return right_list, wrong_list
