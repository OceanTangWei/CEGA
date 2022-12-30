# coding=gbk
import numpy as np
import community

from tools import get_adj_matrix
import copy
import networkx as nx
from tools import *
from CEGA import train,partition_train
import random
from sklearn.metrics.pairwise import cosine_similarity
from tools import compute_structural_similarity
import time




def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  



def read_data(data_name="facebook-twitter", use_attr=False,remove_rate=0.02):
    name1,name2 = data_name.split("-")
    H = None
    if name1 == "facebook":
        g1 = read_tex_graph(name1)
        g2 = read_tex_graph(name2)

        g2, r_edges = create_align_graph(g2, remove_rate=remove_rate)
       
    elif name1 == "email1":
        g1 = read_tex_graph("email")
        g2, _ = create_align_graph(g1, remove_rate=0.1)
        g1, _ = create_align_graph(g1, remove_rate=0.1)
        g2, r_edges = create_align_graph(g2, remove_rate=remove_rate)

    elif name1 == "dblp1":

        g1,attr1 = read_tex_graph("dblp")
        g2, _ = create_align_graph(g1, remove_rate=0.1)
        g1, _ = create_align_graph(g1, remove_rate=0.1)
        g2, r_edges = create_align_graph(g2, remove_rate=remove_rate)

        H = cosine_similarity(attr1,attr1)
   
    elif name1 == "blog1":
        g1 = read_tex_graph("blog")
        
        g2, _ = create_align_graph(g1, remove_rate=0.1)
        g1, _ = create_align_graph(g1, remove_rate=0.1)
        
       
        g2, r_edges = create_align_graph(g2, remove_rate=remove_rate)
 

    elif name1 == "arxiv1":
        g1 = read_tex_graph("arxiv")
        g2, _ = create_align_graph(g1, remove_rate=0.1)
        g1, _ = create_align_graph(g1, remove_rate=0.1)

        g2, r_edges = create_align_graph(g2, remove_rate=remove_rate)
     
    else:
        raise EOFError
    return g1,g2,H


  
def jaccad(x,y):
  return len(x & y)/len(x | y)

if __name__ == '__main__':
    config = {
    'lr' : 0.5,
    'layers': 3,
    'alpha': 5, 
    'c' :0.5 ,
    'window_size': 5 , 
    'b': 5 ,
    'size':128,
    'M':10,
    'steps':30, 
    'device': torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    }
    rate = 0.02

    use_attr = False
    
    setup_seed(0) 
    # suggest setting speed_flag as True when dataname = blog or arxiv
    speed_flag = False
 

    #dataname = "blog1-blog2"
    dataname = "dblp1-dblp2"
    #dataname = "facebook-twitter"
    #dataname = "email1-email2"
    #dataname = "arxiv1-arxiv2"
    
    
    
    print("~~~~~~~{}~~~~~~~~~~{}~~~~~~~~~~~~~~~~~".format(dataname,rate))
   
    g1, g2, H = read_data(dataname,remove_rate=rate)
    if use_attr is False:
      H = None
    
    if speed_flag is False:
      train(g1, g2,H,**config)
    else:
      partition_train(dataname,rate, g1,g2,H,**config)
  
 


      
