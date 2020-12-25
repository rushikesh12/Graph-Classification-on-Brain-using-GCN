import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix

def get_data(dataset = "Training", feature = False):
    healthy = []
    patient = []
    if dataset == "Training" and feature == False:
        for i in range(1,11):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/common_fiber_matrix.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/common_fiber_matrix.txt'), 0])
    
    elif dataset == "Testing" and feature == False:
        for i in range(1,6):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/common_fiber_matrix.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/common_fiber_matrix.txt'), 0])
    
    elif dataset == "Training" and feature == True:
        for i in range(1,11):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 0])
    
    elif dataset == "Testing" and feature == True:
        for i in range(1,6):
            healthy.append([np.genfromtxt('../'+dataset+'/Health/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 1])
            patient.append([np.genfromtxt('../'+dataset+'/Patient/sub'+str(i)+'/pcc_fmri_feature_matrix_0.txt'), 0])
            
    data = []
    for i in range(len(healthy)):
        data.append(healthy[i])
        data.append(patient[i])
    
    del healthy, patient
    return data

# Using the PyTorch Geometric's Data class to load the data into the Data class needed to create the dataset
def create_dataset(data, features = None):
    dataset_list = []
    for i in range(len(data)):
        degree_matrix = np.count_nonzero(data[i][0], axis=1).reshape(150,1)
        weight_matrix = np.diag(np.sum(data[i][0], axis=1)).diagonal().reshape(150,1)
        feature_matrix = np.hstack((degree_matrix, weight_matrix))
        edge_index_coo = coo_matrix(data[i][0])
        edge_index_coo = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype = torch.long)
        if features != None:
            feature_matrix = features[i][0]
        graph_data = Data(x = torch.tensor(feature_matrix, dtype = torch.float32), edge_index=edge_index_coo, y = torch.tensor(data[i][1]))
        dataset_list.append(graph_data)
    return dataset_list