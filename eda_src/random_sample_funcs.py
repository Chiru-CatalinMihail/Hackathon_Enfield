import torch
import numpy as np

__all__ = ['get_first_tumorous_slice', 'get_last_tumorous_slice', 'aggregate_heatmap', 'aggregate_heatmap_average', 'create_boundingbox', 'rmse_dev_vs_val']



# --------------------------------------------------------------------

def get_first_tumorous_slice(patient):
    last_dim = patient.shape[-1]
        
    for i in range(last_dim):
        mask = patient[0,..., i].sum()
        if mask > 0:
            return i
    
    raise Exception("Patient does not have tumorous slices") 
    
    
def get_last_tumorous_slice(patient):
    last_dim = patient.shape[-1]
        
    for i in range(last_dim):
        mask = patient[0,..., last_dim - i - 1].sum()
        if mask > 0:
            return last_dim - i - 1
    
    raise Exception("Patient does not have tumorous slices") 

# --------------------------------------------------------------------

def aggregate_heatmap(dataset):
    sample_stack = torch.zeros_like(dataset[0]['label'][0, ..., 1])     
#     print(sample_stack.shape)
    sample_stack = sample_stack.unsqueeze(0)
    
    for i in range(len(dataset)):
        sample_stack += dataset[i]['label'].sum(dim=3)
        
    sample_stack = sample_stack.squeeze(0)
    
    return sample_stack


def aggregate_heatmap_average(dataset):
    avg_slice_idx = []
    no_tumor_slices = 0
    
    for sample in range(len(dataset)):
        tumor_start = get_first_tumorous_slice(dataset[sample]['label'])
        tumor_end = get_last_tumorous_slice(dataset[sample]['label'])

        no_tumor_slices += int(tumor_end - tumor_start + 1)
    
    sample_stack = aggregate_heatmap(dataset)
    sample_stack /= no_tumor_slices
    
    return sample_stack

def create_boundingbox(dataset):
    boxes = []
    
    for idx, patient in enumerate(dataset):
        temp_list = [patient]
        heatmap = aggregate_heatmap(temp_list)
        
        height = heatmap.sum(dim= 1)
        
        h_min, h_max = 0, 0

        for i in range(height.shape[0]):
            if height[i] > 0:
                h_min = i
                break

        for i in range(height.shape[0] - 1, -1, -1):
            if height[i] > 0:
                h_max = i
                break

        width = heatmap.sum(dim= 0)

        w_min, w_max = 0, 0

        for i in range(width.shape[0]):
            if width[i] > 0:
                w_min = i
                break

        for i in range(width.shape[0] - 1, -1, -1):
            if width[i] > 0:
                w_max = i
                break
                
#         print(h_min, h_max, w_min, w_max)
        boxes.append([h_min, h_max, w_min, w_max])

    return boxes

# --------------------------------------------------------------------

def rmse_dev_vs_val(dev_heatmap, val_heatmap):
    rmse = np.sqrt((np.square(dev_heatmap - val_heatmap))).mean()
    
    return rmse

