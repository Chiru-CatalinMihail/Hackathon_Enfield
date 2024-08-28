import torch
from monai.transforms import Activations, AsDiscrete

__all__ = ['convert_labels_to_one_hot']

def convert_labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    ''' Converts a tensor of labels to a one-hot tensor in which each channel corresponds to a binary decision for each class from the original tensor.'''
    
    one_hot = torch.zeros((2*labels.shape[0], labels.shape[1], labels.shape[2], labels.shape[3])).to(labels.device)
    
    one_hot[0, :, :, :] = (labels == 0).squeeze(1).float()
    one_hot[1, :, :, :] = (labels != 0).squeeze(1).float()

    return one_hot
    
    

def hard_threshold_labels(labels, threshold = 0.5, cutoff_flag = False):
    '''Thresholds the labels to 0 or 1 based on a specified threshold.'''

    sigmoid_activation = Activations(sigmoid=True)

    labels = sigmoid_activation(labels)


    if cutoff_flag:
        hard_thresholding = AsDiscrete(threshold=threshold)
        mask_from_background = (1 - labels[1])
        mask = labels[0]

        # Mean based on the two masks
        labels = (mask_from_background + mask) / 2
        max_value = torch.max(labels)
        min_value = torch.min(labels)

        if max_value > 0:
            labels = (labels - min_value) / (max_value - min_value)

        labels = hard_thresholding(labels)
        labels = np.expand_dims(labels, 0)
        # print(labels.shape)

    return labels
