import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
import random

__all__ = ['build_stack_ordered_overlapping_indices', 'build_stack_ordered_nonoverlapping_indices', 'create_oversampled_index_dataset']

def compute_tumour_percentage_per_patient(tumour):
    '''
    Computes the percentage of tumour in each patient
    '''

    return 100*tumour.sum() / np.prod(tumour.shape)


########################### BUILDING SAMPLES FOR TRAINING AND EVALUATION ###########################
    
def build_stack_ordered_overlapping_indices(split_type : str, slices_per_patient: [int], stack_size : int, patients : list = None, upload_flag : bool = True ) -> list[tuple]:
    ''' 
    The function builds samples used in training
     
    Applies an overlapping sliding window of stack_size images from the first to last CTs of each patient.

    Returns a list of tuples (patient_id, stack_indices, tumour_percentage, has_tumour, original_idx)    
    '''
    stacks_in_order_indices =[]


    # Real index of the stack in the dataset, memorized for reordering the shuffled dataset 
    real_idx = 0
    
    # Saves the indices of the sliding window for each patient
    for patient_id, slices in tqdm(enumerate(slices_per_patient)):
        crt_slices = []

        for i in range(0, slices - stack_size + 1):
            stacks_range = np.arange(i, i+stack_size)
            
            crt_minivolume_mask = patients[patient_id][1][..., i:i+stack_size]
            tumour_percentage = compute_tumour_percentage_per_patient(crt_minivolume_mask)

            # Based on MSD_EDA_hyperparameter, there are 1000 samples with tumour percentage in [0, 1e-5], we will not consider them as having tumour
            crt_slices.append((patient_id, stacks_range, tumour_percentage, tumour_percentage > 1e-5, real_idx))

            real_idx += 1
        
        
        stacks_in_order_indices += crt_slices

    if upload_flag and not os.path.exists(f'./ordered_{split_type}_indices_stack={stack_size}.pkl'):
        with open(f'./ordered_overlapping_{split_type}_indices_stack={stack_size}.pkl', 'wb') as f:
            pkl.dump(stacks_in_order_indices, f)

    return stacks_in_order_indices

def build_stack_ordered_nonoverlapping_indices(split_type : str, slices_per_patient: [int], stack_size : int, tumours : list = None, upload_flag : bool = True ) -> list[tuple]:
    ''' 
    
    Iterates a non-overlapping sliding window of stack_size images for each patient per batch. 

    The stacks are built with a stride of "stack_size" over the entire volume of each patient, and the last stack is padded with the last slice of the volume up to "stack_size" slices.

    Used for evaluation, to keep the slices in order and memorise the results for fine-coarse segmentation.

    Returns a list of tuples (patient_id, stack_indices) '''

    stacks_in_order_indices =[]
    
    for patient_id, slices in enumerate(slices_per_patient):
        # Non-overlapping sliding window of stack_size images, with stride = stack_size

        padding = stack_size - slices % stack_size
        remaining_difference = -1

        for i in range(0, slices, stack_size):
            remaining_difference = i + stack_size - slices
            if remaining_difference > 0:
                break

            stacks_in_order_indices.append((patient_id, np.arange(i, i+stack_size)))

        # If the last stack is smaller than stack_size, we pad it with the last slice of the volume
        if padding  % stack_size != 0 and remaining_difference > 0:
            remaining_slices_indices = np.arange(i, slices)
            repeated_slices = np.repeat(slices - 1, padding)
            batch_indices = np.hstack((remaining_slices_indices, repeated_slices))
            stacks_in_order_indices.append((patient_id, batch_indices))


    if upload_flag and not os.path.exists(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl'):
        with open(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl', 'wb') as f:
            pkl.dump(stacks_in_order_indices, f)

    return stacks_in_order_indices

########################### BALANCING SAMPLES IN A BATCH ###########################

def oversample_tumours_undersample_healthy(ordered_stacks, tumour_percent_threshold : float = 0.125, samples_proportion : float = 0.7, undersample_flag : bool = False, undersample_size : int = 300):
    '''
    Splits the dataset into two portions: tumorous and healthy / with small tumours

    If undersample_flag is True, shuffles the small tumour portion, and keeps only undersample_size samples from it afterwards

    Oversamples the tumorous portion based on the desired final proportion of the dataset - samples_proportion

    Returns big_tumour_stacks, small_tumour_stacks - the two portions of the dataset
    '''

    ordered_stacks.sort(key = lambda x: x[2], reverse = True)

    small_tumour_stacks = list(filter(lambda x: x[2] < tumour_percent_threshold, ordered_stacks))

    print("Small tumour stacks", len(small_tumour_stacks))

    # TODO shuffle tensor and keep only undersample_size
    if undersample_flag:
        np.random.shuffle(small_tumour_stacks)
        if undersample_size < len(small_tumour_stacks):
            small_tumour_stacks = small_tumour_stacks[:undersample_size]

    length_small_tumour_stacks = len(small_tumour_stacks)

    print("Small tumour stacks2", length_small_tumour_stacks)


    big_tumour_stacks = list(filter(lambda x: x[2] > tumour_percent_threshold, ordered_stacks))

    print("Big tumour stacks", len(big_tumour_stacks))


    length_big_tumour_stacks = len(big_tumour_stacks)
    oversampling_factor = int(length_small_tumour_stacks / ((1-samples_proportion) * length_big_tumour_stacks))

    print("Oversampling factor", oversampling_factor)

    if oversampling_factor > 1:
        big_tumour_stacks = big_tumour_stacks * oversampling_factor
    else:
        keep_undersample_indices = int(length_small_tumour_stacks * samples_proportion / (1 - samples_proportion))

        print("Keep undersample indices", keep_undersample_indices)
        np.random.shuffle(big_tumour_stacks)
        big_tumour_stacks = big_tumour_stacks[:keep_undersample_indices]


    print("Big tumour stacks2", len(big_tumour_stacks))

    return small_tumour_stacks, big_tumour_stacks

def balance_batches(small_tumour_stacks : list, big_tumour_stacks: list, cancerous_samples_in_batch : float, batch_size : int) -> list:
    '''
    Creates the dataset ordered in tumorous and healthy samples such that when splitting in batches by batch_size, the desired percentage of tumourous samples is achieved - cancerous_samples_in_batch
    '''

    length = len(small_tumour_stacks) + len(big_tumour_stacks)


    # dataset_percentages = len(big_tumour_stacks) / length
    # if dataset_percentages / length > cancerous_samples_in_batch + 0.1 or dataset_percentages / length < cancerous_samples_in_batch - 0.1:
    #     assert False, f"Dataset percentage is {dataset_percentages} and the desired percentage {cancerous_samples_in_batch} is too different from the given distribution"        

    # TODO: Aici daca cancerous_samples_in_batch e diferit de 0.5, trebuie gandita o alta procedura de alegerea sample-urilor
    half_batch = batch_size // 2

    # Keeps only a multiple of half_batch healthy samples
    if len(small_tumour_stacks) % half_batch != 0:
        remove_indices_small = len(small_tumour_stacks) % half_batch
        small_tumour_stacks = small_tumour_stacks[:-remove_indices_small]

    # Keeps only a multiple of half_batch tumourous samples
    if len(big_tumour_stacks) % half_batch != 0:
        remove_indices_big = len(big_tumour_stacks) % half_batch
        big_tumour_stacks = big_tumour_stacks[:-remove_indices_big]


    print("Length of small tumour stacks", len(small_tumour_stacks))
    print("Length of big tumour stacks", len(big_tumour_stacks))

    # TODO: Ar trebui vazut aici daca nu cumva ar trebui sa fie egale la numar sau proportionale pe baza cancerous_samples_in_batch
    length = len(small_tumour_stacks) + len(big_tumour_stacks)
    
    all_stacks = []

    cancerous_samples_in_batch = int(cancerous_samples_in_batch * batch_size)
    healthy_samples_in_batch = batch_size - cancerous_samples_in_batch

    healthy_idx = 0
    tumour_idx = 0

    for crt_batch in range(0, int(length / batch_size)):

        if healthy_idx + healthy_samples_in_batch > len(small_tumour_stacks) or tumour_idx + cancerous_samples_in_batch > len(big_tumour_stacks):
            break

        all_stacks = all_stacks + small_tumour_stacks[healthy_idx : healthy_idx + healthy_samples_in_batch] + big_tumour_stacks[tumour_idx : tumour_idx + cancerous_samples_in_batch]

        healthy_idx += healthy_samples_in_batch
        tumour_idx += cancerous_samples_in_batch
        
        # Shuffle current batch
        aux = all_stacks[crt_batch * batch_size : (crt_batch + 1) * batch_size]
        random.shuffle(aux)
        all_stacks[crt_batch * batch_size : (crt_batch + 1) * batch_size] = aux

    return all_stacks


def create_oversampled_index_dataset(ordered_stacks, batch_size : int, save_path : str = None,
    tumour_percent_threshold : float = 0.125, samples_proportion : float = 0.7, batches_proportion : float = 0.5, 
    undersample_flag : bool = False, undersample_size : int = 300):
    
    '''
    Based on the threshold set for the tumour percentage, splits the dataset and oversamples the desired portion of the dataset
    Returns the complete list of indices by intercalating the two portions
    '''

    small_tumour_stacks, big_tumour_stacks = oversample_tumours_undersample_healthy(ordered_stacks, tumour_percent_threshold, samples_proportion, undersample_flag, undersample_size)
    
    # Orders the stacks in the dataset such that when spliting in batches, the percentages of tumourous and healthy samples reflect the desired distribution
    all_stacks = balance_batches(small_tumour_stacks, big_tumour_stacks, batches_proportion, batch_size)

    if save_path:
        with open(save_path, 'wb') as f:
            pkl.dump(all_stacks, f)

    print("Length of the dataset", len(all_stacks))
    return all_stacks



