import os
import pickle as pkl
import random
import torch
from torch.utils.data import Dataset

from .sample_patients import build_stack_ordered_nonoverlapping_indices, build_stack_ordered_overlapping_indices, create_oversampled_index_dataset

__all__ = ['msdDatasetTrain', 'msdDatasetEvaluation', 'msdDatasetTest']

#################### OVERLAPPING DATASET FOR TRAINING ####################

class msdDatasetTrain(Dataset):
    def __init__(self, dataset_folder, transform = None, stack_size = 6, batch_size = 16, tumour_percent_threshold = 0.5, samples_proportion = 0.7, undersample_flag = False, undersample_size = 300):
        '''Am stabilit stack_size la 6 pe baza discutiei cu Doamna Udrea care sugera intre 3 si 6 imagini in stack + EDA2
        
        
        self.patients - contains (image, label) pairs for each patient in the dataset
        self.stacks_in_order_indices - contains the tuples (patient_id, stack_indices, other relevant attributes based on split_type) for each stack in the dataset
        
        '''
        self.img_folder = dataset_folder + "images/"
        self.label_folder = dataset_folder + "labels/"
        self.no_patients = len(os.listdir(self.img_folder))

        self.stack_size = stack_size
        self.batch_size = batch_size
        self.transform = transform

        split_type = 'training' if 'training' in dataset_folder else 'validation'

        # Flag that indicates we are working with the training dataset, we want to apply random rotations only for this split
        self.train_flag = split_type == 'training'

        print(split_type)
        self.patients = [self.get_img_and_label(i) for i in range(self.no_patients)]

        # for i, pat in enumerate(self.patients):
        #     print(f'Patient {i} - Image shape: {pat[0].shape}')

        # If the indices for the dataset have been already built, load them, otherwise build them
        if os.path.exists(f'./ordered_overlapping_{split_type}_indices_stack={stack_size}.pkl'):
            print(f'./ordered_overlapping_{split_type}_indices_stack={stack_size}.pkl exists. Loading the ordered indices.')
            with open(f'./ordered_overlapping_{split_type}_indices_stack={stack_size}.pkl', 'rb') as f:
                self.stacks_in_order_indices = pkl.load(f)
        else:
            print(f'./ordered_overlapping_{split_type}_indices_stack={stack_size}.pkl does not exist. Building the ordered indices.')
            slices_per_patient = [self.patients[i][0].shape[-1] for i in range(self.no_patients)]
            self.stacks_in_order_indices = build_stack_ordered_overlapping_indices(split_type, slices_per_patient, stack_size, self.patients, upload_flag = True)
            
        if split_type == 'training':
            batches_proportion = 0.5

            if undersample_flag:
                working_indiced_path = f'./{split_type}_indices_stack={stack_size}_samples{samples_proportion}_batches{batches_proportion}_undersample={undersample_flag}_size={undersample_size}.pkl'
            else:
                working_indiced_path = f'./{split_type}_indices_stack={stack_size}_samples{samples_proportion}_batches{batches_proportion}.pkl'

            if os.path.exists(working_indiced_path):
                print(f'{working_indiced_path} exists. Loading the overall indices.')
                with open(working_indiced_path, 'rb') as f:
                    self.stacks_in_order_indices = pkl.load(f)
            else:
                print(f'{working_indiced_path} does not exist. Building the overall indices.')
                self.stacks_in_order_indices = create_oversampled_index_dataset(self.stacks_in_order_indices, batch_size, save_path = working_indiced_path, 
                                                                                tumour_percent_threshold = tumour_percent_threshold, samples_proportion = samples_proportion,
                                                                                undersample_flag=undersample_flag, undersample_size = undersample_size)
        

        # print(len(self.stacks_in_order_indices))
        # self.device = device
        self.length = len(self.stacks_in_order_indices)


    def shuffle(self):
        ''' Shuffles the indices of the dataset '''
        
        batches = list(range(self.length // self.batch_size))

        random.shuffle(batches)

        new_indices = []

        for i in batches:
            batch = self.stacks_in_order_indices[i * self.batch_size : (i + 1) * self.batch_size]
            random.shuffle(batch)
            new_indices += batch

        self.stacks_in_order_indices = new_indices

    def __len__(self):
        return self.length

    def get_img_and_label(self, patient_id):
        ''' Helper function: For a specified patient returns its image and label stacks from the dataset '''

        img = torch.load(self.img_folder + f'patient_{patient_id}.pt')
        label = torch.load(self.label_folder + f'patient_{patient_id}.pt')
        return img, label

    def __getitem__(self, idx):
        # print(f'Getting item {idx}/{len(self)}')
        # print(self.img_folder + f'patient_{patient_id}.pt')

        # Based on current index, get the patient_id and the slices that form the current stack

        if idx >= 0 and idx < self.length:
            # print('Index: ', idx)
            # print(len(self.stacks_in_order_indices))

            stacks_tuple = self.stacks_in_order_indices[idx]
            patient_id, chosen_stacks = stacks_tuple[0], stacks_tuple[1]

            # print('Patient id: ', patient_id, 'Chosen stacks: ', chosen_stacks)

            img, label = self.patients[patient_id]

            # print('Image shape: ', img.shape, 'Label shape: ', label.shape)
            # Filters the current stack of images and labels for the current batch
            img, label = img[..., chosen_stacks], label[..., chosen_stacks]

            if self.train_flag and self.transform and stacks_tuple[3]:
                output = self.transform({'image': img, 'label': label})
                img, label = output['image'], output['label']

            # label = convert_labels_to_one_hot(label, 2) # Not needed for now, as MONAI handles the conversion internally
            return img, label
        else:
            raise IndexError
        
    

#################### NON-OVERLAPPING DATASET FOR EVALUATION ####################
class msdDatasetEvaluation(Dataset):
    def __init__(self, dataset_folder, transform = None, stack_size = 6, batch_size = 16):
        '''
        
        self.patients - contains (image, label) pairs for each patient in the dataset
        self.stacks_in_order_indices - contains the tuples (patient_id, stack_indices, other relevant attributes based on split_type) for each stack in the dataset
        
        '''
        self.img_folder = dataset_folder + "images/"
        self.label_folder = dataset_folder + "labels/"
        self.no_patients = len(os.listdir(self.img_folder))
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.transform = transform

        split_type = dataset_folder.split('/')[-2]

        # Flag that indicates we are working with the training dataset, we want to apply random rotations only for this split
        self.train_flag = split_type == 'training'

        print(split_type)
        self.patients = [self.get_img_and_label(i) for i in range(self.no_patients)]

        # If the indices for the dataset have been already built, load them, otherwise build them
        if os.path.exists(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl'):
            print(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl exists. Loading the ordered indices.')
            with open(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl', 'rb') as f:
                self.stacks_in_order_indices = pkl.load(f)
        else:
            print(f'./ordered_nonoverlapping_{split_type}_indices_stack={stack_size}.pkl does not exist. Building the ordered indices.')
            slices_per_patient = [self.patients[i][0].shape[-1] for i in range(self.no_patients)]
            self.stacks_in_order_indices = build_stack_ordered_nonoverlapping_indices(split_type, slices_per_patient, stack_size, self.patients, upload_flag = True)

        # self.device = device
        self.length = len(self.stacks_in_order_indices)


    def __len__(self):
        return self.length

    def get_img_and_label(self, patient_id):
        ''' Helper function: For a specified patient returns its image and label stacks from the dataset '''

        img = torch.load(self.img_folder + f'patient_{patient_id}.pt')
        label = torch.load(self.label_folder + f'patient_{patient_id}.pt')
        return img, label

    def __getitem__(self, idx):
        # print(f'Getting item {idx}/{len(self)}')
        # print(self.img_folder + f'patient_{patient_id}.pt')

        # Based on current index, get the patient_id and the slices that form the current stack

        if idx >= 0 and idx < self.length:
            stacks_tuple = self.stacks_in_order_indices[idx]

            patient_id, chosen_stacks = stacks_tuple[0], stacks_tuple[1]

            img, label = self.patients[patient_id]
            # Filters the current stack of images and labels for the current batch
            img, label = img[..., chosen_stacks], label[..., chosen_stacks]

            if self.train_flag and self.transform and stacks_tuple[3]:
                output = self.transform({'image': img, 'label': label})
                img, label = output['image'], output['label']

            # label = convert_labels_to_one_hot(label, 2) # Not needed for now, as MONAI handles the conversion internally
            return img, label
        else:
            raise IndexError
        

#################### NON-OVERLAPPING DATASET FOR TEST ####################
class msdDatasetTest(Dataset):
    def __init__(self, dataset_folder, transform = None, stack_size = 6, batch_size = 16):

        self.img_folder = dataset_folder + "images/"
        self.no_patients = len(os.listdir(self.img_folder))
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.transform = transform


        # Flag that indicates we are working with the training dataset, we want to apply random rotations only for this split
        self.patients = [self.get_img(i) for i in range(self.no_patients)]

        # If the indices for the dataset have been already built, load them, otherwise build them
        if os.path.exists(f'./ordered_nonoverlapping_test_indices_stack={stack_size}.pkl'):
            print(f'./ordered_nonoverlapping_test_indices_stack={stack_size}.pkl exists. Loading the ordered indices.')
            with open(f'./ordered_nonoverlapping_test_indices_stack={stack_size}.pkl', 'rb') as f:
                self.stacks_in_order_indices = pkl.load(f)
        else:
            print(f'./ordered_nonoverlapping_test_indices_stack={stack_size}.pkl does not exist. Building the ordered indices.')
            slices_per_patient = [self.patients[i][0].shape[-1] for i in range(self.no_patients)]
            self.stacks_in_order_indices = build_stack_ordered_nonoverlapping_indices('test', slices_per_patient, stack_size, self.patients, upload_flag = True)

        # self.device = device
        self.length = len(self.stacks_in_order_indices)

    def __len__(self):
        return self.length

    def get_img(self, patient_id):
        ''' Helper function: For a specified patient returns its image stacks from the dataset '''

        img = torch.load(self.img_folder + f'patient_{patient_id}.pt')
        return img

    def __getitem__(self, idx):
        # print(f'Getting item {idx}/{len(self)}')
        # print(self.img_folder + f'patient_{patient_id}.pt')

        # Based on current index, get the patient_id and the slices that form the current stack

        if idx >= 0 and idx < self.length:
            stacks_tuple = self.stacks_in_order_indices[idx]

            patient_id, chosen_stacks = stacks_tuple[0], stacks_tuple[1]

            img = self.patients[patient_id]
            # Filters the current stack of images for the current batch
            img = img[..., chosen_stacks]

            if self.transform and stacks_tuple[3]:
                output = self.transform({'image': img})
                img = output['image']

            return img
        else:
            raise IndexError