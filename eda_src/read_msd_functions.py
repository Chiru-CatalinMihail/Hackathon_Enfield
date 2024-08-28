#### Functions to read the Medical Decathlon dataset ####

# Based either on Monai or on memoized data

import os
import pickle as pkl

import torch
import monai
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    OrientationD,
)


# Types
from .types import *

__all__ = ['read_msd', 'read_memoized_data']

def read_memoized_data(path : str, split_name : str) -> [{str: torch.Tensor, str: torch.Tensor}]:
    '''
    Helper function for read_msd, assumes the memoized path (path) exists and is built properly (See convert_dataset.ipynb)
    Reads the memoized data from the path and returns a list of dictionaries with the keys 'image' and 'label'
    '''

    crt_path = f'{path}/{split_name}'
    
    no_patients = len(os.listdir(f'{crt_path}/images'))
    
    patients = []

    for patient_id in range(no_patients):
        img = torch.load(f'{crt_path}/images/patient_{patient_id}.pt')
        label = torch.load(f'{crt_path}/labels/patient_{patient_id}.pt')
        patients.append({"image": img, "label": label})
    
    return patients

def read_msd(memoized_path : str, split_name : str, base_transform : monai.transforms.compose.Compose = None, load_original_flag : bool = False, tumorous_slices_flag : bool = False) -> (name, Dataset, DataLoader):
    '''
    If it is the first time, downloads the Medical Decathlon dataset from Monai
    
    If we want to load the original data, it reads the data from the path it has been downloaded to
        Using



    If we want the data memoized using convert_dataset.ipynb, it reads the data from the path it has been memoized to
    
    Returns (split_name, dataset_type, dataset, loader)
    '''

    memoized_path_exists = os.path.exists(memoized_path)
    dataset_type = None

    if not memoized_path_exists or load_original_flag:

        if not load_original_flag:
            print(f'Memoized path {memoized_path} does not exist!\nLoading original data instead...')
        ### The dataset where the original MSD data is stored, as nii.gz archives
        crt_dir = os.getcwd()
        datasets_path = f'{crt_dir}/datasets/MedicalDecathlon/'

        download_flag = not os.path.exists(f'{datasets_path}{split_name}/')

        crt_dataset = DecathlonDataset(root_dir = f'{datasets_path}{split_name}/',
                                task = "Task06_Lung", section = split_name,
                                transform = base_transform, download = download_flag)


        dataset_type = 'original'

        if tumorous_slices_flag:
            pkl_path_tumorous_stacks = f'{crt_dir}/tumorous_slices_{split_name}.pkl'
            knows_tumorous_stacks_for_training = os.path.exists(pkl_path_tumorous_stacks)
            if knows_tumorous_stacks_for_training:
                with open(pkl_path_tumorous_stacks, 'rb') as f:
                    patient_tumorous_slices = pkl.load(f)
            else:
                # TODO: Verificat daca acolo se memoreaza tumorous 
                raise AssertionError(f'\nNo tumorous stacks known for {split_name}!\n\nPlease go through MSD_EDA.ipynb to generate the pickle file needed: {pkl_path_tumorous_stacks}')
    
            for i in range(len(crt_dataset)):
                crt_dataset[i]['image'] = crt_dataset[i]['image'][:, :, patient_tumorous_slices[i]]
                crt_dataset[i]['label'] = crt_dataset[i]['label'][:, :, patient_tumorous_slices[i]]

            dataset_type += 'TumorousSlices'
    else:
        print('Loading memoized data')
        crt_dataset = read_memoized_data(memoized_path, split_name)
        dataset_type = 'memoized' + memoized_path.split('Decathlon')[1]

    crt_loader = DataLoader(crt_dataset, batch_size = 1, shuffle = False) #, num_workers = 1)

    if dataset_type[-1] == '/':
        dataset_type = dataset_type[:-1]

    return (split_name, dataset_type, crt_dataset, crt_loader)


def print_unit_test_results(dataset : Dataset, dataset_type: name):
    print(dataset[0]['image'].shape)
    print("Dataset type:", dataset_type)


if __name__ == "__main__":
    KEYS = ["image", "label"]

    base_transform = Compose([
        LoadImageD(keys=KEYS),
        EnsureChannelFirstD(keys=KEYS),
        OrientationD(keys=KEYS, axcodes='RAS'),
    ])


    split_name = 'training'

    # Load dataset
    # memoized_path = f'/raid/CataChiru/MedicalDecathlonTensors/'
    # memoized_path = f'./datasets/MedicalDecathlonJustLungs'
    # memoized_path = f'./datasets/MedicalDecathlonJustTumors'
    memoized_path = f'./datasets/MedicalDecathlonAugmentedTumors'

    # print("Unit Test 1: Load original data for train")
    # split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name, base_transform, load_original_flag = True)
    # print_unit_test_results(dataset, dataset_type)

    print("Unit Test 2: Load memoized data for train")
    split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name)
    print_unit_test_results(dataset, dataset_type)


    # print("Unit Test 3: Load original data with tumorous slices for train")
    # split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name, base_transform, load_original_flag = True, tumorous_slices_flag = True)
    # print_unit_test_results(dataset, dataset_type)


    split_name = 'validation'

    print("Unit Test 4: Load original data for validation")
    split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name, base_transform, load_original_flag = True)
    print_unit_test_results(dataset, dataset_type)

    print("Unit Test 5: Load memoized data for validation")
    split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name)
    print_unit_test_results(dataset, dataset_type)

    print("Unit Test 6: Load original data with tumorous slices for validation")
    split_name, dataset_type, dataset, loader = read_msd(memoized_path, split_name, base_transform, load_original_flag = True, tumorous_slices_flag = True)
    print_unit_test_results(dataset, dataset_type)
