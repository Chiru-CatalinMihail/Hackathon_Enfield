import os
import warnings
warnings.filterwarnings("ignore") # remove some scikit-image warnings


from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    LoadImageD,
    EnsureChannelFirstD,
    Compose,
    OrientationD,
    OrientationD,
)

import torch
import numpy as np
import random
import sys
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

import cv2

import argparse


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

patient_useful_slices = {
    0 : {'start': 60, 'stop': 557},
    1 : {'start': 20 , 'stop': 243},
    2 : {'start': 0, 'stop': 213},
    3 : {'start': 48, 'stop': 437},
    4 : {'start': 52, 'stop': 258},
    5 : {'start': 59, 'stop': 244},
    6 : {'start': 9, 'stop': 122},
    7 : {'start': 44, 'stop': 260},
    8 : {'start': 3, 'stop': 237},
    9 : {'start': 330 , 'stop': 623},
    10 : {'start': 56, 'stop': 280},
    11 : {'start': 26, 'stop': 292},
    12 : {'start': 0, 'stop': 108},
    13 : {'start': 10, 'stop': 227},
    14 : {'start': 25, 'stop': 231},
    15 : {'start': 35, 'stop': 280},
    16 : {'start': 37, 'stop': 262},
    17 : {'start': 29, 'stop': 266},
    18 : {'start': 48, 'stop': 261},
    19 : {'start': 65, 'stop': 275},
    20 : {'start': 145, 'stop': 520},
    21 : {'start': 0, 'stop': 397},
    22 : {'start': 9, 'stop': 225},
    23 : {'start': 6, 'stop': 227},
    24 : {'start': 15, 'stop': 239},
    25 : {'start': 40, 'stop': 149},
    26 : {'start': 30, 'stop': 231},
    27 : {'start': 49, 'stop': 295},
    28 : {'start': 33, 'stop': 217},
    29 : {'start': 50, 'stop': 311},
    30 : {'start': 0, 'stop': 213},
    31 : {'start': 22, 'stop': 248},
    32 : {'start': 88, 'stop': 493},
    33 : {'start': 116, 'stop': 439},
    34 : {'start': 40, 'stop': 300},
    35 : {'start': 0, 'stop': 227},
    36 : {'start': 0, 'stop': 198},
    37 : {'start': 0, 'stop': 120},
    38 : {'start': 0, 'stop': 203},
    39 : {'start': 44, 'stop': 228},
    40 : {'start': 0, 'stop': 229},
    41 : {'start': 35, 'stop': 250},
    42 : {'start': 48, 'stop': 243},
    43 : {'start': 5, 'stop': 230},
    44 : {'start': 88, 'stop': 314},
    45 : {'start': 7, 'stop': 220},
    46 : {'start': 11, 'stop': 232},
    47 : {'start': 21, 'stop': 229},
    48 : {'start': 4, 'stop': 255},
    49 : {'start': 2, 'stop': 122},
    50 : {'start': 14, 'stop': 119}
}

patient_tumorous_slices = {
    0: {'start': 326, 'stop': 411},
    1: {'start': 171, 'stop': 227},
    2: {'start': 147, 'stop': 193},
    3: {'start': 303, 'stop': 335},
    4: {'start': 141, 'stop': 167},
    5: {'start': 131, 'stop': 188},
    6: {'start': 71, 'stop': 95},
    7: {'start': 164, 'stop': 192},
    8: {'start': 145, 'stop': 180},
    9: {'start': 513, 'stop': 549},
    10: {'start': 197, 'stop': 272},
    11: {'start': 223, 'stop': 258},
    12: {'start': 59, 'stop': 83},
    13: {'start': 59, 'stop': 99},
    14: {'start': 178, 'stop': 207},
    15: {'start': 141, 'stop': 172},
    16: {'start': 127, 'stop': 177},
    17: {'start': 192, 'stop': 221},
    18: {'start': 209, 'stop': 239},
    19: {'start': 152, 'stop': 194},
    20: {'start': 439, 'stop': 494},
    21: {'start': 291, 'stop': 341},
    22: {'start': 136, 'stop': 180},
    23: {'start': 137, 'stop': 169},
    24: {'start': 84, 'stop': 112},
    25: {'start': 82, 'stop': 109},
    26: {'start': 79, 'stop': 126},
    27: {'start': 208, 'stop': 293},
    28: {'start': 137, 'stop': 173},
    29: {'start': 120, 'stop': 149},
    30: {'start': 130, 'stop': 173},
    31: {'start': 97, 'stop': 123},
    32: {'start': 356, 'stop': 415},
    33: {'start': 203, 'stop': 287},
    34: {'start': 200, 'stop': 228},
    35: {'start': 148, 'stop': 184},
    36: {'start': 79, 'stop': 105},
    37: {'start': 66, 'stop': 100},
    38: {'start': 117, 'stop': 143},
    39: {'start': 111, 'stop': 181},
    40: {'start': 140, 'stop': 165},
    41: {'start': 80, 'stop': 109},
    42: {'start': 79, 'stop': 172},
    43: {'start': 153, 'stop': 204},
    44: {'start': 147, 'stop': 188},
    45: {'start': 113, 'stop': 142},
    46: {'start': 83, 'stop': 113},
    47: {'start': 140, 'stop': 234},
    48: {'start': 141, 'stop': 170},
    49: {'start': 99, 'stop': 120},
    50: {'start': 68, 'stop': 92}
}

dev_indices_from_training = [6, 10, 13, 37, 38, 40, 46, 48, 49]

def min_max_normalize(img):
    no_stacks = img.shape[-1]

    for i in range(no_stacks):
        img[:, :, :, i] = (img[:, :, :, i] - img[:, :, :, i].min()) / (img[:, :, :, i].max() - img[:, :, :, i].min())

    return img

def clahe_normalize(img, clahe_filter):
    no_stacks = img.shape[-1]

    for i in range(no_stacks):
        img[0, ..., i]  = torch.Tensor(clahe_filter.apply(img[0, ..., i].numpy().astype(np.uint8)))

    return img

def remove_bed(img, bbox):
    '''
    Remove the bed from the image
    '''

    # print(img.shape)

    min_img = img.min()
    new_img = torch.ones_like(img) * min_img

    new_img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], :] = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3], :]

    return new_img


def create_folder_structure(path : str, split_name : str):
    '''
    Create the folder structure for the dataset:
    - PathName/training/images
    - PathName/training/labels
    - PathName/validation/images
    - PathName/validation/labels

    or if the PathName already exists, create the folder structure for the split:
    - PathName/split_name/images
    - PathName/split_name/labels

    returns PathName/split_name/
    '''
        
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(f'{path}/training/')
        os.makedirs(f'{path}/training/images/')
        os.makedirs(f'{path}/training/labels/')
        os.makedirs(f'{path}/validation/')
        os.makedirs(f'{path}/validation/images/')
        os.makedirs(f'{path}/validation/labels/')

    if not os.path.exists(f'{path}/{split_name}/'):
        os.makedirs(f'{path}/{split_name}/')
        os.makedirs(f'{path}/{split_name}/images/')
        os.makedirs(f'{path}/{split_name}/labels/')

    return f'{path}/{split_name}/'



def process_image(img : torch.Tensor, img_idx : int, take_slices_flag : bool = False, slices : {int : {str : int}} = None, normalize_flag : bool = False, normalization_method : str = 'min-max', remove_bed_flag : bool = False, lungs_bbox : (int, int, int, int) = None) -> torch.Tensor:
    '''
    Process the image with the specified augmentations
    '''
    new_img = torch.Tensor(img)

    if take_slices_flag:
        new_img = new_img[:, :, :, slices[img_idx]['start'] : (slices[img_idx]['stop']+1)]

    if normalize_flag:
        if normalization_method == 'min-max':
            new_img = min_max_normalize(new_img)
        elif normalization_method == 'clahe':
            _clipLimit = 4.0
            clahe_filter = cv2.createCLAHE(clipLimit = _clipLimit)

            new_img = clahe_normalize(new_img, clahe_filter)
        else:
            raise ValueError('Invalid normalization method')
        
    if remove_bed_flag:
        new_img = remove_bed(new_img, lungs_bbox)

    return new_img

def process_label(label : torch.Tensor, lbl_idx : int, take_slices_flag : bool = False, slices : {int : {str : int}} = None) -> torch.Tensor:
    '''
    Process the label with the specified augmentations
    '''
    new_label = torch.Tensor(label)

    if take_slices_flag:
        # print(slices[i]['start'], slices[i]['stop'])
        new_label = new_label[:, :, :, slices[lbl_idx]['start'] : (slices[lbl_idx]['stop']+1)]

    return new_label


def save_dataset(dataset, dataset_name, split_name, take_slices_flag : bool = False, slices : {int : {str : int}} = None, normalize_flag : bool = False, normalization_method : str = 'min-max', remove_bed_flag : bool = False, lungs_bbox : (int, int, int, int) = None):
    '''
    Save the dataset to the disk per patient as two files: image and mask
    '''

    crt_path = create_folder_structure(dataset_name, split_name)

    training_flag = 'training' in split_name
    dev_path = None
    print(f'Is training: {training_flag}')
    if training_flag:
        dev_path = create_folder_structure(dataset_name, 'dev')
        print(f'dev_path: {dev_path}')

    train_idx, dev_idx = 0, 0

    for i in tqdm(range(len(dataset))):
        data = dataset[i]

        if training_flag and i in dev_indices_from_training:
            print(f'Patient {i} is in the dev set')
            filename = f'{dev_path}/images/patient_{dev_idx}.pt'
            maskname = f'{dev_path}/labels/patient_{dev_idx}.pt'
            dev_idx += 1
        else:
            filename = f'{crt_path}/images/patient_{train_idx}.pt'
            maskname = f'{crt_path}/labels/patient_{train_idx}.pt'
            train_idx += 1

        new_img = process_image(data['image'], i, take_slices_flag, slices, normalize_flag, normalization_method, remove_bed_flag, lungs_bbox)
        print(f'Patient {i} has {new_img.shape[-1]} slices')
        torch.save(new_img, filename)

        if 'label' in data.keys():
            new_label = process_label(data['label'], i, take_slices_flag, slices)
            torch.save(new_label, maskname)

    print(f'Saved {i+1} patients for {split_name} split')


def load_and_save_split(split_name : str, base_transform : Compose, download_flag : bool = False, argparser = None):
    '''
    Load the dataset and save it as tensors with the specified augmentations
    '''

    remove_bed_bbox_2d = (60, 450, 119, 425)


    print(f'Loading Raw {split_name.capitalize()} MSD Data')

    ### TRAINING DATA ###
    # From Monai: ['training', 'validation', 'test']
    crt_dataset = DecathlonDataset(root_dir = argparser.raw_data_path,
                            task = "Task06_Lung", section = split_name,
                            transform = base_transform, download = download_flag)

    print(f'Saving Processed {split_name.capitalize()} MSD Data')

    taken_slices = None
    taken_slices_flag = False
    
    if split_name == 'training':
        if argparser.take_slices == 'lungs':
            taken_slices = patient_useful_slices
            taken_slices_flag = True
        elif argparser.take_slices == 'cancer':
            taken_slices = patient_tumorous_slices
            taken_slices_flag = True


    normalize_flag = False
    normalization_method = None

    if argparser.normalize == 'min-max':
        normalize_flag = True
        normalization_method = 'min-max'
    elif argparser.normalize == 'clahe':
        normalize_flag = True
        normalization_method = 'clahe'

    remove_bed_flag = argparser.remove_bed

    save_dataset(crt_dataset, argparser.processed_data_path, split_name, take_slices_flag = taken_slices_flag, slices = taken_slices, normalize_flag = normalize_flag, normalization_method = normalization_method, remove_bed_flag = remove_bed_flag, lungs_bbox = remove_bed_bbox_2d)


def parse_args(parser):
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    ### Based on arguments, we can choose to save the dataset as tensors, or as images

    dataset_dir = f'{os.getcwd()}/datasets'

    # Argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--take_slices', type=str, default='cancer', help='Choose the slices to take from the patient: lungs or cancer (str)')
    parser.add_argument('--normalize', type=str, default='min-max', help='Choose the normalization method: min-max or clahe (str)')
    parser.add_argument('--remove_bed', type=bool, default=True, help='Choose to remove the bed from the image (bool)')
    parser.add_argument('--raw_data_path', type=str, default=f'{dataset_dir}/MSD/MedicalDecathlon/', help='Path to the raw data (str)')
    parser.add_argument('--processed_data_path', type=str, default=f'{dataset_dir}/Trimmed_MSD/', help='Path to the processed data (str)')

    args = parse_args(parser)

    print(args.raw_data_path)

    DOWNLOAD_FLAG = not os.path.exists(args.raw_data_path)

    if DOWNLOAD_FLAG:
        print('ATTENTION! DOWNLOAD_FLAG is set to True. This will download the dataset from the internet. Are you sure you want to do this?')

        print('Make sure you provide the correct path to the dataset in the datasets_path variable')

        print('Press any key to continue, or CTRL+C to exit')

        input()

    KEYS = ["image", "label"]

    base_transform = Compose([
        LoadImageD(keys=KEYS),
        EnsureChannelFirstD(keys=KEYS),
        OrientationD(keys=KEYS, axcodes='RAS'),
    ])


    test_keys = ["image"]
    test_transform = Compose([
        LoadImageD(keys=test_keys),
        EnsureChannelFirstD(keys=test_keys),
        OrientationD(keys=test_keys, axcodes='RAS'),
    ])


    # load_and_save_split('training', base_transform, DOWNLOAD_FLAG, args)
    load_and_save_split('validation', base_transform, DOWNLOAD_FLAG, args)
    # load_and_save_split('test', test_transform, DOWNLOAD_FLAG, args)