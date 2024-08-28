### Typing
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset


### Imports
import sys
import numpy as np
import pickle as pkl
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

import json

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as TorchDataLoader

from monai.metrics import MeanIoU, DiceHelper

from .debug_utils import plot_prediction_label_side_by_side
from .postprocessing import hard_threshold_labels
from .metrics import precision_score_, recall_score_, dice_coef, iou

from monai.transforms import Compose, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

__all__ = ['eval', 'generate_predictions']

def eval(val_net : nn.Module,
        msd_val_dataset : Dataset,
        val_loader : TorchDataLoader,
        device : torch.device,
        qualitative_plots_flag : bool = False,
        save_qualitative_plots_flag : bool = False,
        debug_network_flag : bool = False,
        model_name : str = 'unet'):

    '''
    Evaluate the network on the validation set

    Parameters:
    :val_net: nn.Module - The network loaded from a checkpoint
    :msd_val_dataset: Dataset - The validation dataset
    :val_loader: TorchDataLoader - Dataloader for the validation set
    
    :qualitative_plots_flag: bool - If True, the function will plot the prediction and label side by side for the first batch of the validation set
    :save_qualitative_plots_flag: bool - If True, the function will save the predictions overlaid on the images for the validation patients, in order to generate gifs later
    :debug_network_flag: bool - Used for qualitative evaluation of the network on gifs for the first two patients from the validation set
    '''


    dice_metric_eval = DiceHelper(include_background = False, reduction = "mean", get_not_nans=False, ignore_empty=True) # include_background = False,
    iou_metric_eval = MeanIoU(include_background=False, reduction = "mean", get_not_nans=False, ignore_empty=True)

    no_val_iters = 0
    loss_val = []
    dice_val = []
    iou_val = []
    precision_val = []
    recall_val = []

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    save_val_data = [] # Used for debugging the source of low dice scores

    with torch.no_grad():
        val_net.eval()

        count_plots = 0
        idx = 0

        for batch in val_loader:
            val_img, val_label = batch
            val_img, val_label = val_img.to(device), val_label.to(device)

            val_output = val_net(val_img)
            val_output = nn.Softmax(dim=1)(val_output)

            # roi_size = (512, 512, 2)
            # sw_batch_size = 1
            # val_outputs_dice = sliding_window_inference(val_img, roi_size, sw_batch_size, val_net)

            # val_outputs_dice = []
            # for val_crt_output in decollate_batch(val_outputs_dice):
            #     print(val_crt_output == None)
            #     print(val_crt_output.shape)
            #     val_outputs_dice.append(post_pred(nn.Softmax(dim=1)(val_crt_output)))

            # val_label_dice = []
            # for crt_label in decollate_batch(val_label):
            #     val_label_dice.append(post_label(crt_label))

            # compute metric for current iteration
            dice_val.append(dice_metric_eval(y_pred=val_output, y=val_label).item())
            # dice_val.append(dice_coef(val_label[1].to('cpu').detach().numpy(), val_output[1].to('cpu').detach().numpy()))
            # precision_val.append(precision_score_(val_label[1].to('cpu').detach().numpy(), val_output[1].to('cpu').detach().numpy()))
            # recall_val.append(recall_score_(val_label[1].to('cpu').detach().numpy(), val_output[1].to('cpu').detach().numpy()))
            # iou_val.append(iou(val_label[1].to('cpu').detach().numpy(), val_output[1].to('cpu').detach().numpy()))
            iou_metric_eval(y_pred=val_output, y=val_label)
            # print("Dice score : ", dice_metric_eval.aggregate().item())
            # print("IoU score : ", iou_metric_eval.aggregate().item())
            # print("Loss : ", loss_function(val_output, val_label).item())

            # dice_val.append(dice_metric_eval.aggregate().item())
            # loss_val.append(loss_function(val_output, val_label).item())
            # iou_val.append(iou_metric_eval.aggregate().item())
            no_val_iters += 1


            # if recall_val[-1] > 0.8:
            #     save_val_data.append((val_img, val_label, val_output, recall_val[-1]))
            

            if qualitative_plots_flag:
                plot_prediction_label_side_by_side(val_img, val_label, val_output, threshold = 0.5)
                break

            if save_qualitative_plots_flag:

                for i in range(val_img.shape[0]):

                    if idx == len(msd_val_dataset):
                        sys.exit()

                    print(f"Current index: {idx} / {len(msd_val_dataset)}")
                    # Don't save images for the padding indices
                    # Only iterate through the unique slices for each patient
                    patient_id, slices = msd_val_dataset.stacks_in_order_indices[idx]
                    slices = np.unique(slices)
                    print(f"Patient id: {patient_id}, slices: {slices}")
                    
                    if debug_network_flag and patient_id > 1:
                        sys.exit()

                    im = val_img[i].to('cpu').detach().numpy()
                    target = val_label[i].to('cpu').detach().numpy()
                    output = val_output[i].to('cpu').detach().numpy()

                    # Normalize the output to 0 or 1
                    # output = hard_threshold_labels(output)

                    idx += 1
                    for j in range(len(slices)):
                        # plt.imshow(im[0, ..., j], cmap = 'gray')
                        # plt.imshow(output[1, ..., j], cmap = 'jet', alpha = 0.5)
                        # plt.gca().set_axis_off()

                        # plt.savefig(f'./plots/{model_name.upper()}/images/patient{patient_id}_slice{slices[j]}.png')
                        # plt.close()

                        # Comparative plot for training with CAMs

                        # print(slices[j])
                        # print(im.shape)

                        crt_slice = slices[j] % im.shape[-1]

                        _, ax = plt.subplots(1, 3, figsize = (10, 5))

                        ax[0].imshow(im[0, ..., crt_slice], cmap = 'gray')
                        ax[0].imshow(im[1, ..., crt_slice], cmap = 'jet', alpha = 0.5)
                        ax[0].set_axis_off()
                        ax[0].set_title('CAM Overlay')
                        ax[1].imshow(im[0, ..., crt_slice], cmap = 'gray')
                        ax[1].imshow(output[1, ..., crt_slice], cmap = 'jet', alpha = 0.5)
                        ax[1].set_axis_off()
                        ax[1].set_title('Prediction Overlay')
                        ax[2].imshow(im[0, ..., crt_slice], cmap = 'gray')
                        ax[2].imshow(target[0, ..., crt_slice], cmap = 'jet', alpha = 0.5)
                        ax[2].set_axis_off()
                        ax[2].set_title('Label Overlay')

                        plt.savefig(f'./plots/{model_name.upper()}/images/patient{patient_id}_slice{slices[j]}.png')
                        plt.close()

                        if slices[j] == im.shape[-1] - 1:
                            break

        # Aggregate the final mean results
        dice_score_eval = torch.mean(torch.tensor(dice_val)).item()
        mean_iou_eval = iou_metric_eval.aggregate().item()
        # mean_iou_eval = torch.mean(torch.tensor(iou_val)).item()
        p_score_eval = np.mean(precision_val)
        r_score_eval = np.mean(recall_val)


        # # Reset the status
        # dice_metric_eval.reset()
        # iou_metric_eval.reset()

        print(f"Evaluation metrics: dice: {dice_score_eval:.4f}, iou: {mean_iou_eval:.4f}, precision: {p_score_eval:.4f}, recall: {r_score_eval:.4f}")

        return (dice_val, iou_metric_eval, p_score_eval, r_score_eval, loss_val, no_val_iters), save_val_data
    


def get_test_predictions(net : nn.Module, msd_test_dataset : Dataset, test_loader : TorchDataLoader, device : torch.device) -> [torch.Tensor]:
    '''
    Use the network to generate predictions for the test set by creating a list of zero tensors of the same size as the patients and filling them with the network's output
    '''

    test_predictions = [torch.zeros_like(patient) for patient in msd_test_dataset.patients]

    with torch.no_grad():
        net.eval()

        idx = 0

        for batch in tqdm(test_loader):
        
            test_img = batch.to(device)

            test_output = net(test_img)
            test_output = nn.Softmax(dim=1)(test_output)


            for i in range(test_output.shape[0]):

                # Only iterate through the unique slices for each patient
                # And append the predictions to their corresponding place
                patient_id, slices = msd_test_dataset.stacks_in_order_indices[idx]
                # print(patient_id, slices)

                idx += 1
                slices = np.unique(slices)

                test_predictions[patient_id][:, :, :, slices] = test_output[i, 1, ..., :len(slices)].to('cpu').detach()

    return test_predictions



def keep_only_names(names_lists):
    for i in range(len(names_lists)):
        names_lists[i] = names_lists[i].split('/')[-1]
    return names_lists

def generate_predictions(net : nn.Module, msd_test_dataset : Dataset, test_loader : TorchDataLoader, device : torch.device,
                         msd_dataset_path : str = './datasets/MSD', model_name : str = 'unet'):
    '''
    Generate predictions for the test set and save them as nifti files, according to the original MSD dataset json file
    '''

    with open(f'{msd_dataset_path}/MedicalDecathlon/Task06_Lung/dataset.json') as crt_file:
        dataset_json = json.load(crt_file)

    # Remove '/imagesTs/' from the names
    # test_names list contains: ['lung_002.nii.gz', 'lung_007.nii.gz', ...]
    test_names = keep_only_names(dataset_json['test'])

    test_predictions = get_test_predictions(net, msd_test_dataset, test_loader, device)

    for i in tqdm(range(len(test_predictions))):
        # Through PyDicom, we can observe that the shape of the original images is (512, 512, no_slices)
        crt_prediction = test_predictions[i].squeeze(0)
        final_prediction = nib.Nifti1Image(crt_prediction.numpy(), np.eye(4)) #TODO: Check if the affine is correct

        original_image = nib.load(f'{msd_dataset_path}/MedicalDecathlon/Task06_Lung/imagesTs/{test_names[i]}')
        assert crt_prediction.shape == original_image.shape, f"Prediction patient {i} is not alligned with the original image"

        nib.save(final_prediction, f'./predictions/{model_name.upper()}/Task06_Lung/{test_names[i]}')