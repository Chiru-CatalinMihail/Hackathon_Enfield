### Typing
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset


### Imports
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as TorchDataLoader

from monai.metrics import MeanIoU, DiceHelper
import pickle as pkl
import os


__all__ = ['save_checkpoint', 'load_checkpoint', 'train']

#################### GLOBAL VARIABLES ####################
WEIGHT_DECAY_NAME = 'weight_decay'
LEARNING_RATE_NAME = 'lr'


#################### CHECKPOINTS LOGIC ####################

def save_checkpoint(epoch, model, optimizer, loss, path):
    ''' Saves the model and optimizer state_dict, along with the loss and epoch number '''
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    

def load_checkpoint(model, optimizer, path):
    ''' Loads the model and optimizer state_dict, along with the loss and epoch number '''
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

#################### VALIDATION LOOP ######################
def validate(model : nn.Module, optimizer : torch.optim.Optimizer,
            device : torch.device, val_loader : TorchDataLoader,
            checkpoints_path : str, general_name : str, writer : SummaryWriter, 
            dice_metric : DiceHelper, iou_metric : MeanIoU, dice_values : list, iou_values : list,
            epoch_loss : float, epoch : int, MAX_EPOCHS : int, VALIDATION_INTERVAL : int,
            best_dice : float, best_metrics : tuple, best_metric_epoch : int, best_model_name : str):

    # Save current checkpoint of the network
    print(f"Saving checkpoint: {epoch//VALIDATION_INTERVAL} / {MAX_EPOCHS//VALIDATION_INTERVAL}!!!")
    name = checkpoints_path + f'{general_name}_epoch{epoch}.pth'
    save_checkpoint(epoch, model, optimizer, epoch_loss, name)

    # Turn model to "eval" mode
    model.eval()

    # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
    # It will reduce memory consumption for computations that would otherwise have requires_grad=True
    with torch.no_grad():
        iteration_dice = []
        # iteration_ious = []
        # iteration_pixel_accuracies = []
        # iteration_rvds = []

        for val_data in val_loader:
            val_input, val_label = val_data
            val_input, val_label = val_input.to(device), val_label.to(device)

            val_output = model(val_input)
            val_output = nn.Softmax(dim=1)(val_output)

            # Compute metrics for current iteration
            iteration_dice.append(dice_metric(y_pred = val_output, y = val_label).item())
            iou_metric(y_pred= val_output, y=val_label)

    # Aggregate the final mean results
    dice_score = torch.mean(torch.tensor(iteration_dice)).item()
    mean_iou = iou_metric.aggregate().item()

    # Reset the status for the next epoch
    # dice_metric.reset()
    iou_metric.reset()

    dice_values.append(dice_score)
    iou_values.append(mean_iou)

    writer.add_scalar('Dice/val', dice_score, epoch)
    writer.add_scalar('IoU/val', mean_iou, epoch)

    if dice_score > best_dice:
        best_dice = dice_score
        best_metrics = (dice_score, mean_iou)
        best_metric_epoch = epoch
        print("saved new best metric model!!!")

        save_checkpoint(epoch, model, optimizer, epoch_loss, best_model_name)

    print(
        f"current epoch: {epoch},"
        f" current mean dice: {dice_score:.4f},"
        f" current mean iou: {mean_iou:.4f},"
        f" best mean dice: {best_dice:.4f},"
        f" at epoch: {best_metric_epoch}"
    )

    return best_dice, best_metrics, best_metric_epoch


##################### TRAINING LOOP #######################
def train(model : nn.Module, checkpoints_path : str, model_name : str, 
        loss_function : _Loss, loss_key : str, lr_scheduler : LRScheduler,
        optimizer : torch.optim.Optimizer, optimizer_key : str, device : torch.device,
        train_dataset : Dataset, train_loader : TorchDataLoader, val_loader : TorchDataLoader,
        MAX_EPOCHS : int = 2, VALIDATION_INTERVAL : int = 1, EPOCH_OFFSET : int = 0):
 
    no_batches = len(train_loader)
    
    lr_val = optimizer.defaults[LEARNING_RATE_NAME]
    wd_val = optimizer.defaults[WEIGHT_DECAY_NAME]

    # Variables to get the best model
    best_dice, best_metrics, best_metric_epoch = -1, None, -1

    general_name = f'{model_name}_{optimizer_key}_lr{lr_val:.2e}_wd{wd_val:.2e}_{loss_key}loss'
    best_model_name = checkpoints_path + f'{general_name}_best.pth'

    print(best_model_name)
    writer = SummaryWriter(log_dir=f"./pytorch_logging/{general_name}_epochs{MAX_EPOCHS}")
    os.makedirs(checkpoints_path, exist_ok=True)

    # Performance Metrics Computers
    dice_metric = DiceHelper(include_background = False, reduction = "mean", get_not_nans=False, ignore_empty=True) # include_background = False,
    iou_metric = MeanIoU(include_background=False, reduction = "mean", get_not_nans=False, ignore_empty=True)


    # Evaluation metrics per epoch
    train_epoch_losses, dice_values, iou_values = [], [], []
    iou_values = []

    epoch_loss_values = []

    for epoch in range(1 + EPOCH_OFFSET, MAX_EPOCHS + 1):
        print("-" * 12)
        print(f"Epoch {epoch}/{MAX_EPOCHS}")

        # Turn model to "train" mode
        model.train()

        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            train_input, label = batch_data
            train_input, label = train_input.to(device), label.to(device)

            optimizer.zero_grad() # Clear gradients
            output = model(train_input)

            # TODO: Aici putem sa facem mixed precision training
            loss = loss_function(output, label)
            loss.backward() # Compute current gradient

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step() # Update model's parameters

            epoch_loss += loss.item()
            print(f"{step + 1}/{no_batches}, train_loss: {loss.item():.4f}")

        epoch_loss /= no_batches
        train_epoch_losses.append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")

        # if epoch % 5 == 2:
        #     # Decay learning rate
        #     lr_scheduler.step()

        if epoch % VALIDATION_INTERVAL == 0:
            best_dice, best_metrics, best_metric_epoch = validate(model, optimizer, device, val_loader,
                                                                  checkpoints_path, general_name, writer,
                                                                  dice_metric, iou_metric, dice_values, iou_values,
                                                                  epoch_loss, epoch, MAX_EPOCHS, VALIDATION_INTERVAL,
                                                                  best_dice, best_metrics, best_metric_epoch, best_model_name)


        train_dataset.shuffle()

    print(
        f"train completed, metrics correspondic to best dice are: dice: {best_metrics[0]:.4f}, iou: {best_metrics[1]:.4f}" #, acc: {best_metrics[2]:.4f}, rvd: {best_metrics[3]:.4f}"
        f" at epoch: {best_metric_epoch}"
    )

    with open(checkpoints_path + f'{general_name}_metrics_evolution.pkl', 'wb') as f:
        pkl.dump((dice_values, iou_values, epoch_loss_values), f)

    writer.close()

    return best_model_name
