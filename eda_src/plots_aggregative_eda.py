import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


__all__ = ['render_image', 'show_slices', 'plots_slices_per_patient_split', 'plots_tumor_slices_per_patient_split', 'plot_patients_heatmaps', 'plot_aggregative_heatmaps']

def show_slices(crt_loader):
    ''' Plots consecutively the first slice of the stack for all the patients in the loader '''
    for patient_batch in tqdm(crt_loader):
        img = patient_batch['image'][0].numpy()
        label = patient_batch['label'][0].numpy()

        plt.imshow(img[:, :, 0], cmap='gray')
        plt.imshow(label[:, :, 0], cmap='jet', alpha=0.5)
        plt.gca().axis('off')
        plt.show()


def render_image(patient, img_slice):
    plt.imshow(patient[0, ..., img_slice])
    plt.axis('off')
    plt.show()

def plots_bar_hist_interface(counts_list, split_name, counts_name, verbose_flag = False):
    '''
    Interface function used for quantitative measures on slices per patient or tumorous slices per patient

    1 x 2 grid of plots:
    - Barplot: Number of quantity in CT for each patient based on patients ID
    - Histogram: Count of patients with at least determined quantity in CT
    '''

    NO_PATIENTS = len(counts_list)
    mean_counts = np.mean(counts_list)
    std_counts = np.std(counts_list)

    print(f'Mean number of {counts_name}: {mean_counts}')
    print(f'Std for the number of {counts_name}: {std_counts}')

    fig, ax = plt.subplots(1,2, figsize=(15, 5))

    # Slices
    ax[0].bar(range(NO_PATIENTS), counts_list)

    # Add the number of slices as text on top of each bar
    if verbose_flag:
        for i in range(NO_PATIENTS):
            ax[0].text(i, counts_list[i], f'{counts_list[i]}', ha='center')

    # Add the mean as a red line, starting from 0 to NO_PATIENTS
    ax[0].axhline(mean_counts, color='red', label='Mean')

    # Add variance as a yellow faded area
    ax[0].fill_between(np.arange(-0.5, NO_PATIENTS+0.5), mean_counts - std_counts, mean_counts + std_counts, color='yellow', alpha=0.5, label='Variance')

    ax[0].set_xlim(-1, NO_PATIENTS)
    ax[0].set_xlabel('Patient')
    ax[0].set_ylabel(f'Number of {counts_name} in CT')
    ax[0].set_title(f'{split_name.capitalize()}: Number of {counts_name} in CT for each patient')
    ax[0].legend()

    # Slices
    plt.hist(counts_list, bins=10)

    # Add the mean as a red line
    ax[1].set_xlabel(f'Number of {counts_name} in CT')
    ax[1].set_ylabel(f'Number of patients with {counts_name} in CT')
    ax[1].set_title(f'{split_name.capitalize()}: Number of {counts_name} in CT for each patient - Histogram')
    fig.show()
    plt.show()

    return mean_counts, std_counts


def plots_slices_per_patient_split(slices_per_patient, split_name, verbose_flag = False):
    '''
    1 x 2 grid of plots:
    - Barplot: Number of slices in CT for each patient based on patients ID
    - Histogram: Count of patients with at least number of slices in CT
    '''

    return plots_bar_hist_interface(slices_per_patient, split_name, 'slices', verbose_flag)


def plots_tumor_slices_per_patient_split(stacks_full_volume, split_name, verbose_flag = False):
    '''
    1 x 2 grid of plots:
    - Barplot: Number of tumorous stacks in CT for each patient based on patients ID
    - Histogram: Count of patients with at least number of tumorous stacks in CT
    '''

    return plots_bar_hist_interface(stacks_full_volume, split_name, 'tumorous stacks', verbose_flag)


def plot_patients_heatmaps(bins_2d, name, bbox_flag = False, bbox = None):
    '''
    Plots the heatmaps of the tumorous pixels for each patient in the dataset, bordered by the bounding box if bbox_flag is True

    Can be used for the matrices: bins_2d or zoomed_bins_2d
    '''

    NO_PATIENTS = len(bins_2d)
    #  Create heatmap for each patient
    for i in range(NO_PATIENTS):
        plt.title(f'{name.capitalize()}: Patient {i}')
        plt.imshow(bins_2d[i], cmap='hot')

        if bbox_flag:
            h_min, h_max, w_min, w_max = bbox
            plt.gca().add_patch(plt.Rectangle((w_min, h_min), w_max - w_min, h_max - h_min, fill=False, edgecolor='blue', lw=2))

        plt.gca().axis('off')
        plt.show()


def plot_aggregative_heatmaps(bins, binning_volume, name, bbox_flag = False, bbox = None, sort_volumes_flag = False):
    '''
    1 x 2 grid of plots:
    - Plots the heatmap of the tumorous 2D slices for all the patients in the dataset split aggregated in one frame.
        Can be used for the matrices: bins_2d or zoomed_bins_2d
        
    - Plots the volume distribution of the tumorous slices for all the patients in the dataset split
    '''

    color = 'g'

    NO_PATIENTS = len(bins)
    fig, ax = plt.subplots(1,2, figsize=(20, 5))

    ax[0].set_title(f'{name.capitalize()}: Patients\' cummulative tumours in 2D')
    img0 = ax[0].imshow(np.sum(bins, axis=0), cmap='hot')

    if bbox_flag:
        h_min, h_max, w_min, w_max = bbox
        ax[0].add_patch(plt.Rectangle((w_min, h_min), w_max - w_min, h_max - h_min, fill=False, edgecolor=color, lw=2))

    # Make colorbar the same size as the image
    plt.colorbar(img0, ax=ax[0], pad = 0.005)
    ax[0].set_xlabel('Patch width')
    ax[0].set_ylabel('Patch height')
    ax[0].axis('off')
    # ax[0].axis('tight')
    # ax[0].show()
    # fig.show()
    
    if sort_volumes_flag:
        # For each patient, determine the first non-zero bin
        first_non_zero_bin = []

        for i in range(binning_volume.shape[0]):
            non_zero_elems = np.nonzero(binning_volume[i])
            first_non_zero_bin.append((i, non_zero_elems[0][0].item(), len(non_zero_elems[0])))


        # If the min index is the same, sort ascending by the number of non-zero elements
        sorted_bins = sorted(first_non_zero_bin, key = lambda x: (x[1], x[2]))
        sorted_idx = [x[0] for x in sorted_bins]
        
        for tpl in sorted_bins:
            print(tpl)
        
        binning_volume = binning_volume[sorted_idx]
        

    bin_with_most_tumors = np.argmax(binning_volume.sum(axis=0))

    # Diminish space between plots
    plt.subplots_adjust(wspace=-0.075)
    
    # Make ax[1] bigger on x-axis
    ax[1].figure.set_size_inches(50, 5, forward=True)

    ax[1].set_title(f'{name.capitalize()}: Patients volume tumour distribution')
    # Make the cells bigger
    img1 = ax[1].imshow(binning_volume.T, cmap='hot', aspect='auto', interpolation='nearest')

    # Move the colorbar closer to its heatmap
    plt.colorbar(img1, ax=ax[1], pad=0.005) #fraction=0.046, 

    yticks=range(10)
    # Make y start from 0
    ax[1].invert_yaxis()
    ax[1].set_xticks(range(NO_PATIENTS))
    ax[1].set_yticks(yticks)
    ax[1].axhline(bin_with_most_tumors - 1, color=color, linestyle='-', markersize=3)
    ax[1].axhline(bin_with_most_tumors + 1, color=color, linestyle='-', markersize=3)
    ax[1].axis('tight')
    ax[1].set_xlabel('Patient')
    ax[1].set_ylabel('Tumour found in volume (%)')

    fig.show()
    plt.show()