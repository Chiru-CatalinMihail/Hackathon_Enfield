import matplotlib.pyplot as plt
import numpy as np

__all__ = ['debug_plot', 'debug_plot_single', 'plot_prediction_label_side_by_side']

def debug_plot(img, mask):
    '''
    Plots the image and mask as two subplots, for how many stacked images there are
    '''

    # 6x2 subplots, with reduced vertical space
    
    fig, axes = plt.subplots(img.shape[-1], 2, figsize=(10, 10))


    for i in range(img.shape[-1]):
        axes[i, 0].invert_yaxis()
        axes[i, 1].invert_yaxis()
        axes[i, 0].imshow(img[0, ..., i], cmap='gray')
        axes[i, 1].imshow(img[0, ..., i], cmap='gray')
        axes[i, 1].imshow(mask[0, ..., i], alpha=0.5, cmap='jet')
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.subplots_adjust(wspace=0)    
    plt.show()

def debug_plot_single(img, mask):
    a = img[0].to('cpu').detach().numpy()
    b = mask[0].to('cpu').detach().numpy()
    for i in range(img.shape[-1]):
        plt.imshow(a[0, ..., i], cmap = 'gray')
        plt.imshow(b[0, ..., i], cmap = 'jet', alpha = 0.5)
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().set_axis_off()
        plt.show()
        

def plot_prediction_label_side_by_side(img, label, prediction, threshold = 0.5):
    '''
    Plots a 2 x 3 grid with the image, label and prediction side by side on the first row.

    On the second row, the image is plotted with the label and prediction overlayed.
    '''

    for i in range(img.shape[0]):
        print(i)
        if i > 1:
            break
        im = img[i].to('cpu').detach().numpy()
        target = label[i].to('cpu').detach().numpy()
        output = prediction[i].to('cpu').detach().numpy()
        # output = hard_threshold_labels(output, threshold, cutoff_flag = False)

        for j in range(img.shape[-1]):
            fig, ax = plt.subplots(2, 3, figsize=(15, 5))

            ax[0, 0].imshow(im[0, ..., j], cmap = 'gray')
            ax[0, 1].imshow(target[0, ..., j], cmap = 'jet')
            im1 = ax[0, 2].imshow(output[1, ..., j], cmap = 'jet')
            ax[1, 0].imshow(im[0, ..., j], cmap = 'gray')
            ax[1, 1].imshow(im[0, ..., j], cmap = 'gray')
            ax[1, 1].imshow(target[0, ..., j], cmap = 'jet', alpha = 0.5)
            ax[1, 2].imshow(im[0, ..., j], cmap = 'gray')
            im2 = ax[1, 2].imshow(output[1, ..., j], cmap = 'jet', alpha = 0.5)

            plt.colorbar(im1, ax=ax[0, 2])
            plt.colorbar(im2, ax=ax[1, 2])  

            for k in range(2):
                for l in range(3):
                    ax[k, l].invert_yaxis()
                    ax[k, l].set_axis_off()

                    if k == 0:
                        ax[k, l].set_title(['Image', 'Label', 'Prediction'][l])
                    if i == 1:
                        ax[k, l].set_title(['Image', 'Label overlay', 'Prediction overlay'][l])


            # TODO: In alta zi, fa tight layout
            fig.tight_layout()
            plt.show()


def plot_large_grid(vect):
    ''' Plots a large grid of images, with shape (rows, cols)

    Where rows * cols >= no_of_ct_slices
    rows = sqrt(no_of_ct_slices) + 1
    cols = no_of_ct_slices / rows + 1 
    '''

    length = vect.shape[-1]

    rows = int(np.sqrt(length)) + 1
    cols = int(length / rows) + 1
    
    print(rows, cols)
    
    
    fig, ax = plt.subplots(rows, cols)

    fig.set_figheight(15)
    fig.set_figwidth(15)

    break_flag = False

    for i in range(rows):
        if break_flag:
            break

        for j in range(cols):
            crt_idx = cols*i + j 
            if crt_idx >= length:
                break_flag = True
                break

            ax[i, j].imshow(vect[0, ..., crt_idx])
            ax[i, j].axis("off")

    plt.tight_layout()        
    plt.show()
