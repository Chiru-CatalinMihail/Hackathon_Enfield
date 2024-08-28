import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

from .normalization import min_max_normalization

__all__ = ['plot_pca_analysis', 'calculate_mean_components', 'get_pacients_pca_components']


# ----------------------------------------------------- # TODO: Pus un nume aici


def plot_pca_analysis(img, var_threshold):
    image_sum = img
    print(image_sum.shape)

    image_bw = min_max_normalization(img)
    print(image_bw.max())

    pca = PCA()
    pca.fit(image_bw)

    # Getting the cumulative variance
    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

    # How many PCs explain x% of the variance?
    # var_threshold = 70 / 90 / 95

    k = np.argmax(var_cumu>var_threshold)
    print(f"Number of components explaining {var_threshold}% variance: "+ str(k))

    plt.figure(figsize=[10,5])
    plt.title('Cumulative Explained Variance by the number of components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axvline(x=k, color="k", linestyle="--")
    plt.axhline(y=var_threshold, color="r", linestyle="--")
    ax = plt.plot(var_cumu)
    plt.show()
    
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
    image_recon = 255*min_max_normalization(image_recon)
    
    
    fig, axes = plt.subplots(1,2, figsize=[12,8])
    for ax in axes:
        ax.set_axis_off()
        
                
    axes[0].imshow(image_recon)
    axes[0].set_title('Reconstructed Image from PCA')
    
    axes[1].imshow(img)
    axes[1].set_title('Original Image')
    plt.show()



# ----------------------------------------------------- # TODO: Pus un nume aici

def images_pcs_for_threshold(image, var_threshold= 99):
    image_bw = min_max_normalization(image)

    pca = PCA()
    pca.fit(image_bw)

    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

    k = np.argmax(var_cumu>var_threshold)
    
    return k

def calculate_mean_components(patients):
    
    dataset_len = len(patients)

    components = []
    for i in range(dataset_len):
        print(f"Patient {i}/{dataset_len}")

        dt = patients[i]


        img = dt['image'].squeeze(0)
        for slc in tqdm(range(img.shape[-1])):
            print(f"Slice {slc}/{img.shape[-1]}")

            components.append(images_pcs_for_threshold(img[..., slc]))
        
    return components


# ----------------------------------------------------- # TODO: Pus un nume aici

def pca_singural_values(image, taken_components= 150):
    image_bw = min_max_normalization(image)

    pca = PCA()
    pca.fit(image_bw)

    return pca.singular_values_[:taken_components]

def get_patients_pca_components(patients, random_indices):
    
    patient_components = []

    dataset_len = len(patients)
    
    for idx, dt in enumerate(patients):
        print(f"Patient {idx}/{dataset_len}")
        
        img = dt['image'].squeeze(0)
        for slc in range(img.shape[-1]):
            print(f"Slice {slc}/{img.shape[-1]}")
            crt_pacient = [random_indices[idx], slc]
            crt_pacient.append(pca_singural_values(img[..., slc]))
            patient_components.append(crt_pacient)
            
    
    return patient_components