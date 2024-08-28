# # Convert images to gifs

import glob

# Possible future TODO: imageio.imread is deprecated, import imageio2 as imageio instead
import imageio
import sys
import os


# # Files tree
# 
# ```
# ./plots/architectures
# |
# +------------------images
# |                   |
# |                   +-----patientX_sliceY.png
# |
# |
# +------------------gifs/
#                     |
#                     +-----patientX.gif
# ```
# 


def get_val_patients(path : str) -> int:
    '''
    Compute the number of patients in the validation set
    '''    

    unique_patients = set()

    # Get the maximum patient id
    for file in os.listdir(path):
        if file.startswith('patient'):
            patient_id = int(file.split('patient')[-1].split('_')[0])

            unique_patients.add(patient_id)

    return list(sorted(unique_patients))


if __name__ == '__main__':
    model_name = sys.argv[1]
    val_patients = get_val_patients(f'./plots/{model_name.upper()}/images/')
    print(f'Number of patients in the validation set: {val_patients}')

    # Get the frames corresponding to each patient
    patients_frames = {i : [] for i in val_patients}

    images_directory = f'./plots/{model_name.upper()}/images/'

    for patient_id in val_patients:
        if not glob.glob(images_directory + f'patient{patient_id}_slice*.png'):
            print(f'Patient {patient_id} has no images')
            break
        else:
            # Get the images that have patient_id in their name
            patients_frames[patient_id] = glob.glob(images_directory + f'patient{patient_id}_slice*.png')
            print(f'Patient {patient_id} has {len(patients_frames[patient_id])} images')

    # Sort the frames based on the slice number
    for key in patients_frames:
        patients_frames[key] = sorted(patients_frames[key], key = lambda x: int(x.split('_slice')[-1].split('.')[0]))


    os.makedirs(f'./plots/{model_name.upper()}/gifs/', exist_ok=True)

    # Create the gifs for each patient
    for patient_id in val_patients:
        if not glob.glob(images_directory + f'patient{patient_id}_slice*.png'):
            print(f'Patient {patient_id} has no images')
            break

        gif_name = f'./plots/{model_name.upper()}/gifs/patient{patient_id}.gif'
        if os.path.exists(gif_name):
            os.remove(gif_name)

        with imageio.get_writer(f'./plots/{model_name.upper()}/gifs/patient{patient_id}.gif', mode='I') as writer:
            for filename in patients_frames[patient_id]:
                image = imageio.imread(filename)
                writer.append_data(image)


