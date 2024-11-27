import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
import random
import 
from torch.utils.data import Dataset, DataLoader


def interpolate_nan(data, interpolation_method='linear', value_type='regular', fixed_value=0, filter_size=3):
    """
    Interpolates NaN values in a 2D NumPy array using the specified interpolation method and value type.

    Parameters:
    data (numpy.ndarray): The 2D array with NaN values to be interpolated.
    interpolation_method (str): The method of interpolation. Options are:
                                - 'linear': Linear interpolation.
                                - 'cubic': Cubic interpolation.
                                - 'median': Median filter interpolation.
    value_type (str): The type of values to replace NaN values with. Options are:
                      - 'regular': Replace NaN values with regular interpolated values.
                      - 'negative': Replace NaN values with negative interpolated values.
                      - 'fixed': Replace NaN values with a fixed value.
    fixed_value (float): The fixed value to replace NaN values with if value_type is 'fixed'.
    filter_size (int): The size of the median filter window if interpolation_method is 'median'.

    Returns:
    numpy.ndarray: The 2D array with NaN values replaced based on the specified method and value type.
    """
    # Create a mask for NaN values
    nan_mask = np.isnan(data)

    # If there are no NaN values, return the original array
    if not np.any(nan_mask):
        return data

    # Get the indices of the NaN values
    nan_indices = np.argwhere(nan_mask)

    # Get the indices of the non-NaN values
    non_nan_indices = np.argwhere(~nan_mask)

    # Get the values of the non-NaN values
    non_nan_values = data[~nan_mask]

    # Create a copy of the original array to store the result
    result = data.copy()

    if interpolation_method == 'linear':
        # Linear interpolation
        interpolated_values = griddata(non_nan_indices, non_nan_values, nan_indices, method='linear')
    elif interpolation_method == 'cubic':
        # Cubic interpolation
        interpolated_values = griddata(non_nan_indices, non_nan_values, nan_indices, method='cubic')
    elif interpolation_method == 'median':
        # Median filter interpolation
        filtered_data = median_filter(data, size=filter_size)
        interpolated_values = filtered_data[nan_mask]
    else:
        raise ValueError("Invalid interpolation method. Choose from 'linear', 'cubic', or 'median'.")

    if value_type == 'regular':
        result[nan_mask] = interpolated_values
    elif value_type == 'negative':
        result[nan_mask] = -interpolated_values
    elif value_type == 'fixed':
        result[nan_mask] = fixed_value
    else:
        raise ValueError("Invalid value type. Choose from 'regular', 'negative', or 'fixed'.")

    return result



def preprocess_data(data, return_nans = False, return_torch=True):
    if not return_nans:
        data=np.abs(data)
        data=data/10000.0

    else:
        valid_mask = data < 0
        data = np.zeros_like(data, dtype=np.float32)
        data[valid_mask] = data[valid_mask] / 10000.0
        data[~valid_mask] = np.nan 
    if return_torch:
        return torch.tensor(data)
    return data






# Define custom dataset
class ImageDataset(Dataset):
    def __init__(self, files, mask_filename, mean = None, std = None, num_classes = 5, subtile_size = 10560/6, patch_size=(256, 256), stride=128, augment = False, augment_transform = None, return_imgidx = False):
        self.image_files = files
        self.opened_files = {fp:rasterio.open(fp) for fp in self.image_files}      
        self.mean = mean
        self.std = std
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes
        self.transform = augment_transform
        self.return_imgidx = return_imgidx
        with rasterio.open(files[0]) as im:
            image = im.read()
            self.subtile_size = image.shape
        self.mask_filename = mask_filename
        self.mask_reader = MaskReader(self.mask_filename, self.num_classes)
        self.mask = self.mask_reader.read_all()
        self.tile_size = self.mask.shape[-1] 

        class_count = {i:0 for i in range(num_classes)}
        self.indices=[]
        for f in self.image_files:
            for x in range(0, self.subtile_size[1], self.stride):
                for y in range(0, self.subtile_size[2], self.stride):
                    idx_dict = {'file':f, 
                                'x':x, 
                                'y':y, 
                                'transform' : 0,
                                'labels':[],
                                'augmented' : 0
                                }                        
                    self.indices.append(idx_dict)
                    if augment:
                        augment_patch = False
                        subtile_x, subtile_y = extract_integers(f)
                        labels = self.mask_reader.read_window(x, y, subtile_x, subtile_y, self.patch_size[0])
                        unique_dict = unique_counts(labels)
                        #print(unique_dict)
                        unique_values = unique_dict.keys()
                        self.indices[-1]['labels'] = unique_values
                        total_count = labels.numel()  # Number of elements in the tensor
                        
                        
                        for i in range(1,self.num_classes-1):
                            value_count = (labels == i).sum().item()
                            if value_count/total_count > 0.05:
                                #print(f'augmenting patch {x}+{subtile_x}, {y}+{subtile_y}' )
                                #print(unique_values)
                                augment_patch = True                                
                            
                        #print(f'TOTAL classe {i}:{value_count/total_count}')
                        if augment_patch:
                            for t_idx in range(1,len(transf)):
                                idx_dict = {'file':f, 
                                    'x':x, 
                                    'y':y, 
                                    'transform' : t_idx,
                                    'labels' : unique_values,
                                    'augmented' : 1
                                    }
                                self.indices.append(idx_dict)
        if augment: #TODO
            print('Augmenting data...')
                #for idx_dict in self.indices:
                #    subtile_x, subtile_y = extract_integers(idx_dict['file'])
                #    labels = read_mask(self.mask_filename, i, j, subtile_x, subtile_y, self.patch_size[0], self.num_classes)
            num_patches = {c:0 for c in range(self.num_classes)}
            for idx_dict in self.indices:
                #print(idx_dict)
                for c in range(self.num_classes):
                    #print(idx_dict['labels'].keys())
                    if c in idx_dict['labels']:
                        num_patches[c]+=1
            print(num_patches)


            area_threshold = 25
            print(len(self.image_files))
            for f in self.image_files:
                subtile_x, subtile_y = extract_integers(f)
                print(f)
                mask = self.mask_reader.read_window(0, 0, subtile_x, subtile_y, patch_size = self.subtile_size[1])
        
                oh_mask = self.mask_reader.indices_to_one_hot(mask.squeeze())
                centroids_area = {}
                for label in range(1, self.num_classes-1):
                    #print(find_region_centroids_and_areas(oh_mask[label]))
                    cx_cy_a_list = find_region_centroids_and_areas(oh_mask[label], area_threshold)
                    
                    centroids_area[label] = [(tuple(item) + (f,)) for item in cx_cy_a_list]
                #print(centroids_area)
            print(centroids_area)
            for label in centroids_area:
                if 0<label<self.num_classes-1:
                    sorted_area = sorted(centroids_area[label], key=lambda x: x[2], reverse=True)
                    print(sorted_area)
                    counter = num_patches[label]
                    last_counter = 0
                    while counter < num_patches[self.num_classes-1]//2:
                        for t_idx in range(len(transf)):
                            for cx, cy, a, f in sorted_area:
                                if a > area_threshold:                                                           
                                    idx_dict = {'file':f, 
                                        'x':cx-patch_size[0]//2 + random.randint(-patch_size[0]//4, patch_size[0]//4), 
                                        'y':cy-patch_size[1]//2 + random.randint(-patch_size[0]//4, patch_size[0]//4), 
                                        'transform' : random.randint(0,len(transf)),
                                        'labels' : label,
                                        'augmented' : 2
                                        }
                                    self.indices.append(idx_dict)
                                    counter+=1 
                        if last_counter == counter:
                            break
                        last_counter = counter

                            
                    print(f'class {label}: num regions: {counter}/{len(sorted_area)}')
            print('Data Augmentation Done.')
            
            

    def check_augmentation(self):
        non_aug = 0
        first_aug = 0
        second_aug = 0

        pre_aug = {i:0 for i in range(self.num_classes)}
        patch_per_class = {i:0 for i in range(self.num_classes)}
        patch_aug_1 = {i:0 for i in range(self.num_classes)}
        patch_aug_2 = {i:0 for i in range(self.num_classes)}
        
        for idx_dict in self.indices:

            if idx_dict['augmented']==0:
                non_aug+=1
                for c in patch_per_class:
                    if c in idx_dict['labels']:
                        patch_per_class[c]+=1
                        pre_aug[c]+=1
            if idx_dict['augmented']==1:
                first_aug +=1
                for c in patch_per_class:
                    if c in idx_dict['labels']:
                        patch_per_class[c]+=1
                        patch_aug_1[c]+=1
            if idx_dict['augmented']==2:
                second_aug +=1
                for c in patch_per_class:
                    if c == idx_dict['labels']:
                        patch_per_class[c]+=1
                        patch_aug_2[c]+=1
        print(f'Initial data class distribution: {pre_aug}, total of {sum(pre_aug.values())}')
        print(f'First augmentation: {patch_aug_1}, total of {first_aug} patches')
        print(f'Second augmentation: {patch_aug_2}, included a total of {second_aug} patches')
        print(f'Total class distribution after augmentation: {patch_per_class}, total of {sum(patch_per_class.values())}.')



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self.indices[idx]['file']
        subtile_x, subtile_y = extract_integers(f)  
        x, y = self.indices[idx]['x'], self.indices[idx]['y']
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.subtile_size[1] - self.patch_size[0]:
            x = self.subtile_size[1] - self.patch_size[0]
        if y > self.subtile_size[2] - self.patch_size[1]:
            y = self.subtile_size[2] - self.patch_size[1]
        
        window = rasterio.windows.Window(x ,y , self.patch_size[0], self.patch_size[1])            
        image = preprocess_data(self.opened_files[f].read(window = window), mean=self.mean, std=self.std) #np.float32
        labels = self.mask_reader.read_window(x, y, subtile_x, subtile_y, self.patch_size[0])

        if self.transform:
            transf_idx = self.indices[idx]['transform']
            image = self.transform[transf_idx](image)
            labels = self.transform[transf_idx](labels)
        #print(self.onehotmasks)
        #print(labels.shape)
        if not self.return_imgidx:
            return image, labels
        else:
            return image, labels, x, y, f

    def count_classes(self):
        count = classes_counts(self.image_files, self.mask_filename, self.num_classes, subtile_size = self.subtile_size)
        return count
    
    def __del__(self):
        # Close all the files when the dataset object is destroyed
        for src in self.opened_files.values():
            src.close()