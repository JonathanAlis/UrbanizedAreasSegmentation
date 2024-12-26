import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
import random
import rasterio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import re
import os
import yaml
import math
from sklearn.model_selection import StratifiedShuffleSplit

import src.data.mask_processing as mask_processing

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
        
        valid_mask = data >= 0
        data_with_nan = np.zeros_like(data, dtype=np.float32)
        data_with_nan[valid_mask] = data[valid_mask] / 10000.0
        data_with_nan[~valid_mask] = np.nan
        data = data_with_nan 
    if return_torch:
        return torch.tensor(data)
    return data




class MaskReader:
    def __init__(self, mask_filename, classes_mode = 'type'):
        '''
        Classes modes: 
        all: 0 to 9, considers all combinations of types and densities
        type: all 5 types
        densities: all 4 densities
        binary: 0 or 1 
        '''
        self.mask_filename = mask_filename
        self.mask_data = rasterio.open(self.mask_filename)
        self.classes_mode = classes_mode

    def transform_to_class_mode(self, m):
        if self.classes_mode.lower() == 'type':
            m = mask_processing.get_type(m)
        elif self.classes_mode.lower() == 'density':
            m = mask_processing.get_density(m)
        elif self.classes_mode.lower() == 'binary':
            m = mask_processing.get_binary(m)
        else:
            pass# if type_density, or any other, do not change
        return m  

    def read_window(self, x, y, patch_size, return_torch = True):
        window = rasterio.windows.Window(x, y, patch_size, patch_size) 
        m = self.mask_data.read(window = window)
        m = self.transform_to_class_mode(m)
        if return_torch:
            m = torch.from_numpy(m)
        return m

    def read_all(self,return_torch = True):
        m = self.mask_data.read()
        m = self.transform_to_class_mode(m)
        if return_torch:
            m = torch.from_numpy(m)
        return m
    
    def indices_to_one_hot(self, indices):
        indices = indices.squeeze()
        one_hot = np.zeros((indices.shape[0], indices.shape[1], self.num_classes))
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                one_hot[i, j, indices[i, j]] = 1
        return one_hot
    



def get_x_y(string):
    # Use regular expressions to find the patterns 'x=integer' and 'y=integer' (ignoring decimal points and non-digit characters after the numbers)
    x_match = re.search(r'x=(\d+)', string)
    y_match = re.search(r'y=(\d+)', string)
    
    # Extract the numbers as integers, or return None if not found
    x_value = int(x_match.group(1)) if x_match else None
    y_value = int(y_match.group(1)) if y_match else None
    
    return x_value, y_value


def get_tile(string, pattern = "S2-16D_V2_"):
    # Use regular expressions to find the patterns for the tile (ignoring decimal points and non-digit characters after the numbers)
    match = re.search(f'{pattern}(\d+)', string)
    
    tile = match.group(1) if match else None
    
    return tile



def unique_counts(labels):
    class_count_dict, counts = np.unique(labels, return_counts=True)
    class_count_dict = {int(class_) : int(counter_) for class_, counter_ in zip(class_count_dict, counts)}
    return class_count_dict 


class DiagonalFlip1:
    def __call__(self, tensor):
        # Diagonal Flip 1: Transpose the tensor (flip along the top-left to bottom-right diagonal)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Swap the first two dimensions (C, H, W)

class DiagonalFlip2:
    def __call__(self, tensor):
        # Diagonal Flip 2: Flip horizontally and then transpose
        tensor = tensor.flip(2)  # Flip along the width (last dimension)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Then transpose
    

transf = {0 : None,
           1 : transforms.RandomRotation(degrees=(90, 90)),
           2 : transforms.RandomRotation(degrees=(180, 180)),
           3 : transforms.RandomRotation(degrees=(270, 270)),
           4 : transforms.RandomVerticalFlip(p=1.0),
           5 : transforms.RandomHorizontalFlip(p=1.0),
           6 : DiagonalFlip1(),
           7 : DiagonalFlip2(),
           }

# Define custom dataset
class ImageSubtileDataset(Dataset):
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
                        subtile_x, subtile_y = utils.extract_integers(f)
                        labels = self.mask_reader.read_window(x+subtile_x, y+subtile_y, self.patch_size[0])
                        unique_dict = utils.unique_counts(labels)
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
                subtile_x, subtile_y = utils.extract_integers(f)
                print(f)
                mask = self.mask_reader.read_window(subtile_x, subtile_y, patch_size = self.subtile_size[1])
        
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
        labels = self.mask_reader.read_window(x+subtile_x, y+subtile_y, self.patch_size[0])

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

# Define custom dataset
class SubtileDataset(Dataset):
    def __init__(self, files, num_subtiles, classes_mode = 'type', patch_size=(256, 256), stride=128, augment = False, augment_transform = None, return_imgidx = False, return_nans = False, debug = True):
        self.image_files = files
        self.opened_files = {fp:rasterio.open(fp) for fp in self.image_files}      
        self.patch_size = patch_size
        self.stride = stride
        self.num_subtiles = num_subtiles
        self.working_dir = os.path.abspath('..')
        self.classes_mode = classes_mode
        self.return_nans = return_nans
        #self.masks = self.get_masks()
        self.mask_params = self.get_mask_params()

        self.transform = augment_transform
        self.return_imgidx = return_imgidx
        self.debug = debug

        with rasterio.open(files[0]) as im:
            image = im.read()
            self.subtile_size = image.shape

        self.indices=[]
        for f, mp in zip(self.image_files, self.mask_params):
            for x in range(0, self.subtile_size[1], self.stride):
                for y in range(0, self.subtile_size[2], self.stride):
                    idx_dict = {'file':f, 
                                'x':x, 
                                'y':y, 
                                'transform' : 0,
                                'mask_params':mp,
                                'augmented' : 0
                                }                        
                    self.indices.append(idx_dict)
                    if augment:
                        pass

    def get_mask(self, file, x = 0, y = 0):    
        
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        print('x, y: ', x, y, 'subtiles:', subtile_x, subtile_y)
        mask_reader = MaskReader(os.path.join(self.working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                    classes_mode = self.classes_mode
                                    )
        subtile_size = 10560//self.num_subtiles
        mask = mask_reader.read_window(x+subtile_x, y+subtile_y, patch_size = self.patch_size[0])
        
        return mask
    
    def get_mask_params(self):    
        
        mask_params = [] 
        for file in self.image_files:
            print(file)
            subtile_x, subtile_y = get_x_y(file)
            tile = get_tile(file)
            mask_file = os.path.join(self.working_dir,f"data/masks/mask_raster_{tile}.tif")
            subtile_size = 10560//self.num_subtiles
            #mask_reader = MaskReader(os.path.join(self.working_dir,f"data/masks/mask_raster_{tile}.tif"),
            #                            classes_mode = self.classes_mode
            #                            )
            #mask = mask_reader.read_window(0, 0, subtile_x, subtile_y, patch_size = subtile_size)
            mask_params.append({'file' : mask_file,
                                'subtile_x' : subtile_x,
                                'subtile_y' : subtile_y,
                                'subtile_size' : subtile_size,
                                'classes_mode' : self.classes_mode})
        return mask_params

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

            
        f = self.indices[idx]['file']
        subtile_x, subtile_y = get_x_y(f)  
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
        image = preprocess_data(self.opened_files[f].read(window = window), return_nans=self.return_nans, return_torch=True) #np.float32

        mask_params = self.indices[idx]['mask_params']
        mask_file = mask_params['file']
        mask = self.get_mask(f, x = x, y = y)
        

        if self.debug:
            print('LOADING...')
            print(idx)
            print(f)
            print(mask_params['file'])
            print('x, y in subtile:',x, y)

        if self.transform:
            transf_idx = self.indices[idx]['transform']
            image = self.transform[transf_idx](image)
            mask = self.transform[transf_idx](mask)
        #print(self.onehotmasks)
        #print(labels.shape)
        if not self.return_imgidx:
            return image, mask
        else:
            return image, mask, x, y, f

    def __del__(self):
        # Close all the files when the dataset object is destroyed
        for src in self.opened_files.values():
            src.close()

def calculate_class_frequencies(masks, stratify_by = 'type_density'):
    labels = []
    if stratify_by == 'type':
        num_classes = 5
    elif stratify_by == 'density':
        num_classes = 4
    elif stratify_by == 'binary':
        num_classes = 2
    elif stratify_by == 'type_density':
        num_classes = 9
    else:
        num_classes = max([np.max(mask) for mask in masks])+1

    for mask in masks:
        label = np.bincount(mask.flatten(), minlength = num_classes)  # Ensure all classes are included
        labels.append(label)
    return np.array(labels)


def save_yaml(train, val, test, save_to):

    working_dir = os.path.abspath('..')
    save_to = os.path.join(working_dir, 'config', save_to)

    directory = os.path.dirname(save_to)
    if not os.path.exists(directory):
        print(f'creating {directory}')
        os.makedirs(directory)

    num_subtiles = get_num_subtiles(train+test+val)
    tiles = []
    for file in train+val+test:
        tile = get_tile(file)
        if tile not in tiles:
            tiles.append(tile)
    

    yaml_dict = {'tiles': tiles,
                 'num_subtiles': num_subtiles,
                 'train_files': train,
                 'val_files': val,
                 'test_files': test
                 }

    with open(save_to, "w") as yaml_file:
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False)  # Pretty printed
        print('saved', save_to)


def load_yaml(file):
    working_dir = os.path.abspath('..')
    file = os.path.join(working_dir, 'config', file)

    with open(file, "r") as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
        print(loaded_data)
        return loaded_data

def get_num_subtiles(filenames):
    tile_size = 10560
    subtile_starts = []
    for file in filenames:
        subtile_x, subtile_y = get_x_y(file)
        if subtile_x > 0:
            subtile_starts.append(subtile_x)
        if subtile_y > 0:
            subtile_starts.append(subtile_y)
    subtile_starts = list(set(subtile_starts))

    gcd_value = subtile_starts[0]
    for num in subtile_starts[1:]:
        gcd_value = math.gcd(gcd_value, num)

    return tile_size // gcd_value

def train_val_test_stratify(tiles, 
                            num_subtiles,
                            train_size = 0.7, 
                            val_size = 0.15, 
                            save_to = 'subtiles_filenames.yaml',
                            stratify_by = 'type_density'):
    
    # divides into train, validation and test sets, in a stratified way.
    '''
    Stratify_by:
    None: do not stratify
    Type: 

    '''
    ### TODO: vizualizar a distribuicao de classes?
    ### Comentar, colocar no relatorio.

    working_dir = os.path.abspath('..')

    if not isinstance(tiles, list):
        tiles = [tiles]


    all_files = []
    for tile in tiles:
        folder = os.path.join(working_dir,f"data/processed/S2-16D_V2_{tile}/{num_subtiles}x{num_subtiles}_subtiles")
        
        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files if f.endswith('.tif')]
        all_files+=files

        if len(files)!=num_subtiles*num_subtiles:
            raise(f"Still missing {num_subtiles*num_subtiles - len(files)} image compositions. Run src.data.subtile_composition.create_composition() for the tile {tile} and {num_subtiles} subtiles. There is an example at prepare_images.ipynb")
        


    random.seed(42) #for reproducibility
    random.shuffle(all_files)

    train_total = int(train_size*len(all_files))
    val_total = int(val_size*len(all_files))
    test_total = len(all_files) - train_total - val_total

    print(train_total, val_total, test_total)
    
    if stratify_by == '' or stratify_by is None:
        train_set = all_files[:train_total]
        val_set = all_files[train_total:train_total+val_total]
        test_set = all_files[train_total+val_total:]

        if save_to is not None:
            save_yaml(train_set, val_set, test_set, save_to)

        return train_set, val_set, test_set

    # if stratify
    masks = []

    for file in all_files:
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        mask_reader = MaskReader(os.path.join(working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                    classes_mode = stratify_by
                                    )
        subtile_size = 10560//num_subtiles
        mask = mask_reader.read_window(subtile_x, subtile_y, patch_size = subtile_size)
        masks.append(mask)

    label_count = calculate_class_frequencies(masks, stratify_by = stratify_by)

    files_labels = []
    for f, lc in zip(all_files, label_count):
        files_labels.append({'file': f, 'label_count': lc})


    sums = []
    for i in range(label_count[0].shape[0]):
        sum_nth_element = int(sum(item['label_count'][i] for item in files_labels))  #sum of pixels of nth class
        sums.append(sum_nth_element)

    remaining = files_labels
    sorted_indices = sorted(range(len(sums)), key=lambda i: sums[i])
    train_set = []
    val_set = []
    test_set = []

    while len(remaining)>0:
        for i in sorted_indices: # ordered by the sum of pixels the classes
            remaining = sorted(remaining, key=lambda x: x['label_count'][i], reverse=True) #for the class i, sort by the largest pixel count (biggest area of ith class first) 
            
            #random add to train, val or test set.
            def add_to_set(set_, total):
                if len(set_) < total and len(remaining) > 0:
                    file = remaining.pop(0)['file']
                    set_.append(file)
            
            params = [(train_set, train_total), (val_set, val_total), (test_set, test_total)]
            random.shuffle(params)  
            for set_, total in params:
                add_to_set(set_, total)

    if save_to is not None:
        save_yaml(train_set, val_set, test_set, save_to)
    return train_set, val_set, test_set
    
def check_stratification(set_files, num_subtiles, stratify_by):
    working_dir = os.path.abspath('..')
    total_counts = {}
    for file in set_files:      
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        mask_reader = MaskReader(os.path.join(working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                    classes_mode = stratify_by
                                    )
        subtile_size = 10560//num_subtiles
        mask = mask_reader.read_window(subtile_x, subtile_y, patch_size = subtile_size)
        c, count = np.unique(mask, return_counts=True)
        class_count_dict = {int(class_) : int(counter_) for class_, counter_ in zip(c, count)}
        #print(class_count_dict, f)
        for c in class_count_dict:
            if c not in total_counts:
                total_counts[c] = 0     
            total_counts[c] += class_count_dict[c]

    total = sum(total_counts.values())
    for c in total_counts:
        print(c, total_counts[c], total, end='; ')
        total_counts[c] /= total
        
    print(total_counts)
    return total_counts


def extract_integers(string):
    # Use regular expressions to find the patterns 'x=integer' and 'y=integer' (ignoring decimal points and non-digit characters after the numbers)
    x_match = re.search(r'x=(\d+)', string)
    y_match = re.search(r'y=(\d+)', string)
    
    # Extract the numbers as integers, or return None if not found
    x_value = int(x_match.group(1)) if x_match else None
    y_value = int(y_match.group(1)) if y_match else None
    
    return x_value, y_value

def unique_counts(labels):
    class_count_dict, counts = np.unique(labels, return_counts=True)
    class_count_dict = {int(class_) : int(counter_) for class_, counter_ in zip(class_count_dict, counts)}
    return class_count_dict 


class DiagonalFlip1:
    def __call__(self, tensor):
        # Diagonal Flip 1: Transpose the tensor (flip along the top-left to bottom-right diagonal)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Swap the first two dimensions (C, H, W)

class DiagonalFlip2:
    def __call__(self, tensor):
        # Diagonal Flip 2: Flip horizontally and then transpose
        tensor = tensor.flip(2)  # Flip along the width (last dimension)
        return tensor.permute(1, 0, *range(2, tensor.dim()))  # Then transpose