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
from scipy.ndimage import distance_transform_edt, convolve

import src.data.mask_processing as mask_processing


def method_convolution(arr):
    mask = np.isnan(arr)
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.float32)
    kernel /= np.sum(kernel)
    filled_arr = arr.copy()
    filled_arr[mask] = 0
    filled_arr = convolve(filled_arr, kernel, mode='constant', cval=0.0)
    filled_arr[mask] = filled_arr[mask] / convolve(~mask, kernel, mode='constant', cval=0.0)[mask]
    return filled_arr

def fill_nans_nearest(arr: np.ndarray, negate_filled: bool = False):
    """
    Fill NaN values in a 2D array with the nearest non-NaN values.
    If `negate_filled` is True, multiply the filled values by -1.
    """
    # Create a mask of NaN locations
    mask = np.isnan(arr)

    # Find the nearest non-NaN values using distance transform
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled_arr = arr.copy()
    filled_arr[mask] = arr[tuple(indices)][mask]  # Fill NaNs with nearest neighbors

    # Multiply only the filled values by -1 if `negate_filled` is True
    if negate_filled:

        filled_arr[mask] *= -1

    return filled_arr




def preprocess_data(data: np.ndarray, treat_nans: bool | str = False, return_torch: bool = True):
    # data is int16, where negative values represent invalid 
    # treat_nans: can be 
    # if false, returns data with nans.
    # in true, just abs them
    # if a string, use as interpolation method
    
    data = (data.astype(np.float32))/10000.0

    if treat_nans == True: # not a string
        data=np.abs(data)
    else:
        valid_mask = data >= 0
        data[~valid_mask] = np.nan
    
        if treat_nans == 'nearest':
            data = fill_nans_nearest(data, negate_filled=False)
        elif treat_nans == 'linear':
            raise('linear interpolation nan fill not implemented')
    if return_torch:
        return torch.tensor(data, dtype=torch.float32)
    return data




class MaskReader:
    def __init__(self, mask_filename, classes_mode: str = 'type'):
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
    

d4_transforms = {0 : None,
           1 : transforms.RandomRotation(degrees=(90, 90)),
           2 : transforms.RandomRotation(degrees=(180, 180)),
           3 : transforms.RandomRotation(degrees=(270, 270)),
           4 : transforms.RandomHorizontalFlip(p=1.0),
           5 : transforms.Compose([                        # Horizontal flip + 90° rotation
                                    transforms.RandomHorizontalFlip(p=1),
                                    transforms.RandomRotation([90, 90]),
                                ]),
           6 : transforms.Compose([                        # Horizontal flip + 180° rotation
                                    transforms.RandomHorizontalFlip(p=1),
                                    transforms.RandomRotation([180, 180]),
                                ]),
           7 : transforms.Compose([                        # Horizontal flip + 270° rotation
                                    transforms.RandomHorizontalFlip(p=1),
                                    transforms.RandomRotation([270, 270]),
                                ]),
           }



class SubtileDataset(Dataset):
    def __init__(self, files: list, num_subtiles: int, classes_mode: str = 'type', patch_size: int = 256, stride: int = 128, data_augmentation: bool = False, return_imgidx: bool = False, treat_nans: bool|str = False, debug: bool = True):
        self.image_files = files
        self.opened_files = {fp:rasterio.open(fp) for fp in self.image_files}      
        self.patch_size = patch_size
        self.stride = stride
        self.num_subtiles = num_subtiles
        self.working_dir = os.path.abspath('..')
        self.classes_mode = classes_mode
        self.treat_nans = treat_nans
        self.mask_params = self.get_mask_params()
        self.data_augmentation = data_augmentation
        self.return_imgidx = return_imgidx
        self.debug = debug
        self.transforms = d4_transforms #global

        with rasterio.open(files[0]) as im:
            image = im.read()
            self.subtile_size = image.shape


        count_da = 0
        total = 0
        self.indices=[]

        if self.data_augmentation:     
            print('Doing data augmentation...')
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
                    total+=1
                    if self.data_augmentation:                        
                        mask = self.get_mask(f, x, y)
                        if self.check_augmentation(mask, threshold = 0.01):
                            count_da+=1
                            for t_idx in range(1,len(d4_transforms)):
                                idx_dict = {'file':f, 
                                    'x':x, 
                                    'y':y, 
                                    'transform' : t_idx,
                                    'mask_params':mp,
                                    'augmented' : 1
                                    }
                                self.indices.append(idx_dict)
        if self.data_augmentation:     
            print(f'Augmented {count_da} images, of {total}')
            
    def get_mask(self, file: str, x: int = 0, y: int = 0, return_tensor: bool = False):    
        
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        #print('x, y: ', x, y, 'subtiles:', subtile_x, subtile_y)
        mask_reader = MaskReader(os.path.join(self.working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                 classes_mode = self.classes_mode
                                )
        #subtile_size = 10560//self.num_subtiles
        mask = mask_reader.read_window(x+subtile_x, y+subtile_y, patch_size = self.patch_size, return_torch = return_tensor)
        
        return mask
    
    def check_augmentation(self, mask: np.ndarray | torch.Tensor, threshold: float = 0.01):
        if self.classes_mode == 'type':
            ignore_values = (0, 5)
        elif self.classes_mode == 'density':
            ignore_values = (0, 4)
        elif self.classes_mode == 'binary':
            return False
        else:
            ignore_values = (0, 9)

        minority_mask = mask[(mask != ignore_values[0]) & (mask != ignore_values[1])]
        unique_values, counts = np.unique(minority_mask, return_counts=True)
        unique_values = unique_values.tolist()
        counts = counts.tolist()
        if len(unique_values)>=2:
            return False
        if isinstance(mask, np.ndarray):
            proportions = [c/mask.size for c in counts]
        elif isinstance(mask, torch.Tensor):
            proportions = [c/mask.numel() for c in counts]
        if sum(proportions)>threshold:
            #print('data augmentation')
            #print(counts)
            #print(proportions, unique_values)
            #print('SUMMM:', sum(proportions))
            return True
        return False

    def get_mask_params(self):    
        
        mask_params = [] 
        for file in self.image_files:
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
        if x > self.subtile_size[1] - self.patch_size:
            x = self.subtile_size[1] - self.patch_size
        if y > self.subtile_size[2] - self.patch_size:
            y = self.subtile_size[2] - self.patch_size
        
        window = rasterio.windows.Window(x ,y , self.patch_size, self.patch_size)            
        image = preprocess_data(self.opened_files[f].read(window = window), treat_nans = self.treat_nans, return_torch=True) #np.float32

        mask_params = self.indices[idx]['mask_params']
        mask_file = mask_params['file']
        mask = self.get_mask(f, x = x, y = y, return_tensor = True)
        
        

        if self.debug:
            print('LOADING...')
            print(idx)
            print(f)
            print(mask_params['file'])
            print('x, y in subtile:',x, y)

        # applying tranform
        transf_idx = self.indices[idx]['transform']

        if transf_idx > 0:
            image = self.transforms[transf_idx](image)
            mask = self.transforms[transf_idx](mask)
            
        if not self.return_imgidx:
            return image, mask
        else:
            return image, mask, x, y, f

    def __del__(self):
        # Close all the files when the dataset object is destroyed
        for src in self.opened_files.values():
            src.close()

def calculate_class_frequencies(masks, stratify_by: str = 'type_density'):
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

    if not os.path.exists(file):
        return None

    try:
        with open(file, "r") as yaml_file:
            loaded_data = yaml.safe_load(yaml_file)
            print('Loading: ',yaml_filename)

            return loaded_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


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


def yaml_filename(num_subtiles, tiles, stratified_by):
    filename = 'train_val_test_split'
    filename += f'-{num_subtiles}_subtiles-'
    filename += '_'.join(tiles)
    if stratified_by != '' and stratified_by is not None:
        filename += f'-stratified_by_{stratified_by}'
    filename += '.yaml'

    working_dir = os.path.abspath('..')
    filename = os.path.join(working_dir, 'config', filename)

    return filename

def train_val_test_stratify(tiles, 
                            num_subtiles,
                            train_size = 0.7, 
                            val_size = 0.15, 
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
        
    ## Checking if already saved

    save_to = yaml_filename(num_subtiles, tiles, stratify_by)
    loaded_data = load_yaml(save_to)
    if loaded_data is not None:
        print('File already saved, loading it.')
        train_set = loaded_data['train_files']
        val_set = loaded_data['val_files']
        test_set = loaded_data['test_files']
        return train_set, val_set, test_set

    random.seed(42) #for reproducibility
    random.shuffle(all_files)

    train_total = round(train_size*len(all_files))
    val_total = round(val_size*len(all_files))
    test_total = len(all_files) - train_total - val_total
    
    print(f'Training set size:',train_total)
    print(f'Validation set size:',val_total)
    print(f'Test set size:',test_total)

    
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
        if stratify_by == 'type':
            mask = mask_processing.get_type(mask)
        elif stratify_by == 'binary':
            mask = mask_processing.get_binary(mask)
        elif stratify_by == 'density':
            mask = mask_processing.get_density(mask)
            
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