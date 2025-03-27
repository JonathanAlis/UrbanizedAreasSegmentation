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
from tqdm import tqdm
import pickle
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





def preprocess_data(data: np.ndarray, treat_nans: bool | str = False, ch_selection = None, return_torch: bool = True):
    # data is int16, where negative values represent invalid 
    # treat_nans: can be 
    # if false, returns data with nans.
    # in true, just abs them
    # if a string, use as interpolation method
    
    if ch_selection is not None:
        if data.shape[0] > len(ch_selection):
            data = data[ch_selection]
        #else:
        #    raise(ValueError("Channels mismatch"))
        
    dtype = data.dtype
    if dtype == np.int16:
        data = (data.astype(np.float32))/10000.0
    if dtype == np.uint8:
        data = (data.astype(np.float32))/255.0
    
    if treat_nans == True or treat_nans == 'absolute': # not a string
        data=np.abs(data)
    elif treat_nans == 'negative' or treat_nans == False:
        pass
    else: #any other string
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
        4type: 4 types, excludes vazios intraurbanos
        densities: all 4 densities
        binary: 0 or 1 
        '''
        self.mask_filename = mask_filename
        self.mask_data = rasterio.open(self.mask_filename)
        self.classes_mode = classes_mode

    def transform_to_class_mode(self, m):
        if self.classes_mode.lower() == 'type':
            m = mask_processing.get_type(m)
        elif self.classes_mode.lower() == '4types':
            m = mask_processing.get_4types(m)
        elif self.classes_mode.lower() == 'density':
            m = mask_processing.get_density(m)
        elif self.classes_mode.lower() == 'binary':
            m = mask_processing.get_binary(m)
        elif self.classes_mode.lower() == 'equips':
            m = mask_processing.get_equips(m)
        elif self.classes_mode.lower() == 'lote':
            m = mask_processing.get_loteamento(m)
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
    def __init__(self, source: list|str, patch_size: int = 256, stride: int = 128, channels_subset = None,
                 dynamic_sampling: bool = False, data_augmentation: bool = False, 
                 num_subtiles: int = 0, classes_mode: str = 'type', return_imgidx: bool = False, 
                 ignore_most_nans: bool = True, treat_nans: bool|str = False, debug: bool = False, set:str = '', working_dir = None):
        
        ### ----------Definições
        if working_dir is None:
            self.working_dir = os.path.abspath('..')
        else:
            self.working_dir = working_dir
        self.debug = debug
        
        self.transforms = d4_transforms #global
        self.min_step = max(patch_size//8, 16)
        self.daug_threhshold = 0.01
        if channels_subset:
            self.channels_subset = channels_subset
        else:
            self.channels_subset = list(range(12))
        
        ### ----------Parametros obrigatórios
        self.patch_size = patch_size
        self.stride = stride
        self.dynamic_sampling = dynamic_sampling
        self.data_augmentation = data_augmentation

        ### ----------Parametros semi obrigatórios:
        # obrigatório dependendo dos outros parâmetros
        # primeiro defino aqui, se necessário, será substituido
           
        self.classes_mode = classes_mode

        ### --------- Fonte dos arquivos, 
        # pode ser um yaml, ou lista de string, ou lista de dicts contendo os patches
        # se lista, outros parametros sao obrigatórios

        # Lista de strings com os nomes dos arquivos de imagens a carregar
        if isinstance(source, list) and isinstance(source[0], str):
            self.image_files = source  
            self.num_subtiles = num_subtiles
            self.classes_mode = classes_mode
            tiles = []
            for f in source:
                tile_id = f.split('S2-16D_V2_')[1].split('/')
                if tile_id not in tiles:
                    tiles.append(tile_id)
            if self.debug:
                print(tiles)
            self.tiles = tiles
        loaded_patches = False

        # Se é uma string, é de nome de arquivo yaml
        if isinstance(source, str) and source.endswith('.yaml'):
            self.source = source
            self.set = set
            
            loaded_data = load_yaml(os.path.join(self.working_dir, 'config', self.source))
            print(f'Loading from yaml: {self.source}')
            print(loaded_data)
            #Loading from yaml: train_val_test_split/6-subtiles/mode-4types/num_tiles-12/train_val_test_split.yaml

            self.num_subtiles = loaded_data['num_subtiles']
            self.image_files = loaded_data[set]
            self.classes_mode = self.source.split('/')[2].split('mode-')[-1]
            
            if self.set == 'train_files':
                file_list = loaded_data['train_files']
            if self.set == 'val_files':
                file_list = loaded_data['val_files']
            if self.set == 'test_files':
                file_list = loaded_data['test_files']
            tiles = []
            for f in file_list:
                tile_id = f.split('S2-16D_V2_')[1].split('/')[0]
                if tile_id not in tiles:
                    tiles.append(tile_id)
            self.tiles = tiles
            filename = self.get_filename()
            #/train_val_test_split/6-subtiles/mode-4types/patchsize-256/stride-256/DS-True/DA-False/set-train_files/channels-12/sampling.pkl
            
            if os.path.exists(filename):
                print(f"Loading preloaded dataset from {filename}")
                with open(filename, 'rb') as f:
                    self.num_subtiles, self.classes_mode, self.patch_size, self.stride, self.dynamic_sampling, self.data_augmentation, self.image_files, self.patches, self.channels_subset= pickle.load(f)
                    loaded_patches = True
            else:
                #print('dataset_6-subtiles_mode-type_patchsize-256_stride-256_dynamicsampling-True_dataaugmentation-True_set-train_files_source-train_val_test_split-6_subtiles-032027-stratified_by_type')
                print(f"Preloaded dataset file not found, initializing normally...")
                print(filename)

        # Se a fonte for os patches já calculados     
        patches_from_dict = False
        #print(source)
        if isinstance(source, list) and isinstance(source[0], dict):
            self.patches = source
            patches_from_dict = True
            
            self.opened_files = {dic['file']:rasterio.open(dic['file']) for dic in source}    
            self.image_files = [dic['file'] for dic in source]  
            loaded_patches = True
            
                #{'file': '/home/jonathan/UrbanizedAreasSegmentation/data/processed/S2-16D_V2_033029/6x6_subtiles/q_12ch/x=8800_y=5280.tif'
                
        else:
            self.opened_files = {fp:rasterio.open(fp) for fp in self.image_files}      
        
        #print(self.opened_files)
        ### outras definições
        self.treat_nans = treat_nans  
        self.ignore_most_nans = ignore_most_nans      
        self.return_imgidx = return_imgidx
        self.debug = debug        
        
        
        if self.classes_mode == 'type':
            self.num_classes = 5
        elif self.classes_mode == '4types':
            self.num_classes = 4
        elif self.classes_mode == 'density':
            self.num_classes = 4
        elif self.classes_mode == 'all':
            self.num_classes = 9
        else:
            self.num_classes = 2
        
        with rasterio.open(self.image_files[0]) as im:
            image = im.read()
            self.subtile_size = image.shape

        ### ---------------- Cálculo dos patches
        if not loaded_patches and not patches_from_dict:
            
            total_not_augmented = 0
            self.patches=[]
            for f in self.image_files:
                for x in range(0, self.subtile_size[1], self.stride):
                    for y in range(0, self.subtile_size[2], self.stride):
                        add_to_dataset = True
                        if self.ignore_most_nans == True:
                            if self.has_more_than_percent_negative(f, x, y, threshold = 0.01):
                                add_to_dataset = False
                        if add_to_dataset:
                            idx_dict = {'file':f, 
                                'x':x, 
                                'y':y, 
                                'transform' : 0,
                                'step_shift' : 0
                                }               
                            self.patches.append(idx_dict)
                            total_not_augmented+=1
            else:
                total_not_augmented = len(self.patches)

        ### ---------- Amostragem dinâmica
        if self.dynamic_sampling:  
            aug1 = self.add_dynamic_sampling()

        ### ---------- Data augmentation
        if self.data_augmentation:
            aug2 = self.d4_data_augmentation()

            counter_pixel, counter_img = self.count_classes()  
        if self.dynamic_sampling and self.data_augmentation:
            print(f'After data augmentation:')
            print(f'Pixels for each class:', counter_pixel)
            print(f'Num images with each class:', counter_img)

            #print(f'Starting from {total_not_augmented} images')
            print(f'Dinamic Window step added {aug1} images')
            print(f'Data augmentation added {aug2} images with transform')
            print(f'Total: {len(self)}')

        if isinstance(source, str):
            self.save_to_file()

    def save_to_file(self):
        """Save the instance to a file."""
        filename = self.get_filename()
        with open(filename, 'wb') as f:
            pickle.dump((self.num_subtiles, self.classes_mode, self.patch_size, self.stride, self.dynamic_sampling, self.data_augmentation, self.image_files, self.patches, self.channels_subset), f)
        print(f"Dataset instance saved to {filename}")

    def get_filename(self):
        #train_val_test_split/6-subtiles/mode-4types/patchsize-256/stride-256/DS-True/DA-False/set-train_files/channels-12_sampling.pkl
        """Generate a unique filename based on instance parameters."""
        subfolder = f'train_val_test_split/'
        subfolder += f'{self.num_subtiles}-subtiles/'
        subfolder += f'mode-{self.classes_mode}/'
        subfolder += f'num_tiles-{len(self.tiles)}/'
        subfolder += f'patchsize-{self.patch_size}/'
        subfolder += f'stride-{self.stride}/'
        subfolder += f'DS-{self.dynamic_sampling}/'
        subfolder += f'DA-{self.data_augmentation}/'
        subfolder += f'channels-{len(self.channels_subset)}/'
        os.makedirs(os.path.join(self.working_dir, 'config', subfolder), exist_ok=True)        
        filename = f'{self.set}_sampling.pkl'        
        #Loading instance from /home/jonathan/UrbanizedAreasSegmentation/config/dataset_6-subtiles_mode_type_patchsize-256_stride-256_dynamicsampling-True_dataaugmentation-True_set-<class 'set'>_source-train_val_test_split-6_subtiles-032027-stratified_by_type.pkl

        return os.path.join(self.working_dir, 'config', subfolder, filename)

    def add_dynamic_sampling(self):
        num_included_images = 0
        print('Doing dynamic sampling (DS)')
        counter_pixel, counter_img = self.count_classes()
        print(f'Before DS:')
        print(f'Num of pixels for each class:', counter_pixel)
        print(f'% of pixels for each class:', [100*c/sum(counter_pixel) for c in counter_pixel])
        print(f'Num of images with each class:', counter_img)
        #Num of images with each class: tensor([12665,   194,   428,  1819])
        print(f'% of images with each class:', [100*c/(len(self)) for c in counter_img])
        #minority_classes = sorted(range(len(counter_img)), key=lambda i: counter_img[i], reverse=True)[1:]
        minority_classes = [i for i in sorted(range(len(counter_img)), key=lambda i: counter_img[i], reverse=True)[1:]
                            if counter_img[i] > 0
                            ]
        
        print(minority_classes)
        steps = [self.stride // 2**math.floor(math.log2(math.sqrt(counter_img[0]//counter_img[mc]))) for mc in minority_classes] 
        print(steps)
        #print([math.sqrt(counter_img[0]//counter_img[mc]) for mc in minority_classes])
        aug_indices = []
        for i_d in tqdm(self.patches):
            x, y, f = i_d['x'], i_d['y'], i_d['file']
            #print(x, y, f)
            center_mask = self.get_mask(f, x, y)
            for s, mc in enumerate(minority_classes):
                if counter_img[mc] > 0 and mc in center_mask:
                    ### STEP DINÂMICO
                    
                    step = self.stride // 2**math.floor(math.log2(math.sqrt(counter_img[0]//counter_img[mc])))#math.ceil(2**(s+1) counter_img[0]//counter_img[mc]
                    #print(step, counter_img[0]/counter_img[mc])
                    if step < self.min_step:
                        step = self.min_step
                    #print(step, 2**math.floor(math.log2(math.sqrt(counter_img[0]//counter_img[mc]))))
                    for x_shift in range(-self.patch_size+step, self.patch_size, step):
                        for y_shift in range(-self.patch_size+step, self.patch_size, step):  
                            #print(x,y, list(range(-self.patch_size+step, self.patch_size, step)), list(range(-self.patch_size+step, self.patch_size, step)))                          
                            if x_shift!=0 and y_shift!=0:
                                if 0 <= x+x_shift < self.subtile_size[1] - self.patch_size:
                                    if 0 <= y+y_shift < self.subtile_size[2] - self.patch_size:
                                        #print('.', end='')
                                        mask = self.get_mask(f, x+x_shift, y+y_shift)
                                        if mc in mask:
                                            #print(np.unique_counts(mask))

                                            idx_dict = {'file':f, 
                                                        'x':x+x_shift, 
                                                        'y':y+y_shift, 
                                                        'transform' : 0,
                                                        'step_shift' : step,
                                                        }
                                            if idx_dict not in self.patches:
                                                aug_indices.append(idx_dict)
                                                num_included_images+=1
        if 0:
            print(aug_indices)
        self.patches.extend(aug_indices)
        
        counter_pixel, counter_img = self.count_classes()
        print(f'After DS:')
        print(f'Num of pixels for each class:', counter_pixel)
        print(f'% of pixels for each class:', [100*c/sum(counter_pixel) for c in counter_pixel])
        print(f'Num of images with each class:', counter_img)
        #Num of images with each class: tensor([12665,   194,   428,  1819])
        print(f'% of images with each class:', [100*c/(len(self)) for c in counter_img])
        return num_included_images

    def plot_sampled_outlines(self, area_limits=None, save_to = None):
        if not self.dynamic_sampling:
            return
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors

        len(self.image_files)
        rows = min(2, math.floor(math.sqrt(len(self.image_files))))
        cols = min(2, math.floor(math.sqrt(len(self.image_files))))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))  # Adjust figure size
        axes = axes.flatten()  # Flatten for easy iteration


        class_colors = {
            0: "#000000",  # Black (Background)
            1: "#0000FF",  # Cyan
            2: "#FF00FF",  # Magenta
            3: "#FFFF00",  # Yellow
            4: "#FFFFFF"   # White
        }

        cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
        bounds = sorted(class_colors.keys()) + [max(class_colors.keys()) + 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        

        for ax, chosen_file in zip(axes,self.image_files):
            # Determine limits automatically if not provided
            if area_limits is None:
                x_min = min(square["x"] for square in self.patches)
                x_max = max(square["x"] + self.patch_size for square in self.patches)
                y_min = min(square["y"] for square in self.patches)
                y_max = max(square["y"] + self.patch_size for square in self.patches)
            else:
                x_min, x_max, y_min, y_max = area_limits
            
            mask = np.zeros((x_max-x_min, y_max-y_min), dtype=np.uint8)
            
            squares = []
            for idx in range(len(self)):
                x, y, f = self.idx_to_xy(idx, return_filename=True)
                step = self.patches[idx]['step_shift']
                if x_min <= x <= x_max and y_min <= y <= y_max and f == chosen_file:
                    mask_ = self.get_mask(f, x = x, y = y, return_tensor = True)  
                    mask[y:y+self.patch_size, x:x+self.patch_size] = mask_
                    squares.append({'x':x, 'y':y, 'step':step})

            im = ax.imshow(mask, cmap=cmap, extent=[0, mask.shape[1], mask.shape[0], 0], alpha=1)


            # Draw grid
            ax.set_xticks(range(x_min, x_max + 1, self.patch_size))
            ax.set_yticks(range(y_min, y_max + 1, self.patch_size))
            #ax.grid(True, linestyle="--", linewidth=0.5)

            # Plot squares
            for square in squares:
                if x_min <= square["x"] <= x_max and y_min <= square["y"] <= y_max:
                    if square['step'] == 0: 
                        rect = patches.Rectangle(
                            (square["x"], square["y"]),  
                            self.patch_size, self.patch_size,  
                            linewidth=1, edgecolor="green", facecolor="none"
                        )
                    else:
                        rect = patches.Rectangle(
                            (square["x"], square["y"]),  
                            self.patch_size, self.patch_size,  
                            linewidth=1, edgecolor="red", facecolor="none"
                        )
                    ax.add_patch(rect)

            # Set limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            #ax.set_aspect('equal')
            #cbar = plt.colorbar(im, ax=ax, ticks=unique_values)
            
        class_labels = {
                0: "Fundo",
                1: "Loteamento vazio",
                2: "Outros equipamentos",
                3: "Vazio intraurbano",
                4: "Área Urbanizada"
            }
        # Create a discrete colorbar (legend)
        #cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  
        #cbar = plt.colorbar(im, cax=cbar_ax, ticks=list(class_colors.keys()))
        #if class_labels:
        #    cbar.set_ticklabels([class_labels[val] for val in class_colors.keys()])
        #cbar.ax.tick_params(size=0)  # Remove unnecessary tick marks
        legend_patches = [
        patches.Patch(facecolor=class_colors[val], edgecolor="black", linewidth=1, label=class_labels[val])
            for val in class_colors.keys()
        ]
        legend_patches = [patches.Patch(color=class_colors[val], label=class_labels[val]) for val in class_colors.keys()]
        fig.legend(handles=legend_patches, loc="upper right", title="Classes")
        # Assign class labels if provided
        
        plt.tight_layout()#rect=[0, 0, 0.85, 1])  # Adjust layout to fit colorbar
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.show()



    def plot_transformed(self, save_to=None):
        if not self.data_augmentation:
            return
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors
        max_classes_in_patch = 0
        
        for idx in range(len(self)):
            transform = self.patches[idx]['transform']
            x, y, f = self.patches[idx]['x'], self.patches[idx]['y'],self.patches[idx]['file']
            vals, counts = np.unique_counts(self.get_mask(f, x, y))
            if len(vals)>max_classes_in_patch and transform > 0:
                max_classes_in_patch = len(vals)
                chosen_patch = self.patches[idx]

        t_imgs = [torch.zeros((self.patch_size, self.patch_size), dtype = torch.uint8) for _ in range(8)]
        count = 0
        for idx in range(len(self)):
            x, y, f, transform, step = self.patches[idx]['x'], self.patches[idx]['y'], self.patches[idx]['file'], self.patches[idx]['transform'], self.patches[idx]['step_shift']
            if x == chosen_patch['x'] and y == chosen_patch['y'] and f == chosen_patch['file'] and step == chosen_patch['step_shift']:
                count +=1
                t_imgs[self.patches[idx]['transform']] = self.get_mask(f, x, y, return_tensor=True)#.cpu().detach().numpy()
            if count == 8:
                break
        plt.figure(figsize=(20,20))
        rows = 2
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size
        axes = axes.flatten()  # Flatten for easy iteration

        class_colors = {
            0: "#000000",  # Black (Background)
            1: "#0000FF",  # Blue
            2: "#FF00FF",  # Magenta
            3: "#FFFF00",  # Yellow
            4: "#FFFFFF"   # White
        }

        cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
        bounds = sorted(class_colors.keys()) + [max(class_colors.keys()) + 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        for i, ax, mask in zip(range(8), axes,t_imgs):
            if i > 0:
                mask = self.transforms[i](mask)
            mask = mask.squeeze().numpy()
            im = ax.imshow(mask, cmap=cmap, extent=[0, mask.shape[1], mask.shape[0], 0], alpha=1)
            ax.axis('off')

        fig.tight_layout()
        if save_to:
            plt.savefig(save_to, dpi=300, bbox_inches="tight")
        plt.show()
         
    def stats(self):
        num_patches = len(self.patches)
        num_subtiles = self.num_subtiles
        total_pixels = 0
        stats = {label:{'num_patches':0, 'num_pixels':0, 'pct_patches':0, 'pct_pixels':0, 'num_dynamic':0, 'num_augmented':0} for label in range(self.num_classes)}
        for idx in range(len(self)):
            x, y, f = self.idx_to_xy(idx, return_filename=True)
            x, y, transform, step = self.patches[idx]['x'], self.patches[idx]['y'], self.patches[idx]['transform'], self.patches[idx]['step_shift']
            mask = self.get_mask(f, x = x, y = y, return_tensor = True)  
            vals, counts = torch.unique(mask, return_counts = True)
            vals = vals.tolist()
            counts = counts.tolist()
            total_pixels += mask.numel()
            #print(mask.numel())
            #print(counts)
            assert sum(counts)==mask.numel()
            for label, count in zip(vals, counts):
                stats[label]['num_patches']+=1
                stats[label]['num_pixels']+=count
                if step > 0:
                    stats[label]['num_dynamic']+=1
                if transform>0:
                    stats[label]['num_augmented']+=1
        for label in range(self.num_classes):
            stats[label]['pct_patches'] = stats[label]['num_patches']/num_patches
            stats[label]['pct_pixels'] = stats[label]['num_pixels']/total_pixels
        

        print('Total subtiles:', num_subtiles)
        print('Total patches:', num_patches)
        print('Total pixels:', num_patches)
        print('Per class:')
        for label in range(self.num_classes):
            print(f'Class {label}:')
            print(stats[label])
            
                
    def has_more_than_percent_negative(self, f, x, y, threshold):
        data = self.get_mask(f, x, y)
        if isinstance(data, np.ndarray):
            num_positive = np.sum(data < 0)
        elif isinstance(data, torch.Tensor):
            num_positive = torch.sum(data < 0).item()
        else:
            raise TypeError("Input data must be either a NumPy array or PyTorch tensor.")

        total_elements = data.size

        return (num_positive / total_elements) > threshold


    def d4_data_augmentation(self):
        augmented_stage_2 = 0
        counter_pixel, counter_img = self.count_classes()
        print(f'Before data augmentation stage 2:')
        print(f'Pixels for each class:', counter_pixel)
        print(f'Num images with each class:', counter_img)
        count_da = 0
        images_da = 0
        minority_classes = sorted(range(len(counter_img)), key=lambda i: counter_img[i], reverse=True)[1:]
        print(minority_classes)
        print([counter_img[0]//counter_img[mc] if counter_img[mc]>0 else 0 for mc in minority_classes])

        img_percentual = [c/len(self) for c in counter_img]
        rate = [1/c for c in img_percentual]

        aug_indices = []
        for i_d in tqdm(self.patches):
            x, y, f, step = i_d['x'], i_d['x'], i_d['file'], i_d['step_shift']
            
            mask = self.get_mask(f, x, y)
            augment_rate = self.check_augmentation_2(mask, counter_img)# thresholds = self.augmented_thresholds)
            #print(augment_rate, ',x:', x, ',y:', y, end = '|')#f'|{i_d["test_id"]}| ')
            aug_count = 0
            if 1:#for aug_rate in range(math.ceil(augment_rate/8)): #exemplo: augment_rate = 20, faz o loop 3 vezes
                for t_idx in range(1,len(d4_transforms)): # aqui sao 8. 20-> 3 loops externos, quando passa o 20 da break.
                    aug_count+=1
                    idx_dict = {'file':f, 
                        'x':x, 
                        'y':y, 
                        'transform' : t_idx,
                        'step_shift' : step,
                        }
                    aug_indices.append(idx_dict)
                    augmented_stage_2+=1
                    if aug_count >= augment_rate:
                        break   
            count_da+=aug_count
            images_da+=1
        self.patches.extend(aug_indices)

        return count_da



        
    def get_mask(self, file: str, x: int = 0, y: int = 0, return_tensor: bool = False):    
        
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        #print('x, y: ', x, y, 'subtiles:', subtile_x, subtile_y)

        mask_reader = MaskReader(os.path.join(self.working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                 classes_mode = self.classes_mode
                                )
        #subtile_size = 10560//self.num_subtiles
        mask = mask_reader.read_window(x+subtile_x, y+subtile_y, patch_size = self.patch_size, return_torch = return_tensor)
        
        return mask #shape (5, w, h)
    
    def check_augmentation(self, mask: np.ndarray | torch.Tensor, thresholds: tuple = (0.01, 0.25)):
        if self.classes_mode == 'type':
            ignore_values = (0, 4)
        elif self.classes_mode == 'density':
            ignore_values = (0, 3)
        elif self.classes_mode == 'binary':
            return False, -1
        else:
            ignore_values = (0, 8)


        minority_mask = mask[(mask != ignore_values[0]) & (mask != ignore_values[1])]
        #minority_mask = mask[(mask != ignore_values[0])]

        unique_values, counts = np.unique(minority_mask, return_counts=True)
        unique_values = unique_values.tolist()
        counts = counts.tolist()
        if len(unique_values)==1:
            return False, -1
        if isinstance(mask, np.ndarray):
            proportions = [c/mask.size for c in counts]
        elif isinstance(mask, torch.Tensor):
            proportions = [c/mask.numel() for c in counts]
        if thresholds[0]<sum(proportions)<thresholds[1]:
            #print('data augmentation')
            #print(counts)
            #print(proportions, unique_values)
            #print('SUMMM:', sum(proportions))
            return True, sum(proportions)
        return False, sum(proportions)
    
    def check_augmentation_2(self, mask: np.ndarray | torch.Tensor, class_count: list):
        #check it there is a class, and if it is greater than threshold
        
        rate = [c for c in class_count]
        img_percentual = [c/sum(class_count) for c in class_count]
        rate = [1/c for c in img_percentual]
        augment_by = 1
        if isinstance(mask, np.ndarray):
            unique_classes, counts = np.unique(mask, return_counts=True) 
        elif isinstance(mask, torch.Tensor):
            unique_classes, counts = mask.unique(return_counts=True) 
        #print('BBB', unique_classes, counts)  
        for class_, count in zip(unique_classes, counts):
            #print('CCC', count, threshold * self.patch_size**2)
            if class_ != 0:
                if count > self.daug_threhshold * self.patch_size**2:
                    if augment_by < rate[class_]:
                        augment_by = rate[class_]
                        #print('new augment:', augment_by, 'class', class_)
        return augment_by - 1
    
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

    def count_classes(self):
        class_counts = torch.zeros(self.num_classes, dtype=torch.int64)
        class_counts_img = torch.zeros(self.num_classes, dtype=torch.int64)
        
        for idx in range(self.__len__()):
            x, y, f = self.idx_to_xy(idx, return_filename=True)
            mask = self.get_mask(f, x = x, y = y, return_tensor = True)
            unique_classes, counts = mask.unique(return_counts=True)   
            for class_idx, count in zip(unique_classes, counts): 
                class_counts[class_idx] += count.item()
                if count.item() > 0:
                    class_counts_img[class_idx]+=1
        return class_counts,class_counts_img
    

    def idx_to_xy(self, idx, return_filename = False):
        f = self.patches[idx]['file']
        subtile_x, subtile_y = get_x_y(f)  
        x, y = self.patches[idx]['x'], self.patches[idx]['y']
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.subtile_size[1] - self.patch_size:
            x = self.subtile_size[1] - self.patch_size
        if y > self.subtile_size[2] - self.patch_size:
            y = self.subtile_size[2] - self.patch_size
        if return_filename:
            return x, y, f
        return x, y



    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y, f = self.idx_to_xy(idx, return_filename=True)        
        window = rasterio.windows.Window(x ,y , self.patch_size, self.patch_size)            
        image = preprocess_data(self.opened_files[f].read(window = window), 
                                treat_nans = self.treat_nans,
                                ch_selection = self.channels_subset, 
                                return_torch=True) #np.float32

        #mask_params = self.patches[idx]['mask_params']
        #mask_file = mask_params['file']
        mask = self.get_mask(f, x = x, y = y, return_tensor = True)        
        
        if self.debug:
            print('LOADING...')
            print(idx)
            print(f)
            print('x, y in subtile:',x, y)

        # applying tranform
        transf_idx = self.patches[idx]['transform']
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

#### Fim da classe SubtileDataset
#### -----------------------------------------------------


def calculate_class_frequencies(masks, stratify_by: str = 'type_density'):
    labels = []
    if stratify_by == 'type':
        num_classes = 5
    elif stratify_by == 'density':
        num_classes = 4
    elif stratify_by == '4types':
        num_classes = 4
    elif stratify_by == 'binary':
        num_classes = 2
    elif stratify_by == 'type_density' or stratify_by == 'all':
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
            print('Loading: ',file)

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


def yaml_filename(num_subtiles, tiles, classes_mode):
    subfolder = f'train_val_test_split/'
    subfolder += f'{num_subtiles}-subtiles/'
    subfolder += f'mode-{classes_mode}/'
    subfolder += f'num_tiles-{len(tiles)}/'

    filename = 'train_val_test_split.yaml'

    return subfolder+filename

def train_val_test_stratify(tiles, 
                            num_subtiles,
                            train_size = 0.7, 
                            val_size = 0.15, 
                            stratify_by = '',
                            subfolder = None,
                            debug = False,
                            working_dir = None):
    
    # divides into train, validation and test sets, in a stratified way.
    '''
    Stratify_by:
    None: do not stratify
    Type: 

    '''
    ### TODO: vizualizar a distribuicao de classes?
    ### Comentar, colocar no relatorio.

    if not working_dir:
        working_dir = os.path.abspath('..')

    if isinstance(tiles, str):
        tiles = [tiles]


    all_files = []
    for tile in tiles:
        print(tile)
        folder = os.path.join(working_dir,f"data/processed/S2-16D_V2_{tile}/{num_subtiles}x{num_subtiles}_subtiles")
        if subfolder:
            folder = os.path.join(folder, subfolder)

        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files if f.endswith('.tif')]
        all_files+=files

        if len(files)!=num_subtiles*num_subtiles:
            raise ValueError(
                            f"Still missing {num_subtiles * num_subtiles - len(files)} image compositions. "
                            f"Run src.data.subtile_composition.create_composition() for the tile {tile} and {num_subtiles} subtiles. "
                            f"There is an example at prepare_images.ipynb"
                        )       
    ## Checking if already saved

    save_to = yaml_filename(num_subtiles, tiles, stratify_by)
    loaded_data = load_yaml(save_to)
    if loaded_data is not None:
        print('File already saved, loading it.')
        train_set = loaded_data['train_files']
        val_set = loaded_data['val_files']
        test_set = loaded_data['test_files']
        return train_set, val_set, test_set

    random.seed(0) #for reproducibility
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
        elif stratify_by == '4types':
            mask = mask_processing.get_4types(mask)
        elif stratify_by == 'binary':
            mask = mask_processing.get_binary(mask)
        elif stratify_by == 'density':
            mask = mask_processing.get_density(mask)        
        elif stratify_by == 'equips':
            mask = mask_processing.get_equips(mask)
        elif stratify_by == 'lote':
            mask = mask_processing.get_loteamento(mask)
        #else: nao muda, tem os valores de 0 a 9 ja
        #print(mask.unique())
            
        masks.append(mask)


    label_count = calculate_class_frequencies(masks, stratify_by = stratify_by)
    if debug:
        print(label_count)
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
    
def count_classes(set_files, num_subtiles, agregate_by = '', output_latex = False):
    working_dir = os.path.abspath('..')
    pixel_counts = {}
    class_counts = {}
    
    for file in set_files:      
        subtile_x, subtile_y = get_x_y(file)
        tile = get_tile(file)
        mask_reader = MaskReader(os.path.join(working_dir,f"data/masks/mask_raster_{tile}.tif"),
                                    classes_mode = agregate_by
                                    )
        subtile_size = 10560//num_subtiles
        mask = mask_reader.read_window(subtile_x, subtile_y, patch_size = subtile_size)
        c, count = np.unique(mask, return_counts=True)
        class_count_dict = {int(class_) : int(counter_) for class_, counter_ in zip(c, count)}
        #print(class_count_dict, f)
        for c in class_count_dict:
            if c not in pixel_counts:
                pixel_counts[c] = 0
                class_counts[c] = 0     
            pixel_counts[c] += class_count_dict[c]
            class_counts[c] += 1

    #print(total_counts)
    #print(class_counts)
    percent_counts = pixel_counts.copy()
    total = sum(percent_counts.values())
    for c in percent_counts:
        #print(c, total_counts[c], total, end='; ')
        percent_counts[c] /= total
    percent_counts = {k: f'{100*percent_counts[k]:.2f}' for k in sorted(percent_counts)}
    pixel_counts = {k: pixel_counts[k] for k in sorted(pixel_counts)}
    #print(' & '.join([str(pc) for pc in pixel_counts.values()]))
    if output_latex:
        print(' & '.join([pc+'\%' for pc in percent_counts.values()]))
    else:
        print(', '.join([f'{classe}: {pc}%' for classe, pc in zip(['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'],percent_counts.values())]))
        print('----------------------')

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
    


import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm 

def compute_pca_from_dataloader(dataloader, num_channels=12, num_components=3, batch_size=256, device="cuda", save_path=None):
    """
    Computes PCA iteratively from a PyTorch DataLoader and saves the PCA components.

    Args:
        dataloader: PyTorch DataLoader yielding (image, label).
        num_channels: Number of input channels.
        num_components: Number of principal components.
        batch_size: Batch size for IncrementalPCA.
        device: Device ('cuda' or 'cpu').
        save_path: Path to save/load the PCA components (.npy file).

    Returns:
        torch.Tensor: PCA components of shape [num_components, num_channels].
    """
    if save_path and os.path.exists(save_path):
        # Load and return if already computed
        print(f"Loading PCA components from {save_path}")
        return torch.tensor(np.load(save_path), dtype=torch.float32)

    ipca = IncrementalPCA(n_components=num_components, batch_size=batch_size)

    for images, _ in tqdm(dataloader, desc="Computing PCA", unit="batch"):
        images = images.to(device)
        B, C, W, H = images.shape
        reshaped = images.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)  # [B*W*H, C]
        ipca.partial_fit(reshaped.cpu().numpy())  # Convert to NumPy and fit

    pca_components = ipca.components_  # Shape: [num_components, num_channels]

    if save_path:
        np.save(save_path, pca_components)  # Save to file

    return torch.tensor(pca_components, dtype=torch.float32)




def apply_pca_weights(images: torch.Tensor|np.ndarray, pca_weights: str | torch.Tensor):
    """
    Applies PCA transformation to reduce channels from 12 to 3.
    
    Supports input images with or without a batch dimension:
      - [B, 12, W, H] (batched) or 
      - [12, W, H] (non-batched)
    
    Args:
        images (torch.Tensor): Input tensor of shape [B, 12, W, H] or [12, W, H].
        pca_weights (str or torch.Tensor): Path to saved weights or a tensor.
    
    Returns:
        torch.Tensor: Transformed tensor with shape:
                      [B, 3, W, H] if input was batched, or [3, W, H] if non-batched.
    """
    # If pca_weights is a file path, load the weights
    if isinstance(pca_weights, str):
        pca_weights = torch.tensor(np.load(pca_weights), dtype=torch.float32)
    
    # Ensure pca_weights is a torch.Tensor.
    if not isinstance(pca_weights, torch.Tensor):
        raise TypeError("pca_weights must be a string (path) or a torch.Tensor")
    
    # If the images are provided as a numpy array, you might want to convert them to torch.Tensor:
    if isinstance(images, np.ndarray):
        images = torch.tensor(images)
    
    # Determine if the input is batched or not.
    batched = (images.ndim == 4)  # expects [B, 12, W, H]
    if not batched:
        # Expecting shape [12, W, H]. Add a batch dimension.-> [1, 12, W, H]
        images = images.unsqueeze(0)
    
    # Check that the channel dimension of the images (index 1) matches pca_weights' channel dimension.
    # Assume pca_weights shape is [3, 12] (i.e. mapping from 12 channels to 3 channels).
    assert images.shape[1] == pca_weights.shape[0], \
        f"Input channels ({images.shape[1]}) must match PCA weight size ({pca_weights.shape[0]})"
    
    # Apply PCA transformation using einsum.
    # This computes a weighted sum over the 12 channels for each batch element.
    # "bchw,jc->bjhw" means:
    #   b: batch, c: channels, h: height, w: width, j: output channel index, c: input channel
    transformed = torch.einsum("bchw,jc->bjhw", images.float(), pca_weights.T)
    
    # If the input was not batched, remove the batch dimension.
    if not batched:
        transformed = transformed.squeeze(0)
    
    return transformed
