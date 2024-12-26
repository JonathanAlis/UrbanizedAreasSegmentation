
import os
import sys
sys.path.append(os.path.abspath('..'))
from rasterio.coords import BoundingBox

#import src.data.preprocess_data as preprocess_data

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def print_top_max_min(arr, n=100):
    
    unique_values = np.unique(arr)
    sorted_unique_values = np.sort(unique_values)

    top_n_max = sorted_unique_values[-n:]
    top_n_min = sorted_unique_values[:n]

    print(f"Top {n} maximum values:", top_n_max[::-1])
    print(f"Top {n} minimum values:", top_n_min)
    print("NaNs:", np.count_nonzero(arr == -32768 )
)
    
    

def find_channel(string):

  last_underscore_index = string.rfind('_')
  dot_index = string.rfind('.')

  if last_underscore_index != -1 and dot_index != -1 and last_underscore_index < dot_index:
      return string[last_underscore_index + 1:dot_index]

  return None




def display_images(images, limit=-1):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    for i, ax in enumerate(axes.flat):
        im = np.squeeze(images[i])#[:limit,:limit]
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def read_channel_by_dates(in_folder, tile, dates, channel, prefix, window):
    all_data = []
    for date in dates:
        path = os.path.join(in_folder, date, f'{prefix}_{tile}_{date}_{channel}.tif')
        with rasterio.open(path) as src:
            data = src.read(window = window)
            meta = src.meta
        all_data.append(data)
    return np.concatenate(all_data, axis=0), meta



def composition_SCL(SCL, data, option, scl_min, scl_max):
    """
    Process data based on SCL values within a given range and return either average or median.
    
    Args:
    SCL (numpy array): Array of shape (num_dates, width, height) with values to filter data.
    data (numpy array): Array of shape (num_dates, width, height) containing values to process.
    option (str): Either 'Average' or 'Median' to determine the type of processing.
    scl_min (int): Minimum SCL value for range.
    scl_max (int): Maximum SCL value for range.
    
    Returns:
    numpy array: Processed array of shape (1, 256, 256).
    """
    # Create a mask where SCL values are within the given range
    mask = (SCL >= scl_min) & (SCL <= scl_max)
    filtered_data = np.where(mask, data, np.nan)
    
    if option == 'Average':
        result = np.nanmean(filtered_data, axis=0, dtype=np.float32)
    elif option == 'Median':
        result = np.nanmedian(filtered_data, axis=0, dtype=np.float32)
    else:
        raise ValueError("Option must be 'Average' or 'Median'")
    return result

def find_date(string, prefix):
    prefix_index = string.find(prefix)

    if prefix_index != -1:
        start_index = prefix_index + len(prefix)
        end_index = start_index+8
        if end_index != -1 and end_index <= len(string):
            digits_str = string[start_index:end_index]
            try:
                return digits_str
            except ValueError:
                return None
    return None


def create_composition(in_folder, 
                       out_folder, 
                       tile = '032027', 
                       num_subtiles = 4, 
                       prefix = 'S2-16D_V2',
                       option = 'Average',
                       rewrite = False
                       ):
    in_folder = os.path.join(in_folder, f'{prefix}_{tile}')
    dates = os.listdir(in_folder)
    dates = [d for d in dates if d.isnumeric() and len(d)==8]
    print('Dates:')
    print(os.listdir(in_folder))

    print(f'Creating composition of {len(dates)} dates')
    channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']#, 'EVI', 'NBR', 'SCL', 'NDVI', 'CLEAROB', 'TOTALOB', 'thumbnail', 'PROVENANCE'])
    width = 10560

    
    subtile_width = width//num_subtiles
    for i in range(0,width, subtile_width):
        for j in range(0,width, subtile_width):

            full_save_dir = os.path.join(out_folder, f'{prefix}_{tile}', f'{num_subtiles}x{num_subtiles}_subtiles')
            if not os.path.exists(full_save_dir):
                os.makedirs(full_save_dir)
                print(f'Creating folder {full_save_dir}')            
            
            save_path = os.path.join(full_save_dir, f'{prefix}_{tile}_x={i}_y={j}.tif')


            if not rewrite:
                try:
                    with rasterio.open(save_path) as src:
                        # Read the raster data
                        data = src.read()
                        if data.shape == (12,width//num_subtiles, width//num_subtiles):
                            print(f'File {save_path} already exists, skipping.')
                            continue 
                        else: 
                            raise Exception("Incorrect data shape") 
                except:
                    pass
            print(f'Generating {save_path}...')


            #image composition

            channel_images = []
            window = rasterio.windows.Window(i ,j , subtile_width, subtile_width)
            scl_data, meta = read_channel_by_dates(in_folder, tile, dates, 'SCL', prefix, window)
            
            for ch in channels:
                channel_i, meta = read_channel_by_dates(in_folder, tile, dates, ch, prefix, window)
                SCL_composed_image = composition_SCL(scl_data, channel_i, option = option, scl_min=4, scl_max=6)
                #SCL_composed_image = preprocess_data.interpolate_nan(SCL_composed_image, interpolation_method= 'median', value_type = 'negative')
                SCL_composed_image = interpolate_nan(SCL_composed_image, interpolation_method= 'median', value_type = 'negative')
                channel_images.append(SCL_composed_image)
                #print(ch, channel_images[-1].shape, channel_images[-1].dtype)
                #print(SCL_composed_image[:10,:10])
                
            channel_composition = np.stack(channel_images, axis=0)
            channel_composition=channel_composition.astype(np.int16)
            #print_top_max_min(channel_composition, n=10)


            

            meta['width']= subtile_width
            meta['height']= subtile_width
            meta['count'] = channel_composition.shape[0]
            print('META', meta)

            with rasterio.open(save_path, 'w', **meta) as dst:
                dst.write(channel_composition)
                print(save_path,'saved')
            display_images(np.abs(channel_composition))
            print(f'subtile {i}, {j}')
            print(channel_composition.shape)

    return
    #TODO: check nan, change bounds, 

     


from scipy.interpolate import griddata
from scipy.ndimage import median_filter
def interpolate_nan(data, interpolation_method='linear', value_type='regular', fixed_value=-1):
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
    num_nans = data.size - np.sum(nan_mask)
    if num_nans == 0 or not np.any(nan_mask):
       # If there are no NaN values, return the original array
        return data

    # Get the indices of the NaN values
    nan_indices = np.argwhere(nan_mask)

    # Get the indices of the non-NaN values
    non_nan_indices = np.argwhere(~nan_mask)

    # Get the values of the non-NaN values
    non_nan_values = data[~nan_mask]

    # Create a copy of the original array to store the result
    result = data.copy()

    # For linear or cubic interpolation, it needs at least 4 points:
    # QhullError: QH6214 qhull input error: not enough points(2) to construct initial simplex (need 4)
    x_coords = non_nan_indices[:, 0]
    y_coords = non_nan_indices[:, 1]
    do_median = num_nans < 4 or np.all(x_coords == x_coords[0]) or np.all(y_coords == y_coords[0]) or interpolation_method == 'median' 
    np.all(x_coords == x_coords[0])  # Check if all x-coordinates are the same
    
    if interpolation_method == 'linear' and not do_median:
        # Linear interpolation
        interpolated_values = griddata(non_nan_indices, non_nan_values, nan_indices, method='linear')
    elif interpolation_method == 'cubic' and not do_median:
        # Cubic interpolation
        interpolated_values = griddata(non_nan_indices, non_nan_values, nan_indices, method='cubic')
    elif do_median:
        # Median filter interpolation
        filtered_data = median_filter(data, size=5)
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

