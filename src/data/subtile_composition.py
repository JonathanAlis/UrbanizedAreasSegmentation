
import os
import sys

import rasterio.windows
sys.path.append(os.path.abspath('..'))
from rasterio.coords import BoundingBox

import src.data.preprocess_data as preprocess_data

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


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




def display_images(images: np.ndarray, limit: int =-1, save_to: str|None = None):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    for i, ax in enumerate(axes.flat):
        im = np.squeeze(images[i])[:limit,:limit]
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)

def read_channel_by_dates(in_folder:str, tile:str, dates:list, channel:str, prefix:str, window:rasterio.windows.Window):
    all_data = []
    for date in dates:
        path = os.path.join(in_folder, date, f'{prefix}_{tile}_{date}_{channel}.tif')
        with rasterio.open(path) as src:
            data = src.read(window = window)
            meta = src.meta
        all_data.append(data)
    return np.concatenate(all_data, axis=0), meta



def composition_SCL(SCL:np.ndarray, data:np.ndarray, option:str, scl_min:int, scl_max:int):
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
        result = np.nanmedian(filtered_data, axis=0)
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

### Composição temporal
def create_composition(in_folder, 
                       out_folder, 
                       tile = '032027', 
                       num_subtiles = 6, 
                       prefix = 'S2-16D_V2',
                       option = 'Average',
                       rewrite = False,
                       display = 3
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
                SCL_composed_image =  preprocess_data.fill_nans_nearest(SCL_composed_image, negate_filled = True)
                channel_images.append(SCL_composed_image)
                
            channel_composition = np.stack(channel_images, axis=0)
            channel_composition=channel_composition.astype(np.int16)
            #print_top_max_min(channel_composition, n=10)

            meta['width']= subtile_width
            meta['height']= subtile_width
            meta['count'] = channel_composition.shape[0]
            #transform = src.meta['transform']
            print('META', meta)

            with rasterio.open(save_path, 'w', **meta) as dst:
                dst.write(channel_composition)
                print(save_path,'saved')
            if display>0:
                display_images(np.abs(channel_composition))
                display -= 1
            print(f'subtile {i}, {j}')
            print(channel_composition.shape)

    return
    #TODO: check nan, change bounds, 


### Composição temporal
def composition_PCA(in_folder, 
                       out_folder, 
                       tile = '032027',                       
                       n_components = 12, 
                       num_subtiles = 6, 
                       prefix = 'S2-16D_V2',
                       option = 'Average',
                       rewrite = False,
                       display = 3
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

            full_save_dir = os.path.join(out_folder, f'{prefix}_{tile}', f'{num_subtiles}x{num_subtiles}_subtiles', f'{n_components}_components')
            if not os.path.exists(full_save_dir):
                os.makedirs(full_save_dir)
                print(f'Creating folder {full_save_dir}')            
            
            save_path = os.path.join(full_save_dir, f'{prefix}_{tile}_x={i}_y={j}.tif')


            if not rewrite:
                try:
                    with rasterio.open(save_path) as src:
                        # Read the raster data
                        data = src.read()
                        if data.shape == (n_components,width//num_subtiles, width//num_subtiles):
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
                SCL_composed_image =  preprocess_data.fill_nans_nearest(SCL_composed_image, negate_filled = False)
                channel_images.append(SCL_composed_image)
                
            channel_composition = np.stack(channel_images, axis=0)


            channel_composition=channel_composition.astype(np.int16)
            #print_top_max_min(channel_composition, n=10)

            meta['width']= subtile_width
            meta['height']= subtile_width
            meta['count'] = channel_composition.shape[0]
            #transform = src.meta['transform']
            print('META', meta)

            with rasterio.open(save_path, 'w', **meta) as dst:
                dst.write(channel_composition)
                print(save_path,'saved')
            if display>0:
                display_images(np.abs(channel_composition))
                display -= 1
            print(f'subtile {i}, {j}')
            print(channel_composition.shape)

    return



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


def quantize_to_uint8(image):
    """
    Quantize a single multi-channel image to uint8 format.
    
    Parameters:
        image (np.ndarray): Input image with shape (n_channels, height, width).
    
    Returns:
        quantized_image (np.ndarray): Quantized image with shape (n_channels, height, width).
    """
    n_channels, height, width = image.shape
    quantized_channels = []
    
    for channel in range(n_channels):
        channel_data = image[channel]
        lower = np.percentile(channel_data, 2)  # 2nd percentile
        upper = np.percentile(channel_data, 98)  # 98th percentile
        
        # Check if the range is zero (constant channel)
        if upper == lower:
            # Assign a default value (e.g., 0) for this channel
            quantized_channel = np.zeros_like(channel_data, dtype=np.uint8)
        else:
            # Normalize and clip to [0, 255]
            quantized_channel = np.clip((channel_data - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
        
        quantized_channels.append(quantized_channel)
    
    # Stack the quantized channels back together
    quantized_image = np.stack(quantized_channels, axis=0)
    return quantized_image




import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error

def test_incremental_pca(image_list, max_components=None, batch_size=1000, plot_error = True):
    """
    Test Incremental PCA with different numbers of components, calculate explained variance,
    reconstruction error, and plot the results in four graphs.
    
    Parameters:
        image_list (list of np.ndarray): List of input images, each with shape (n_channels, height, width).
        max_components (int, optional): Maximum number of components to test. 
                                         If None, defaults to n_channels.
        batch_size (int): Number of pixels per batch for Incremental PCA.
    
    Returns:
        ipca: Fitted IncrementalPCA object.
    """
    # Get dimensions of the first image to determine n_channels
    n_channels, height, width = image_list[0].shape
    
    # Combine all images into a single dataset for PCA fitting
    all_pixels = []
    for image in image_list:
        pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
        all_pixels.append(pixels)
    all_pixels = np.vstack(all_pixels)  # Shape: (total_n_pixels, n_channels)
    
    # Determine the maximum number of components to test
    if max_components is None:
        max_components = n_channels
    
    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=max_components)
    
    # Fit Incremental PCA in mini-batches
    for i in range(0, all_pixels.shape[0], batch_size):
        batch = all_pixels[i:i + batch_size]
        ipca.partial_fit(batch)
    
    # Calculate explained variance and reconstruction errors
    explained_variance = ipca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    if plot_error:
        reconstruction_errors = []
        for n_components in range(1, max_components + 1):
            reduced_data = ipca.transform(all_pixels)[:, :n_components]  # Transform and select n_components
            mse = compute_reconstruction_error(all_pixels, reduced_data, ipca)
            reconstruction_errors.append(mse)
    
    # Plot the results
    plt.figure(figsize=(16, 8))
    
    # 1. Explained Variance per Component
    plt.subplot(2, 2, 1)
    plt.plot(range(1, max_components + 1), explained_variance, marker='o', linestyle='-', color='b')
    plt.title('Explained Variance per Component')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    
    # 2. Cumulative Explained Variance
    plt.subplot(2, 2, 2)
    plt.plot(range(1, max_components + 1), cumulative_variance, marker='o', linestyle='-', color='r')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Variance')  # Optional: Highlight 95% threshold
    plt.legend()
    plt.grid(True)
    
    # 3. Reconstruction Error vs. Number of Components
    plt.subplot(2, 2, 3)
    plt.plot(range(1, max_components + 1), reconstruction_errors, marker='o', linestyle='-', color='g')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    # 4. Combined View: Cumulative Variance and Reconstruction Error
    plt.subplot(2, 2, 4)
    plt.plot(range(1, max_components + 1), cumulative_variance, marker='o', linestyle='-', color='r', label='Cumulative Variance')
    plt.plot(range(1, max_components + 1), np.array(reconstruction_errors) / max(reconstruction_errors), marker='o', linestyle='-', color='g', label='Normalized MSE')
    plt.title('Combined View: Variance and Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Metrics')
    plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Variance')  # Optional: Highlight 95% threshold
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return ipca

def reduce_image_channels_incremental_pca_channel_first(image_list, n_components, batch_size=1000):
    """
    Reduce the number of channels in a list of multi-channel images with
    channel-first format (n_channels, height, width) using Incremental PCA.
    
    Parameters:
        image_list (list of np.ndarray): List of input images, each with shape (n_channels, height, width).
        n_components (int): Number of components to keep after PCA.
        batch_size (int): Number of pixels per batch for Incremental PCA.
    
    Returns:
        reduced_image_list (list of np.ndarray): List of reduced images, each with shape (n_components, height, width).
    """
    # Get dimensions of the first image to determine n_channels
    n_channels, height, width = image_list[0].shape
    
    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Fit Incremental PCA on all images
    for image in image_list:
        pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
        n_pixels = pixels.shape[0]
        
        # Fit Incremental PCA in mini-batches
        for i in range(0, n_pixels, batch_size):
            batch = pixels[i:i + batch_size]
            ipca.partial_fit(batch)
    
    # Apply PCA to each image individually
    reduced_image_list = []
    for image in image_list:
        pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
        n_pixels = pixels.shape[0]
        
        # Transform pixels in mini-batches
        reduced_pixels = []
        for i in range(0, n_pixels, batch_size):
            batch = pixels[i:i + batch_size]
            reduced_batch = ipca.transform(batch)
            reduced_pixels.append(reduced_batch)
        
        reduced_pixels = np.vstack(reduced_pixels)  # Combine all transformed batches
        reduced_image = reduced_pixels.T.reshape(n_components, height, width)  # Reshape back to (n_components, height, width)
        reduced_image_list.append(reduced_image)
    
    return reduced_image_list



from sklearn.metrics import mean_squared_error

def compute_reconstruction_error(original_data, reduced_data, ipca):
    """
    Compute the reconstruction error after PCA.
    
    Parameters:
        original_data (np.ndarray): Original data with shape (n_pixels, n_channels).
        reduced_data (np.ndarray): Reduced data with shape (n_pixels, n_components).
        ipca (IncrementalPCA): Fitted IncrementalPCA object.
    
    Returns:
        mse (float): Mean squared error between original and reconstructed data.
    """

    # Pad reduced_data with zeros to match the expected number of components
    n_components = reduced_data.shape[1]
    padded_data = np.zeros((reduced_data.shape[0], ipca.n_components_))
    padded_data[:, :n_components] = reduced_data
    
    # Reconstruct the data
    reconstructed_data = ipca.inverse_transform(padded_data)
    
    # Compute the mean squared error
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse



import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error

class IPCAHandler:
    """
    A class to handle Incremental PCA operations, including fitting, transforming,
    saving/loading models, and partial fitting.
    """
    def __init__(self, n_components, batch_size=1000):
        """
        Initialize the IPCAHandler.
        
        Parameters:
            n_components (int): Number of components for PCA.
            batch_size (int): Number of samples per batch for Incremental PCA.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = None  # Placeholder for the IncrementalPCA object
    
    def fit_incremental_pca(self, image_list):
        """
        Fit or partially fit an Incremental PCA model to a list of images.
        
        Parameters:
            image_list (list of np.ndarray): List of input images, each with shape (n_channels, height, width).
        """
        # Get dimensions of the first image to determine n_channels
        n_channels, height, width = image_list[0].shape
        
        # Initialize Incremental PCA only if it hasn't been initialized yet
        if self.ipca is None:
            self.ipca = IncrementalPCA(n_components=self.n_components)
        
        # Combine all images into a single dataset for PCA fitting
        all_pixels = []
        for image in image_list:
            pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
            all_pixels.append(pixels)
        all_pixels = np.vstack(all_pixels)  # Shape: (total_n_pixels, n_channels)
        
        # Fit Incremental PCA in mini-batches
        for i in range(0, all_pixels.shape[0], self.batch_size):
            batch = all_pixels[i:i + self.batch_size]
            self.ipca.partial_fit(batch)
    
    def save_model(self, file_prefix="ipca_model"):
        """
        Save the fitted Incremental PCA model to a file.
        
        Parameters:
            file_prefix (str): Prefix for the filename. The number of components will be appended.
        """
        if self.ipca is None:
            raise ValueError("No fitted IPCA model to save. Call fit_incremental_pca first.")
        
        # Create a filename based on the number of components
        filename = f"{file_prefix}_ncomponents_{self.n_components}.pkl"
        
        # Save the model using joblib
        import joblib
        joblib.dump(self.ipca, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, file_prefix="ipca_model"):
        """
        Load a previously saved Incremental PCA model from a file.
        
        Parameters:
            file_prefix (str): Prefix for the filename. The number of components will be appended.
        """
        # Construct the filename based on the number of components
        filename = f"{file_prefix}_ncomponents_{self.n_components}.pkl"
        
        # Load the model using joblib
        import joblib
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        
        self.ipca = joblib.load(filename)
        print(f"Model loaded from {filename}")
    
    def transform_data(self, image_list):
        """
        Transform a list of images using the fitted IPCA model.
        
        Parameters:
            image_list (list of np.ndarray): List of input images, each with shape (n_channels, height, width).
        
        Returns:
            transformed_images (list of np.ndarray): Transformed images with reduced dimensions.
        """
        if self.ipca is None:
            raise ValueError("No fitted IPCA model to use for transformation. Call fit_incremental_pca or load_model first.")
        
        transformed_images = []
        for image in image_list:
            n_channels, height, width = image.shape
            pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
            transformed_pixels = self.ipca.transform(pixels)[:, :self.n_components]  # Reduce to n_components
            transformed_image = transformed_pixels.T.reshape(self.n_components, height, width)
            transformed_images.append(transformed_image)
        
        return transformed_images
    

def test_ipca_with_error(image_list, max_components, batch_size=1000):
    """
    Test Incremental PCA with different numbers of components, calculate explained variance,
    reconstruction error, and plot the results.
    
    Parameters:
        image_list (list of np.ndarray): List of input images, each with shape (n_channels, height, width).
        max_components (int): Maximum number of components to test.
        batch_size (int): Number of samples per batch for Incremental PCA.
    """
    # Combine all images into a single dataset for testing
    n_channels, height, width = image_list[0].shape
    all_pixels = []
    for image in image_list:
        pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
        all_pixels.append(pixels)
    all_pixels = np.vstack(all_pixels)  # Shape: (total_n_pixels, n_channels)
    
    # Lists to store metrics for plotting
    explained_variances = []
    cumulative_variances = []
    reconstruction_errors = []
    
    for n_components in range(1, max_components + 1):
        # Create an IPCAHandler instance for this n_components
        ipca_handler = IPCAHandler(n_components=n_components, batch_size=batch_size)
        ipca_handler.fit_incremental_pca(image_list)
        
        # Calculate explained variance
        explained_variance = ipca_handler.ipca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        explained_variances.append(explained_variance)
        cumulative_variances.append(cumulative_variance[-1])
        
        # Calculate reconstruction error
        reduced_data = ipca_handler.ipca.transform(all_pixels)[:, :n_components]
        mse = compute_reconstruction_error(all_pixels, reduced_data, ipca_handler.ipca)
        reconstruction_errors.append(mse)
    
    # Plot the results
    plt.figure(figsize=(16, 8))
    
    # 1. Explained Variance per Component
    plt.subplot(2, 2, 1)
    for n_components, explained_variance in enumerate(explained_variances, start=1):
        plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='-', label=f"{n_components} Components")
    plt.title('Explained Variance per Component')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True)
    
    # 2. Cumulative Explained Variance
    plt.subplot(2, 2, 2)
    plt.plot(range(1, max_components + 1), cumulative_variances, marker='o', linestyle='-', color='r')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Variance')  # Optional: Highlight 95% threshold
    plt.legend()
    plt.grid(True)
    
    # 3. Reconstruction Error vs. Number of Components
    plt.subplot(2, 2, 3)
    plt.plot(range(1, max_components + 1), reconstruction_errors, marker='o', linestyle='-', color='g')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    # 4. Combined View: Cumulative Variance and Reconstruction Error
    plt.subplot(2, 2, 4)
    plt.plot(range(1, max_components + 1), cumulative_variances, marker='o', linestyle='-', color='r', label='Cumulative Variance')
    plt.plot(range(1, max_components + 1), np.array(reconstruction_errors) / max(reconstruction_errors), marker='o', linestyle='-', color='g', label='Normalized MSE')
    plt.title('Combined View: Variance and Error')
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized Metrics')
    plt.axhline(y=0.95, color='gray', linestyle='--', label='95% Variance')  # Optional: Highlight 95% threshold
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compute_reconstruction_error(original_data, reduced_data, ipca):
    """
    Compute the reconstruction error after PCA.
    
    Parameters:
        original_data (np.ndarray): Original data with shape (n_pixels, n_channels).
        reduced_data (np.ndarray): Reduced data with shape (n_pixels, n_components).
        ipca (IncrementalPCA): Fitted IncrementalPCA object.
    
    Returns:
        mse (float): Mean squared error between original and reconstructed data.
    """
    # Pad reduced_data with zeros to match the expected number of components
    n_components = reduced_data.shape[1]
    padded_data = np.zeros((reduced_data.shape[0], ipca.n_components_))
    padded_data[:, :n_components] = reduced_data
    
    # Reconstruct the data
    reconstructed_data = ipca.inverse_transform(padded_data)
    
    # Compute the mean squared error
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse