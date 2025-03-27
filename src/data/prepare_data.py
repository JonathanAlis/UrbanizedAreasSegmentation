    

import rasterio
import matplotlib.pyplot as plt
import os
import numpy as np

import sys
sys.path.append(os.path.abspath('..'))
import src.data.preprocess_data as preprocess_data


def prepare_image(tile, channel, window, working_dir = None):
    #window = rasterio.windows.Window(x, y, subtile_size, subtile_size)
    if working_dir is None:
        working_dir = os.path.abspath('..')

    ### Le todas as datas
    all_filtered = []
    dates = os.listdir(f'{working_dir}/data/raw/S2-16D_V2_{tile}')
    for date in dates:

        ### Abre o canal SCL
        scl = f'{working_dir}/data/raw/S2-16D_V2_{tile}/{date}/S2-16D_V2_{tile}_{date}_SCL.tif'
        with rasterio.open(scl) as scr:
            scl_channel = np.squeeze(scr.read(window = window)) 
        mask = (scl_channel >= 4) & (scl_channel <= 6)

        ### Abre o canal 
        data_path = f'{working_dir}/data/raw/S2-16D_V2_{tile}/{date}/S2-16D_V2_{tile}_{date}_{channel}.tif'
        with rasterio.open(data_path) as scr:
            channel_data = np.squeeze(scr.read(window = window))

        ### Filtra, só onde o mask SCL é entre 4 e 6. 
        filtered_data = np.where(mask, channel_data, np.nan)
        all_filtered.append(filtered_data)

    ### Todas as datas em um só numpy array
    all_filtered = np.stack(all_filtered, axis=0)
    
    ### Composição temporal, por mediana
    composite = np.nanmedian(all_filtered, axis=0)
    
    interpolated = preprocess_data.fill_nans_nearest(composite, negate_filled=False).astype(np.int16)

    return interpolated
    
def quantize(image, low_percentile = 2, high_percentile = 98, lower_fixed = None, higher_fixed = None):
    ### default: 2, 98 percentiles.
    ### if fixed defined, use them instead
    lower = np.percentile(image, low_percentile)  # 2nd percentile
    upper = np.percentile(image, high_percentile)  # 98th percentile
    #print(lower, upper)
    if lower_fixed:
        lower = lower_fixed
    if higher_fixed:
        upper = higher_fixed
        
    # Check if the range is zero (constant channel)
    if upper == lower:
        # Assign a default value (e.g., 0) for this channel
        quantized = np.zeros_like(image, dtype=np.uint8)
    else:
        # Normalize and clip to [0, 255]
        quantized = np.clip((image - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)            
    return quantized

def prepare_tile(tile, channel, num_subtiles = 6, tile_size = 10560, working_dir = None):
    if working_dir is None:
        working_dir = os.path.abspath('..')
    subtile_size = tile_size//num_subtiles
    for x in range(0, tile_size, tile_size//num_subtiles):
        for y in range(0, tile_size, tile_size//num_subtiles):
            window = rasterio.windows.Window(x, y, subtile_size, subtile_size)
            quantized = prepare_image(tile=tile, channel=channel, window=window)
            yield quantized



import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib

class IPCAHandler:
    def __init__(self, n_components, batch_size=1000):
        """
        Initialize the IPCAHandlerWithFrozenNormalization.
        
        Parameters:
            n_components (int): Number of components for PCA.
            batch_size (int): Number of samples per batch for Incremental PCA.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=n_components)
        self.lower_percentile = None  # Frozen 2nd percentile
        self.upper_percentile = None  # Frozen 98th percentile
        self.is_frozen = False  # Flag to indicate if normalization is frozen
    
    def _reshape_data(self, images):
        """
        Reshape a list of images or a single image to (n_pixels, n_channels).
        
        Parameters:
            images (np.ndarray or list of np.ndarray): Input images with shape (n_channels, height, width).
        
        Returns:
            reshaped_data (np.ndarray): Reshaped data with shape (n_pixels, n_channels).
        """
        if isinstance(images, np.ndarray):  # Single image
            n_channels, height, width = images.shape
            return images.reshape(n_channels, -1).T  # Shape: (height * width, n_channels)
        
        elif isinstance(images, list):  # List of images
            reshaped_data = []
            for image in images:
                n_channels, height, width = image.shape
                reshaped_data.append(image.reshape(n_channels, -1).T)  # Shape: (height * width, n_channels)
            return np.vstack(reshaped_data)  # Combine all pixels
        
        else:
            raise ValueError("Unsupported input type. Expected np.ndarray or list of np.ndarray.")
    
    def fit(self, data_source):
        """
        Fit the IPCA model incrementally using either a list or a generator.
        
        Parameters:
            data_source (list or generator): A list of images or a generator yielding batches of data.
        """
        if self.is_frozen:
            raise ValueError("Normalization is frozen. Cannot fit new data after freezing.")
        
        # Process data from the source
        if isinstance(data_source, list):  # If data_source is a list
            all_pixels = self._reshape_data(data_source)
            
            # Fit Incremental PCA in mini-batches
            for i in range(0, all_pixels.shape[0], self.batch_size):
                batch = all_pixels[i:i + self.batch_size]
                self.ipca.partial_fit(batch)
        
        else:  # If data_source is a generator
            for batch in data_source:
                # Reshape the batch to (n_pixels, n_channels)
                pixels = self._reshape_data(batch)
                self.ipca.partial_fit(pixels)
    
    def transform_and_quantize(self, data_source):
        """
        Transform and quantize data using the fitted IPCA model and frozen stats.
        Freezes normalization statistics after the first transformation.
        
        Parameters:
            data_source (np.ndarray, list, or generator): 
                - Single image (np.ndarray) with shape (n_channels, height, width).
                - List of images (list of np.ndarray), each with shape (n_channels, height, width).
                - Generator yielding individual images.
        
        Returns:
            quantized_images (list of np.ndarray): Quantized images with shape (n_components, height, width).
        """
        if not hasattr(self.ipca, "components_"):
            raise ValueError("IPCA model has not been fitted yet. Call fit first.")
        
        # Normalize input to always work with a generator
        if isinstance(data_source, np.ndarray):  # Single image
            data_source = [data_source]  # Convert to a single-element list
        
        if isinstance(data_source, list):  # List of images
            data_source = iter(data_source)  # Convert to a generator
        
        # Freeze normalization statistics if not already frozen
        if not self.is_frozen:
            self._freeze_normalization(data_source)
            self.is_frozen = True
        
        # Reset the generator (if needed) and process all images
        quantized_images = []
        for image in data_source:
            quantized_image = self._transform_and_quantize_single(image)
            quantized_images.append(quantized_image)
        
        return quantized_images
    
    def _transform_and_quantize_single(self, image):
        """
        Helper method to transform and quantize a single image.
        
        Parameters:
            image (np.ndarray): Input image with shape (n_channels, height, width).
        
        Returns:
            quantized_image (np.ndarray): Quantized image with shape (n_components, height, width).
        """
        n_channels, height, width = image.shape
        pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
        transformed_pixels = self.ipca.transform(pixels)  # Transform to (n_pixels, n_components)
        
        # Normalize and quantize each channel
        quantized_channels = []
        for channel in range(self.n_components):
            channel_data = transformed_pixels[:, channel]
            lower = self.lower_percentile[channel]
            upper = self.upper_percentile[channel]
            
            # Normalize and clip to [0, 255]
            quantized_channel = np.clip((channel_data - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
            quantized_channels.append(quantized_channel.reshape(height, width))
        
        return np.stack(quantized_channels, axis=0)
    
    def _freeze_normalization(self, data_source):
        """
        Compute and freeze normalization statistics based on the provided data source.
        
        Parameters:
            data_source (generator): Generator yielding individual images.
        """
        all_transformed_data = []
        
        for image in data_source:
            n_channels, height, width = image.shape
            pixels = image.reshape(n_channels, -1).T  # Reshape to (n_pixels, n_channels)
            transformed_pixels = self.ipca.transform(pixels)  # Transform to (n_pixels, n_components)
            all_transformed_data.append(transformed_pixels)
        
        combined_transformed_data = np.vstack(all_transformed_data)
        
        # Compute and freeze the 2nd and 98th percentiles
        self.lower_percentile = np.percentile(combined_transformed_data, 2, axis=0)
        self.upper_percentile = np.percentile(combined_transformed_data, 98, axis=0)
    
    def save_model(self, file_prefix="ipca_model"):
        """
        Save the fitted IPCA model and frozen normalization statistics to a file.
        
        Parameters:
            file_prefix (str): Prefix for the filename. The number of components will be appended.
        """
        
        # Create a filename based on the number of components
        filename = f"{file_prefix}_ncomponents_{self.n_components}.pkl"
        
        # Save the model and statistics using joblib
        joblib.dump({
            "ipca": self.ipca,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "is_frozen": self.is_frozen
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, file_prefix="ipca_model", n_components=None):
        """
        Load a previously saved IPCA model and frozen normalization statistics from a file.
        
        Parameters:
            file_prefix (str): Prefix for the filename. The number of components will be appended.
            n_components (int, optional): Number of components to load. If None, defaults to self.n_components.
        """
        if n_components is None:
            n_components = self.n_components
        
        # Construct the filename based on the number of components
        filename = f"{file_prefix}_ncomponents_{n_components}.pkl"
        
        # Load the model and statistics using joblib
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        
        loaded_data = joblib.load(filename)
        self.ipca = loaded_data["ipca"]
        self.lower_percentile = loaded_data["lower_percentile"]
        self.upper_percentile = loaded_data["upper_percentile"]
        self.is_frozen = loaded_data["is_frozen"]
        self.n_components = n_components
        print(f"Model loaded from {filename}")

import numpy as np
import random

from tqdm import tqdm

def batch_generator(single_image_generator, batch_size, total_images, shuffle=True):
    """
    Generator that yields batches of images from a single-image generator with progress tracking.
    
    Parameters:
        single_image_generator (generator): A generator that yields individual images.
        batch_size (int): Number of images per batch.
        total_images (int): Total number of images in the dataset.
        shuffle (bool): Whether to shuffle the images before batching.
    
    Yields:
        batch (list of np.ndarray): A batch of images with length `batch_size`.
    """
    buffer = []  # Temporary buffer to collect images
    
    # Wrap the generator with tqdm for progress tracking
    progress_bar = tqdm(total=total_images, desc="Processing", unit="image")
    
    for image in single_image_generator:
        buffer.append(image)
        progress_bar.update(1)  # Increment the progress bar
        
        # If the buffer reaches the batch size, yield a batch
        if len(buffer) >= batch_size:
            if shuffle:
                random.shuffle(buffer)  # Shuffle the buffer to randomize order
            yield buffer[:batch_size]  # Yield a batch of size `batch_size`
            buffer = buffer[batch_size:]  # Remove the yielded images from the buffer
    
    # Yield any remaining images in the buffer (if not empty)
    if buffer:
        if shuffle:
            random.shuffle(buffer)  # Shuffle the remaining images
        yield buffer
    
    # Close the progress bar when done
    progress_bar.close()


from pyproj import CRS
from rasterio.transform import Affine

def save(img, x, y, tile, num_subtiles = 6, working_dir = None):
    #utiliza a mascara só para obter a transform
    if working_dir is None:
        working_dir = os.path.abspath('..')
    original_mask = os.path.join(working_dir, 'data', 'masks', f'mask_raster_{tile}.tif') 
    if not os.path.exists(original_mask):
        print(f"mask for tile {tile} do not exist")
    with rasterio.open(original_mask) as src:
        full_tile_transform = src.meta['transform']

    dtype = img.dtype
    num_channels = img.shape[0]
    folder = f"{working_dir}/data/processed/S2-16D_V2_{tile}/{num_subtiles}x{num_subtiles}_subtiles/"
    
    if dtype != np.int16:
        folder+=f'q_{num_channels}ch/'

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    output_file = os.path.join(folder, f'x={x}_y={y}.tif')
    print(f"Saving as {output_file}")
    
    a, b, c, d, e, f  = full_tile_transform.to_gdal()  # or use transform.a, transform.e, etc.
    new_c = c + x
    new_f = f + y
    subtile_transform = Affine(a, b, new_c, d, e, new_f)         

    custom_crs_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown based on GRS80 ellipsoid",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",-12],PARAMETER["longitude_of_center",-54],PARAMETER["standard_parallel_1",-2],PARAMETER["standard_parallel_2",-22],PARAMETER["false_easting",5000000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    crs = CRS.from_wkt(custom_crs_wkt)

    meta = {"tile": tile,
            #"source": "Generated by ReconstructTile: https://github.com/JonathanAlis/UrbanizedAreasSegmentation/blob/main/src/training/post_processing.py",
            #"author": "Jonathan Alis",
            }
    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=10560//num_subtiles,
        width=10560//num_subtiles,
        count=num_channels,
        dtype=img.dtype,
        crs = crs,
        transform=subtile_transform,
        compress='DEFLATE',
        predictor=1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=254
    ) as dst:
        dst.write(img)
        dst.update_tags(**meta)


