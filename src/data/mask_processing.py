import os
import json
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
import numpy as np
from rasterio.enums import MergeAlg
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import CRS
import fiona
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Hardcoded mapping of (Tipo, Densidade) to label_id
label_mapping = {
    ('Loteamento vazio', 'Loteamento vazio'): 1,
    ('Outros equipamentos urbanos', 'Pouco densa'): 2,
    ('Outros equipamentos urbanos', 'Densa'): 3,
    ('Vazio intraurbano', 'Pouco densa'): 4,
    ('Vazio intraurbano', 'Densa'): 5,
    ('Vazio intraurbano remanescente', 'Pouco densa'): 6,
    ('Vazio intraurbano remanescente', 'Densa'): 7,
    ('Área urbanizada', 'Pouco densa'): 8,
    ('Área urbanizada', 'Densa'): 9,
}

value_label_mapping = {
    1: 'Loteamento vazio, Loteamento vazio',
    2: 'Outros equipamentos urbanos, Pouco densa',
    3: 'Outros equipamentos urbanos, Densa',
    4: 'Vazio intraurbano, Pouco densa',
    5: 'Vazio intraurbano, Densa',
    6: 'Vazio intraurbano remanescente, Pouco densa',
    7: 'Vazio intraurbano remanescente, Densa',
    8: 'Área urbanizada, Pouco densa',
    9: 'Área urbanizada, Densa'
}

density_mapping = {
    1: 'Loteamento vazio',
    2: 'Pouco densa',
    3: 'Densa',
    4: 'Pouco densa',
    5: 'Densa',
    6: 'Pouco densa',
    7: 'Densa',
    8: 'Pouco densa',
    9: 'Densa'
}

def get_density(mask): #based on the density_mapping
    lookup_table = np.array([0, 1, 2, 3, 2, 3, 2, 3, 2, 3]) #0->0, 1->1, 2->2, 3->3, 4->2, etc
    density = np.take(lookup_table, mask)
    return density

type_mapping = {
    1: 'Loteamento vazio',
    2: 'Outros equipamentos urbanos',
    3: 'Outros equipamentos urbanos',
    4: 'Vazio intraurbano',
    5: 'Vazio intraurbano',
    6: 'Vazio intraurbano',
    7: 'Vazio intraurbano',
    8: 'Área urbanizada',
    9: 'Área urbanizada'
}

def get_type(mask): #based on the type_mapping
    lookup_table = np.array([0, 1, 2, 2, 3, 3, 3, 3, 4, 4]) #0->0, 1->1, 2->2, 3->2, 4->3, etc
    type = np.take(lookup_table, mask)
    return type

# Example Usage:
# mapping = load_mapping("path/to/directory")
def apply_mapping(gdf, mapping):
    """
    Applies a loaded mapping to a GeoDataFrame to create the 'label_id' column.
    """
    gdf["label_id"] = gdf.apply(lambda row: mapping.get((str(row['Tipo']), str(row['Densidade']))), axis=1)
    return gdf


def filter_polygons_by_bounding_box(gdf, bbox_list):
    # Step 1: Create a bounding box polygon
    minx, miny, maxx, maxy = bbox_list
    bounding_box = box(minx, miny, maxx, maxy)
    
    # Step 2: Filter the GeoDataFrame to keep only polygons that intersect with the bounding box
    filtered_gdf = gdf[gdf.intersects(bounding_box)]
    
    return filtered_gdf


class RasterizeMasks:
    def __init__(self, urban_shp_path, bdc_grid_path, custom_crs_wkt, resolution=10):
        """
        Initializes the RasterizeMasks class with shapefiles and parameters.
        
        Args:
            urban_shp_path (str): Path to the urban shapefile.
            bdc_grid_path (str): Path to the BDC grid shapefile.
            custom_crs_wkt (str): Custom CRS in WKT format.
            resolution (int, optional): Raster resolution in meters. Defaults to 10.
        """
        # Load the urban shapefile and convert it to the custom CRS
        self.urban_shp = gpd.read_file(urban_shp_path)
        self.custom_crs = CRS.from_wkt(custom_crs_wkt)
        self.urban_shp = self.urban_shp.to_crs(self.custom_crs)
        
        # Load and transform the BDC grid shapefile to the same CRS
        self.bdc_grid = gpd.read_file(bdc_grid_path, engine='fiona')
        self.bdc_grid = self.bdc_grid.to_crs(self.custom_crs)
        
        # Set rasterization resolution
        self.resolution = resolution

    def raster_tile(self, tile_name, save_path = None):
        """
        Rasterize the polygons of a specific tile into a numpy array.
        Optionally saves the raster to a file if `save_path` is provided.
        
        Args:
            tile_name (str): The name of the tile to rasterize.
            save_path (str, optional): Path to save the rasterized image. Defaults to None.
        
        Returns:
            np.ndarray: The rasterized image as a numpy array.
        """

        # Select the tile from the grid by its name
        tile = self.bdc_grid[self.bdc_grid['tile'] == tile_name]
        
        # Perform a spatial intersection between the urban areas and the selected tile
        tile_polygons = gpd.overlay(self.urban_shp, tile, how='intersection')
        
        # Apply a label mapping to convert categories to numerical labels
        tile_polygons = apply_mapping(tile_polygons, label_mapping)
        
        # Get the unique labels and sort them to prioritize rasterization order
        unique_labels = tile_polygons['label_id'].unique()
        unique_labels.sort()

        # Calculate the raster bounds and output shape based on the resolution
        bounds = tile.total_bounds
        out_shape = (
            int((bounds[3] - bounds[1]) / self.resolution),
            int((bounds[2] - bounds[0]) / self.resolution)
        )
        
        # Define the affine transformation for rasterization
        transform = from_bounds(
            *bounds,
            width=int((bounds[2] - bounds[0]) / self.resolution),
            height=int((bounds[3] - bounds[1]) / self.resolution)
        )
        
        # Initialize the raster with a default value of 255 (no data)
        final_raster = 255 * np.ones(out_shape, dtype=np.uint8)

        # Rasterize each label in reverse order to handle nested polygons correctly
        for label in unique_labels[::-1]:
            print(f'Rasterizing id {label}: {list(label_mapping.items())[label-1]}')
            
            # Filter polygons belonging to the current label
            filtered_gdf = tile_polygons[tile_polygons['label_id'] == label]
            
            # Generate geometry-value pairs for rasterization
            shapes = (
                (geom, value)
                for geom, value in zip(filtered_gdf.geometry, filtered_gdf['label_id'])
            )
            
            # Rasterize the current group of geometries
            temp_raster = rasterize(
                shapes,
                out_shape=out_shape,
                fill=255,  # Background value
                transform=transform,
                all_touched=True,  # Include all pixels touched by the geometry
                merge_alg=MergeAlg.replace  # Replace existing values with the new ones
            )
            
            # Update the final raster by taking the minimum (to preserve priority)
            final_raster = np.minimum(final_raster, temp_raster)

        # Replace remaining 255 values with 0 (no label)
        final_raster[final_raster == 255] = 0

        # Store results in class variables
        self.current_mask = final_raster
        self.current_transform = transform
        self.current_tile_name = tile_name
        self.current_out_shape = out_shape
        self.current_tile = tile
        self.current_bounds = bounds

        # Save the raster to file if a path is provided
        if save_path is not None:
            self.save_raster(save_path)

        return final_raster

    def save_raster(self, output_path):
        """
        Saves the rasterized mask to a file.

        Args:
            output_path (str): Path to save the raster file.
        """
        if self.current_mask is None or self.current_transform is None or self.current_tile_name is None:
            raise ValueError("No raster data available. Run 'raster_tile()' first.")
        
        output_file = os.path.join(output_path, f'mask_raster_{self.current_tile_name}.tif')
        # Save raster to file using rasterio

        meta = {
                "tile_name": self.current_tile_name,
                "tile": self.current_tile,
                "label_mapping": label_mapping,
                "source": "Generated by RasterizeMasks: github...",
                "author": "Jonathan Alis",
            }
        print(f"Saving as {output_file}")
        with rasterio.open(
            output_file, 'w',
            driver='GTiff',
            height=self.current_out_shape[0],
            width=self.current_out_shape[1],
            count=1,
            dtype=self.current_mask.dtype,
            crs=self.current_tile.crs,
            transform=self.current_transform,
            compress='DEFLATE',
            predictor=1,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=254
        ) as dst:
            dst.write(self.current_mask, 1)
            dst.update_tags(**meta)
    
    def open_raster(self, raster_path):
        with rasterio.open(raster_path) as src:
            # Read the raster data
            raster_data = src.read(1)  # Read the first band (assumes single-band raster)
            
            # Access metadata
            metadata = src.meta  # Complete metadata dictionary
            transform = src.transform  # GeoTransform matrix
            crs = src.crs  # Coordinate Reference System
            bounds = src.bounds  # Bounding box of the raster

            self.current_mask = raster_data
            self.current_transform = src.transform
            self.current_tile_name = src.meta.tile_name
            self.current_out_shape = raster_data.shape
            self.current_tile = src.meta.tile
            self.current_bounds = src.bounds

            return raster_data
        

    def plot(self):
        if self.current_mask is None or self.current_transform is None or self.current_tile_name is None:
            raise ValueError("No raster data available. Run 'raster_tile()' first.")
        plot_mask(self.current_mask, title=f"Máscara do tile {self.current_tile_name}")



def plot_mask(mask, title="Área urbanizada", mapping = 'tipo e densidade'):
    """Plot a given mask."""

    if mapping == 'tipo e densidade':
        value_to_label = value_label_mapping
    elif mapping == 'tipo':
        value_to_label = type_mapping
    elif mapping == 'densidade':
        value_to_label = density_mapping
        

    # Create a unique label-to-color mapping
    unique_labels = list(set(value_to_label.values()))
    label_to_color = {label: plt.cm.tab10(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    # Map the image values to the corresponding label colors
    value_to_color = {value: label_to_color[label] for value, label in value_to_label.items()}
    colored_image = np.zeros((*mask.shape, 3))  # Create an RGB image
    for value, color in value_to_color.items():
        colored_image[mask == value] = color[:3]  # Assign RGB values (ignore alpha)

    # Plot the image
    plt.figure(figsize=(15, 15))
    plt.imshow(colored_image)
    plt.title(title)
    plt.axis("off")  # Hide axes if unnecessary

    # Create a legend
    legend_elements = [
        Patch(facecolor=color, label=label)
        for label, color in label_to_color.items()
    ]

    plt.legend(
    handles=legend_elements,
    loc="upper left",  # Position inside the plot
    title="Legenda",
    bbox_to_anchor=(1.05, 1),  # Move the legend box outside the plot (right side)
    borderaxespad=0.2,
    fontsize=10,
    fancybox=True,
    shadow=True
    )
    plt.show()


