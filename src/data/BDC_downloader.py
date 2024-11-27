import pystac_client
import numpy as np
import matplotlib.pyplot as plt
from rasterio.coords import BoundingBox
import rasterio
import os


def get_items_by_tile(tile = '032027',
                      collections=['S2-16D-2'], 
                      datetime='2019-01-01/2019-12-31',
                      bounding_box_filter = BoundingBox(left=-44.32729056461349, bottom=-20.80707257210646, right=-43.34879738591701, top=-19.823979268627337)
                      ):
    #-33.69111 to 2.81972 and longitude from -72.89583 to -34.80861
    parameters = dict(access_token='')

    service = pystac_client.Client.open('https://data.inpe.br/bdc/stac/v1/', parameters=parameters)
    item_search = service.search(bbox=bounding_box_filter,
                                datetime=datetime,
                                collections=collections)
    
    items = list(item_search.item_collection())
    items = [item for item in items if tile in str(item.assets['B01'])]

    return items


def download_and_save_item(item, save_dir = 'data/raw', compress='lzw'):
    """Reads a raster from a URI and optionally saves it to a file.

    Args:
        uri (str): The URI of the raster data.
        save_folder (str, optional): The path to the folder where the raster file will be saved. If None, the data is not saved.
        compress (str, optional): The compression method to use when saving. Default is 'lzw'.

    Returns:
        rasterio.Dataset: The raster data as a rasterio Dataset object.
    """

    info_dict = item.to_dict() 
    prefix = info_dict['id'] # 'S2-16D_V2_032027_20191219'
    collection, version, tile, date = prefix.split('_')
    print(f'collection: {collection}')
    print(f'version: {version}')
    print(f'tile: {tile}')
    print(f'date: {date}')
    
    
    full_save_dir = os.path.join(save_dir, collection+'_'+version+'_'+tile, date)
    uri = item.assets

    for k in item.assets.keys():
        uri = item.assets[k].href
        filename = os.path.basename(uri)
        if not os.path.exists(full_save_dir):
            os.makedirs(full_save_dir)
            print(f'creating dir {full_save_dir}')        
        save_path = os.path.join(full_save_dir, filename)
        read_ok = False
        if os.path.exists(save_path):
            try:
                with rasterio.open(save_path) as src:
                    data = src.read()
                    meta = src.meta
                    read_ok = True  
                    print(f'{save_path} already downloaded. Skipping.')                    
            except:
                print(f'Error reading {save_path}')
                if os.path.exists(save_path):
                    os.remove(save_path)

        if not read_ok:
            with rasterio.open(uri) as src:
                print(f'Downloading {uri}...')
                print(f'Saving to {save_path}.')
                
                data = src.read()
                meta = src.meta
                meta['compress'] = compress
                with rasterio.open(save_path, 'w', **meta) as dst:
                    dst.write(data)
                    print(filename,'saved')
        
 


