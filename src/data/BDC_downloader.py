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
    #query={"bdc:tile": {"eq": tile}}
    items = list(item_search.item_collection())
    items = [item for item in items if tile in str(item.assets['B01'])]

    return items


def get_tile(tile:str = '016009',
              collections=['S2-16D-2'], 
              datetime='2019-01-01/2019-12-31'
              ):
    """
    Get list of items from a tile, from BDC
    """
    parameters = dict(access_token='')

    service = pystac_client.Client.open(
        'https://data.inpe.br/bdc/stac/v1/', 
        parameters=parameters
    )
    
    # Construct the query to filter by tile in the B01 asset path
    query={"bdc:tile": {"eq": tile}}
    
    item_search = service.search(
        datetime=datetime,
        collections=collections,
        query=query  # Add the query here
    )
    return list(item_search.item_collection())


def get_tiles(tiles:str|list = '016009',
              collections=['S2-16D-2'], 
              datetime='2019-01-01/2019-12-31'
              ):
    
    """
    Get items from multiple tiles
    """
    if not isinstance(tiles, list):
        tiles = [tiles]
    items = []    
    for tile in tiles:
        tile_items = get_tile(tile, collections=collections,datetime=datetime)
        
        items+=tile_items
    return items  # No need for post-filtering



def get_max_coverage_items(tile, N = 4, threshold = 98, collections=['S2-16D-2'], datetime='2019-01-01/2019-12-31'):
    """
    Get N items that cover at least theshold (%) area.
    
    """
    print(tile)
    items = get_tile(tile, collections=collections, datetime=datetime)

    # at this point, tile_items contains al items of a tile    
    
    #now, counting valid pixels (SCL channel between 4 and 6) for each date
    counts = []
    for i, it in enumerate(items):
        #if not isinstance(it, dict):
        #    it = it.to_dict()
        
        uri = it.assets['SCL'].href
        #it['assets']['SCL']['href']
        with rasterio.open(uri) as src:
            data = src.read()
        arr = np.where((data >= 4) & (data <= 6), 1, 0)  # Binary mask (valid coverage)
        valid_pixel_count = np.sum(arr) / arr.size  # Count number of 1s in arr
        if valid_pixel_count*100 > threshold:
            return [it] #if this tile already have enough valid pixels, return it 
        counts.append(valid_pixel_count)  # Store it

        
    ordered_idx = np.argsort(counts)
    ordered_idx = ordered_idx[::-1]
    #items of a tile, ordered by highest to lowest % of valid pixels
    ordered_items = [items[i] for i in ordered_idx]

    print(ordered_items)


    selected_items = []
    sum_overlap = None  # Track total covered area
    
    for i, it in enumerate(ordered_items):
        selected_items.append(it)
        uri = it.assets['SCL'].href
        # Read raster
        with rasterio.open(uri) as src:
            data = src.read()
        arr = np.where((data >= 4) & (data <= 6), 1, 0)  # Binary mask (valid coverage)
        
        if sum_overlap is None:
            sum_overlap = np.zeros_like(arr, dtype=bool)  # Initialize sum array
        else:
            sum_overlap = np.logical_or(sum_overlap, arr)  # Update cumulative mask


        # Compute new overlap if this array is added
        new_overlap = np.sum(np.logical_or(sum_overlap, arr))
        
        print('Sobreposição:', new_overlap)
        #if overlap of current tiles pass threshold, return it
        if 100 * new_overlap / arr.size > threshold:
            return selected_items        
        if len(selected_items) >= N:
            return selected_items
    return selected_items


#def get_
def download_and_save_item(item, save_dir = 'data/raw', compress='lzw'):
    """Reads a raster from a URI and optionally saves it to a file.

    Args:
        item: item object, must contain a uri to download.
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
        
 


