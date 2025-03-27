import rasterio
import os
import re
import numpy as np
import ast
import argparse

import src.data.mask_processing as mask_processing



def pipeline(config_file=None):
    working_dir = os.path.abspath('.')
    if not config_file:
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'masks_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)

    params, tiles = parse_file(config_file)

    urban_shp_path = os.path.join(working_dir, params['shapefile_path'])
    bdc_grid_path = os.path.join(working_dir, params['bdc_grid_path'])
    save_dir = os.path.join(working_dir, params['save_dir'])

    custom_crs_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["Unknown based on GRS80 ellipsoid",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",-12],PARAMETER["longitude_of_center",-54],PARAMETER["standard_parallel_1",-2],PARAMETER["standard_parallel_2",-22],PARAMETER["false_easting",5000000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    rasterizer = mask_processing.RasterizeMasks(urban_shp_path, bdc_grid_path, custom_crs_wkt)
    for tile in tiles:
        mask = rasterizer.raster_tile(tile, save_path=os.path.join(working_dir, save_dir))
        

def parse_file(file):

    # Expressões regulares para identificar valores
    regex_param = re.compile(r"(\w+)\s*=\s*(.+)")  # Captura parâmetros no formato chave = valor
    regex_digits = re.compile(r"\b\d{6}\b")  # Captura números de exatamente 6 dígitos

    # Dicionário para armazenar os parâmetros
    parametros = {}
    # Lista para armazenar os números de 6 dígitos
    tiles = []

    # Ler o arquivo linha por linha
    with open(file, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            
            if linha.startswith("#") or not linha:
                continue
            # Verifica se a linha contém um parâmetro no formato chave=valor
            match_param = regex_param.match(linha)
            if match_param:
                chave, valor = match_param.groups()
                parametros[chave] = valor
                continue

            # Verifica se a linha contém um número de 6 dígitos
            match_digits = regex_digits.search(linha)
            if match_digits:
                tiles.append(str(match_digits.group()))

                # Exibir os resultados
    print("Parâmetros:", parametros)
    print("Tiles:", tiles)
    return parametros, tiles

if __name__ == "__main__":
    pipeline()