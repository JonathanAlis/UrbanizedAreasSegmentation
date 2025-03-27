import rasterio
import os
import re
import numpy as np
import ast
import argparse

from src.data.prepare_data import prepare_image, quantize, save


all_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

def pipeline(config_file=None):
    working_dir = os.path.abspath('.')
    if not config_file:
        working_dir = os.path.abspath('.')
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'preprocessing_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)

    params, tiles = parse_file(config_file)
    raw_dir = os.path.join(working_dir, params['raw_dir'])
    save_dir = os.path.join(working_dir, params['raw_dir'])
    tile_size = int(params['tile_size'])
    num_subtiles = int(params['num_subtiles'])
    subtile_size = tile_size // num_subtiles
    quantization_max = int(params['quantization_max'])

    channels = ast.literal_eval(params['channels_to_save'])
    print(channels)
    # se tiles estiver vazio
    if len(tiles) == 0:
        print(f'Tiles vazios, buscando em {raw_dir}')
        regex_digits = re.compile(r"\d{6}\b")
        raw_tiles_folders = os.listdir(raw_dir)
        for folder_name in raw_tiles_folders:
            match_digits = regex_digits.search(folder_name)
            #print(match_digits)
            if match_digits:
                tiles.append(str(match_digits.group()))
        print('Tiles encontrados:', tiles)
    # loop principal
    for tile in tiles:
        print(f'Processando Tile: {tile}')
        for x in range(0, tile_size, subtile_size):
            for y in range(0, tile_size, subtile_size):
                print(f'----Subtile na posição: ({x}, {y})')
                image = []
                for i, channel in enumerate(channels):
                    if channel in all_channels:
                        print(f'--------Banda: {channel}')

                        window = rasterio.windows.Window(x, y, subtile_size, subtile_size)
                        # prepara as imagens por subtiles
                        channel_image = prepare_image(tile=tile, channel=channel, window=window, working_dir=working_dir)
                        if quantization_max>0:
                            channel_image = quantize(channel_image, lower_fixed=0, higher_fixed=quantization_max)
                        image.append(channel_image)
                    
                image = np.stack(image, axis=0)
                save(image, x, y, tile, num_subtiles, working_dir = working_dir)
                #for k, chd in channels_dict.items(): 
                #    indices = [i for i, value in enumerate(chd) if value in channels_dict[12]]
                #    save(qimage[indices], x, y, tile, num_subtiles)
        


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