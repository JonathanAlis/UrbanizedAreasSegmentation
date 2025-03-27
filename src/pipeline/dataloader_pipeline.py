import rasterio
import os
import re
import numpy as np
import ast
import argparse
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess_data import yaml_filename, SubtileDataset, train_val_test_stratify


def pipeline(config_file=None):
    
    working_dir = os.path.abspath('.')
    if not config_file:
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'dataloader_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)

    params, tiles = parse_file(config_file)
    print(params)
    print(tiles)

    num_channels = int(params['num_channels'])
    num_subtiles = int(params['num_subtiles'])
    classes_mode = params['classes_mode']

    if classes_mode == 'type':
        num_classes = 5
    elif classes_mode == '4types':
        num_classes = 4
    elif classes_mode == 'density':
        num_classes = 4
    elif classes_mode == 'binary':
        num_classes = 2
    elif classes_mode == 'all':
        num_classes = 9
    # train test val split com estratificação
    train_files, val_files, test_files = train_val_test_stratify(tiles, 
                                                                  num_subtiles,
                                                                    train_size = 0.6, 
                                                                    val_size = 0.2, 
                                                                    stratify_by = classes_mode,
                                                                    subfolder=f'q_{num_channels}ch', 
                                                                    working_dir = working_dir)


    
    patch_size = int(params['patch_size'])
    stride = int(params['stride'])
    dynamic_sampling = params['dynamic_sampling'].lower() == 'true'
    data_augmentation = params['data_augmentation'].lower() == 'true'
    batch_size = int(params['batch_size'])
    
    #yaml_filename = yaml_filename(num_subtiles, tiles, classes_mode)
    #print('Trying to read from:', yaml_filename)
    train_dataset = SubtileDataset(train_files, 
                                    set = 'train_files',
                                    patch_size=patch_size, 
                                    stride=stride,
                                    classes_mode=classes_mode,
                                    #channels_subset= indices,
                                    dynamic_sampling = dynamic_sampling,
                                    data_augmentation = data_augmentation,
                                    working_dir = working_dir
                                    )
    
    val_dataset = SubtileDataset(val_files, 
                                    set = 'val_files',
                                    patch_size=patch_size, 
                                    stride=stride,
                                    classes_mode=classes_mode,
                                    #channels_subset= indices,
                                    dynamic_sampling = False,
                                    data_augmentation = False,
                                    working_dir = working_dir
                                    )
    
    test_dataset = SubtileDataset(test_files, 
                                    set = 'test_files',
                                    patch_size=patch_size, 
                                    stride=stride,
                                    classes_mode=classes_mode,
                                    #channels_subset= indices,
                                     dynamic_sampling = False,
                                    data_augmentation = False,
                                    working_dir = working_dir 
                                    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    output = result_dict = {"train_loader": train_loader,
                            "val_loader": val_loader,
                            "test_loader": test_loader,
                            "dynamic_sampling": dynamic_sampling,
                            "data_augmentation": data_augmentation,
                            "batch_size": batch_size,
                            "num_channels": num_channels,
                            "num_subtiles": num_subtiles,
                            "classes_mode": classes_mode,
                            "num_classes": num_classes,
                            "num_tiles": len(tiles)  # Calcula o comprimento da lista tiles
                        }

    return output


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