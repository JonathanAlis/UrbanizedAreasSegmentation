import rasterio
import os
import re
import numpy as np
import ast
import argparse
from torch.utils.data import Dataset, DataLoader
import torch

import src.training.train_model as train
import src.pipeline.dataloader_pipeline as dataloader_pipeline
import src.models.unets as unets
import src.models.hrnets as hrnets

def pipeline(config_file=None):
    
    working_dir = os.path.abspath('.')
    if not config_file:
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'training_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)

    params, tiles = parse_file(config_file)
    print(params)
    num_epochs = int(params['num_epochs'])
    patience = int(params['patience'])

    dataloader_config = params['dataloader_config']
    dataloader_config_full_path = os.path.join(working_dir, 'data/config',dataloader_config)
    print('dataloader_config_full_path:', dataloader_config_full_path)
    dataloader_params = dataloader_pipeline.pipeline(dataloader_config_full_path)

    train_loader = dataloader_params["train_loader"]
    val_loader = dataloader_params["val_loader"]
    test_loader = dataloader_params["test_loader"]
    dynamic_sampling = dataloader_params["dynamic_sampling"]
    data_augmentation = dataloader_params["data_augmentation"]
    batch_size = dataloader_params["batch_size"]
    num_channels = dataloader_params["num_channels"]
    num_subtiles = dataloader_params["num_subtiles"]
    classes_mode = dataloader_params["classes_mode"]
    num_classes = dataloader_params["num_classes"]    
    num_tiles = dataloader_params["num_tiles"]
    

    ### training
    models = ast.literal_eval(params['models'])
    loss = params['loss']
    weighted_loss = params['weighted_loss'].lower() == 'true'

    for model in models:
        model_name = model
        
        if dynamic_sampling:
            print(model_name)
            model_name+='-DS'
            print(model_name)
        if data_augmentation:
            model_name+='-DA'
        model_name = model_name+f'-{loss}'
        if weighted_loss:
            model_name+='W'
        model_name+=f"-{num_channels}ch" #nof channels
        model_name+=f"-{num_tiles}tt" #nof train tiles

        print('--------------------')
        print('Training', model_name)
        # define quais indices

        if weighted_loss:                   
            class_counts, per_image = train_loader.dataset.count_classes()
            class_weights = 1.0 / class_counts  # Inverse of class frequencies
            class_weights = class_weights / torch.sum(class_weights)  # Normalize
        else:    
            class_weights = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if model_name.startswith('UNetSmall-'):
            model = unets.UNetSmall(in_channels=num_channels, out_channels=num_classes).to(device) 
        elif model_name.startswith('UNet-'):
            model = unets.UNet(in_channels=num_channels, out_channels=num_classes).to(device) 
        elif model_name.startswith('UNetResNet34-'):
            model = unets.UNetResNet34(in_channels=num_channels, out_channels=num_classes).to(device) 
        elif model_name.startswith('UNetEfficientNetB0-'):
            model = unets.UNetEfficientNetB0(in_channels=num_channels, out_channels=num_classes).to(device) 
        elif model_name.startswith('UNetConvNext-'):
            model = unets.UNetConvNext (in_channels=num_channels, out_channels=num_classes).to(device) 
        elif model_name.startswith('HRNetW18'):
            model = hrnets.HRNetSegmentation(in_channels= num_channels, num_classes=num_classes, backbone="hrnet_w18_small", pretrained=True,).to(device)
        elif model_name.startswith('HRNetW32'):
            model = hrnets.HRNetSegmentation(in_channels= num_channels, num_classes=num_classes, backbone="hrnet_w32", pretrained=True,).to(device)
        elif model_name.startswith('HRNetW48'):
            model = hrnets.HRNetSegmentation(in_channels= num_channels, num_classes=num_classes, backbone="hrnet_w48", pretrained=True,).to(device)
        else:
            print(f'Modelo {model_name} não está no param grid. Pulando...')
            continue

        print(model_name, '############################')
        train.train_model(model, 
                            train_loader, 
                            val_loader, 
                            epochs=num_epochs, 
                            loss_mode = loss,
                            device = device,
                            num_classes = num_classes, 
                            simulated_batch_size = batch_size, #model_params['batch_size'] ,
                            patience = patience,
                            weights = class_weights,
                            show_batches = 0, 
                            working_dir = working_dir,
                            save_to = model_name+'.pth')

            

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