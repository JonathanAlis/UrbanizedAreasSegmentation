import rasterio
import os
import re
import numpy as np
import ast
import argparse
from torch.utils.data import Dataset, DataLoader
import torch

import src.training.train_model as train
import src.models.unets as unets
import src.models.hrnets as hrnets
import src.data.preprocess_data as data
import src.data.view as view
import src.training.post_processing as post

def pipeline(config_file=None):
    
    working_dir = os.path.abspath('.')
    if not config_file:
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'tile_assembly_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)

    params, tiles = parse_file(config_file)
    print(params)

    #obtém parametro pelo nome do modelo
    model_name = params['model']
    divided_params = model_name.split('-')
    patch_size = int(divided_params[1])
    stride = int(params['stride']) #patch_size-32
    edge_removal = int(params['edge_removal'])
    batch_size = int(params['batch_size'])
    classes_mode = divided_params[2]
    apply_cleaning = params['apply_cleaning'].lower() == 'true'
    print(model_name)
    num_channels = int(re.search(r'-([0-9]+)ch-', model_name).group(1))
    if classes_mode == '4types':
        num_classes = 4
    num_subtiles = params['num_subtiles']

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
        raise(ValueError(f'Modelo {model_name} não foi treinado'))
    ckp_file = os.path.join(working_dir, 'models', model_name+'.pth')
    checkpoint = torch.load(ckp_file, weights_only=False)
    model.load_state_dict(checkpoint['best_model_state_dict'])
    
    for tile_id in tiles:
        # --------------- opening files -----------------
        folder = os.path.join(working_dir,f"data/processed/S2-16D_V2_{tile_id}/{num_subtiles}x{num_subtiles}_subtiles/q_{num_channels}ch")
        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files if f.endswith('.tif')]

        # --------------- creating a dataloader -----------------

        tile_dataset = data.SubtileDataset(files, 
                                        num_subtiles = 6,
                                        classes_mode=classes_mode,
                                        patch_size=patch_size, 
                                        stride=stride, #//2, 
                                        dynamic_sampling = False,
                                        data_augmentation = False,
                                        return_imgidx = True,
                                        working_dir=working_dir)
        dataloader = DataLoader(tile_dataset, batch_size=batch_size, shuffle=False)


        ### ---------------- assemblyng tile
        # aqui cria o objeto de reconstrucao do tile
        tile = post.ReconstructTile(patch_size = patch_size, stride = stride, edge_removal=edge_removal, 
                                    num_classes=num_classes, num_channels=num_channels, tile_id=tile_id)


        print('Inferindo patches...')        
        runner = train.EpochRunner('test', model, dataloader, num_classes=num_classes, 
                                    simulated_batch_size = dataloader.batch_size, device = device)  
        for image, label, logits, pred, x, y, f, in runner.run_generator(show_pred = 0):
            tile.add_batch(x, y, f, logits, pred, label, image)
        print('Montando patches em tile...')
        tile.set_pred()   
        #print('Pos processamento...')
        #labels, pred_patch, clean_pred, clean_noholes, clean_noholes_2, noholes, noholes2, rules = tile.post_process(0,0)

        print('Salvando...')        
        tile.save_pred(working_dir=working_dir, folder_name = model_name.replace('.pth', ''))
        loss, CE, dice, report, acc, cm = runner.get_metrics()


        print(f'Test Loss: {loss}, {CE}, {dice}')
        print(f'Test Accuracy: {acc}')
        #print(f'Test confusion matrix:')
        view.plot_single_confusion_matrix(cm)
        print(report)

        #### ----------------- Cleaning tile
        if apply_cleaning:
            print('Limpeza')
            tile.post_process_tile()
            tile.save_cleaning(working_dir=working_dir, folder_name = model_name.replace('.pth', ''))
            del tile
            import gc
            gc.collect()




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