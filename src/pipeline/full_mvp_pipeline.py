import os

import src.pipeline.download_pipeline as download
import src.pipeline.preprocessing_pipeline as preprocessing
import src.pipeline.masks_pipeline as masks
import src.pipeline.dataloader_pipeline as dataloader
import src.pipeline.training_pipeline as training
import src.pipeline.tile_assembly_pipeline as tile_assembly

def print_message(etapa):
    print(f'______________________{etapa.upper()}______________________')

if __name__ == "__main__":
    working_dir = os.path.abspath('.')

    preset_folder = os.path.join(working_dir, 'data/config/presets/')
    
    print_message('Download')
    download.pipeline(preset_folder+'download_1_tile.txt')
    
    print_message('Pré-processamento')
    preprocessing.pipeline(preset_folder+'preprocess_preset_0.txt')
    
    print_message('Máscaras')
    masks.pipeline(preset_folder+'masks_1_tile.txt')
    
    print_message('Treinamento')
    training.pipeline(preset_folder+'training_preset_0.txt')
    
    print_message('Montagem do tile')
    tile_assembly.pipeline(preset_folder+'tile_assembly_preset_0.txt')

