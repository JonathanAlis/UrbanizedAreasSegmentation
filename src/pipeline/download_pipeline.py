import os
import re
import argparse

from src.data.BDC_downloader import get_tiles, get_max_coverage_items, download_and_save_item


def pipeline(config_file=None):
    working_dir = os.path.abspath('.')
    if not config_file:
        parser = argparse.ArgumentParser(description="Arquivo de configuração")
        parser.add_argument("config_file", nargs="?", default=None, help="Arquivo de configuração")
        args = parser.parse_args()
        
        if args.config_file is None:
            config_file = os.path.join(working_dir, 'data', 'config', 'download_params.txt')
        else:
            config_file = args.config_file
        print('Config file:', config_file)    

    params, tiles = parse_file(config_file)
    save_dir = os.path.join(working_dir,params['save_dir'])

    for tile in tiles:
        items  = get_max_coverage_items(tile, 
                                        N = int(params['N']), 
                                        threshold = float(params['threshold']), 
                                        collections=[params['collections']], 
                                        datetime=params['datetime'])
        for item in items:
            download_and_save_item(item, save_dir = save_dir)



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