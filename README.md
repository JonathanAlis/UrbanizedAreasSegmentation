# Projeto de Segmentação de Áreas Urbanizadas (AU)

Este o repositório do projeto de segmentação de AU, em colaboração com o IBGE.
Aqui, o objetivo é utilizar modelos convolucionais de Deep Learning para a segmentação de áreas urbanizadas a partir de imagens de satélite.


## Table of Contents

- [Estrutura](#project-structure)
- [Setup](#setup)
- [Uso](#usage)

## Estrutura de arquivos

O projeto é estruturado da seguinte forma:
```
/
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── preprocess_data.py
│   │   └── ...
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_training.py
│   │   ├── model_inference.py
│   │   └── ...
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── ...
│   └── pipeline/
│       └── ...
├── data/
│   ├── raw/
│   │   └── ...
│   ├── processed/
│   │   └── ...
│   └── config
│   │   └──...
├── models/
│   ├── unet/
│   │   └── ...
│   └── hrnet/
│   
├── notebooks_produto2/
│   └── ...
├── notebooks_produto3/
│   └── ...
├── configs/
│   └── ...
├── experimental_results/
│   └── ...
├── requirements.txt
└── README.md

```
### Key Directories and Files

- **src/**: Contém os códigos fintes organizados em modulos
  - **data_processing/**: Aquisição, organização e processamento de dados
  - **models/**: Modelos UNet e HRNet utilizados para treino e teste
  - **pipeline/**: Contém os scripts para rodar os módulos básicos, como o download, preparação, treino, etc
  - **train/**: Scripts de treinamento e pós processamento
- **data/**: Contém dados brutos e processados.
- **models/**: Pasta que contém os checkpoints dos modelos salvos
- **notebooks_produto2/**: Jupyter notebooks contendo os pasos e experimentos explorados no produto 2
- **notebooks_produto3/**: Jupyter notebooks contendo os pasos e experimentos explorados no produto 3
- **requirements.txt**: Lista de dependências
- **README.md**: Este arquivo.

## Setup

### 1. Clone o repositório

```bash
git clone https://github.com/JonathanAlis/UrbanizedAreasSegmentation.git
cd UrbanizedAreasSegmentation
```

### 2. Set Up de um Virtual Environment
Use venv (Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Instalação de dependências
```bash
pip install -r requirements.txt
```

### 4. Baixe e deszipe os arquivos shapefile de máscaras das áreas urbanizadas

https://www.ibge.gov.br/geociencias/cartas-e-mapas/redes-geograficas/15789-areas-urbanizadas.html?=&t=downloads

O arquivo deve ficar neste caminho:

/data/masks/AreasUrbanizadas2019_Brasil/AU_2022_AreasUrbanizadas2019_Brasil.shp


### 5. Adjuste os sys.path nos Notebooks
Nos notebooks, inclua a raiz do projeto  ao sys.path para fazer os imports:

```python
import os
import sys
sys.path.append(os.path.abspath('..'))
```

## Uso

### 1. Excecução do pipeline completo (MVP)
```bash
python -m src.pipeline.full_mvp_pipeline
```

### 2. Gera predições de tiles
```bash
python -m src.pipeline.tile_assembly_pipeline ./data/config/presets/tile_assembly_preset1.txt
```
