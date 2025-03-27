import argparse
import pipeline.download_pipeline as download
from data.preprocess import preprocess_data
from model.train_model import train_model
from model.evaluate import evaluate_model
from training.postprocess import postprocess_results

def main():
    parser = argparse.ArgumentParser(description="Executa etapas específicas do pipeline.")
    parser.add_argument('--config', type=str, required=True, help="Caminho para o arquivo de configuração.")
    parser.add_argument('--step', type=str, choices=['download', 'preprocess', 'train', 'evaluate', 'postprocess'], required=True, help="Etapa a ser executada.")
    
    args = parser.parse_args()

    if args.step == 'download':
        download.pipeline(args.config)
    elif args.step == 'preprocess':
        preprocess_data(args.config)
    elif args.step == 'train':
        train_model(args.config)
    elif args.step == 'finetune':
        train_model(args.config)
    elif args.step == 'evaluate':
        evaluate_model(args.config)
    elif args.step == 'reconstruct':
        postprocess_results(args.config)

if __name__ == "__main__":
    main()