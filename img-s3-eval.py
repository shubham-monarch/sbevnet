import argparse

from img_s3_data_handler import ImgS3DataHandler
from evaluate import evaluate_sbevnet


class ImgS3Eval: 

    @staticmethod
    def generate_model_dataset(config_path: str):
        ImgS3DataHandler.generate_model_dataset(config_path=config_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    ImgS3Eval.generate_model_dataset(config_path=args.config)
    evaluate_sbevnet(config_path=args.config)

    ImgS3DataHandler.restructure_predictions_folder(config_path=args.config)

if __name__ == "__main__":
    main()