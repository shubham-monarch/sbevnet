import argparse
import yaml
import os

from data_handler import S3_DataHandler

class ImgS3Handler: 

    @staticmethod
    def download_s3_data(s3_uri: str, aws_dir: str):
        S3_DataHandler.download_s3_folder(s3_uri, aws_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    s3_config = config['s3_data_handler']
    base_dir = s3_config['base_dir']
    aws_dir = os.path.join(base_dir, "GT-aws")

    ImgS3Handler.download_s3_data(
        s3_uri=s3_config['s3_uri'],
        aws_dir=aws_dir
    )

if __name__ == "__main__":
    main()