import argparse
import yaml
import os
from typing import List
from pathlib import Path
from data_handler import S3_DataHandler
from helpers import get_logger
import random
import glob
import shutil


class ImgS3Handler: 

    @staticmethod
    def fetch_s3_svo_images(s3_uri: str, aws_dir: str):
        logger = get_logger("ImgS3Handler")
        
        if os.path.exists(aws_dir) and os.listdir(aws_dir):
            logger.warning("───────────────────────────────")
            logger.warning(f"AWS directory already exists: {aws_dir}")
            logger.warning("───────────────────────────────")
            return
        
        S3_DataHandler.download_s3_folder(s3_uri, aws_dir)

    @staticmethod
    def get_valid_leaf_folders(leaf_folders: List[str], folders_to_sample: List[str]) -> List[str]:
         
        valid_leaf_folders = set()
        for folder in leaf_folders:
            for keys in folders_to_sample:
                if keys in folder:
                    valid_leaf_folders.add(Path(folder).parent)
                    break

        valid_leaf_folders = list(valid_leaf_folders)
        return valid_leaf_folders

    @staticmethod
    def sample_img_pairs_from_folder(base_dir: str, folder_path: str, num_images_to_sample: int) -> List[List[str]]:
        """
        Samples pairs of left and right images from a specified folder.

        Args:
            base_dir (str): The base directory to search within.
            folder_path (str): The path to the folder containing the images, relative to base_dir.
            num_images_to_sample (int): The number of image pairs to sample.

        Returns:
            List[List[str]]: A list of image pairs, where each pair is a list containing the 
                             absolute paths to the left and right images as strings.  Returns
                             an empty list if no pairs are found.
        """
        left_images = list(Path(base_dir / folder_path).rglob("*_left.jpg"))

        logger = get_logger("ImgS3Handler")

        logger.info("───────────────────────────────")
        logger.info(f"folder_path: {folder_path}")
        logger.info(f"len(left_images): {len(left_images)}")
        logger.info("───────────────────────────────")

        img_pairs = []
        for left_img in left_images:
            right_img = Path(str(left_img).replace("_left.jpg", "_right.jpg"))
            if right_img.exists():
                img_pairs.append((left_img, right_img))

        num_samples = min(num_images_to_sample, len(img_pairs))
        sampled_pairs = random.sample(img_pairs, num_samples)

        return [(str(left), str(right)) for left, right in sampled_pairs]

    @staticmethod
    def generate_GT_train_test(base_dir: str, folders_to_sample: List[str], num_images_to_sample: int):
        
        logger = get_logger("ImgS3Handler")

        GT_test = os.path.join(base_dir, "GT-test")
        os.makedirs(GT_test, exist_ok=True)

        leaf_folders = S3_DataHandler._get_leaf_folders(base_dir)

        logger.warning("───────────────────────────────")
        logger.warning(f"leaf_folders: {leaf_folders[0]}")
        logger.warning("───────────────────────────────")

        

        logger.info("───────────────────────────────")  
        logger.info(f"len(leaf_folders): {len(leaf_folders)}")
        logger.info("───────────────────────────────")
       
        valid_leaf_folders = ImgS3Handler.get_valid_leaf_folders(leaf_folders, folders_to_sample)

        logger.info("───────────────────────────────")
        logger.info(f"len(valid_leaf_folders): {len(valid_leaf_folders)}")
        for folder in valid_leaf_folders:
            logger.info(f"-{folder}")
        logger.info("───────────────────────────────")

        for valid_leaf_folder in valid_leaf_folders:
            sampled_pairs = ImgS3Handler.sample_img_pairs_from_folder(base_dir=base_dir, folder_path=valid_leaf_folder, num_images_to_sample=num_images_to_sample)

            logger.info("───────────────────────────────")
            logger.info(f"len(sampled_pairs): {len(sampled_pairs)}")
            logger.info("───────────────────────────────")

            for img_pair in sampled_pairs:
                logger.info(f"-{img_pair}")
            
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    s3_config = config['s3_data_handler']
    base_dir = s3_config['base_dir']
    aws_dir = os.path.join(base_dir, "GT-aws")

    ImgS3Handler.fetch_s3_svo_images(
        s3_uri=s3_config['s3_uri'],
        aws_dir=aws_dir
    )

    ImgS3Handler.generate_GT_train_test(
        base_dir=base_dir,
        folders_to_sample=s3_config['folders_to_sample'],
        num_images_to_sample=s3_config['num_images_to_sample']
    )

if __name__ == "__main__":
    main()