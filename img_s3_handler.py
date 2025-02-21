import argparse
import yaml
import os
from typing import List, Any, Dict
from pathlib import Path
from helpers import get_logger
import random
import glob
import shutil
import cv2
import numpy as np


from data_handler import S3_DataHandler, ModelDataHandler
from svo_eval import EvalSVO


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
    def generate_sample_img_pairs(base_dir: str, folders_to_sample: List[str], num_images_to_sample: int):
        
        leaf_folders = S3_DataHandler._get_leaf_folders(base_dir)
        valid_leaf_folders = ImgS3Handler.get_valid_leaf_folders(leaf_folders, folders_to_sample)

        img_pairs_to_process = []
        for folder in valid_leaf_folders:
            sampled_pairs = ImgS3Handler.sample_img_pairs_from_folder(base_dir=base_dir, 
                                                                      folder_path=folder, 
                                                                      num_images_to_sample=num_images_to_sample)    
            img_pairs_to_process.extend(sampled_pairs)

        return img_pairs_to_process

    @staticmethod
    def write_img_data_to_folder(output_dir: str, left_img: np.ndarray, right_img: np.ndarray, 
                                ipm_left_img: np.ndarray, H_img_to_bev: np.ndarray, filename: str = None):
        """
        Writes image data and homography matrix to a numbered subfolder.
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, "left.jpg"), left_img)
        cv2.imwrite(os.path.join(output_dir, "right.jpg"), right_img)
        cv2.imwrite(os.path.join(output_dir, "ipm-left.png"), ipm_left_img)
        np.save(os.path.join(output_dir, "H_img_to_bev.npy"), H_img_to_bev)

        if filename is not None:
            with open(os.path.join(output_dir, "file_name.txt"), "w") as f:
                f.write(filename)

    @staticmethod
    def write_img_data_to_GT(GT_dir: str, img_pairs_to_process: List[List[str]], config_path: str):
        os.makedirs(GT_dir, exist_ok=True)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        K = np.array(config['K']).astype(np.float32)
        bev_size = int(config['bev_size'])  # ensure bev_size is integer
        ground_height = float(config['ground_height'])  # ensure ground_height is float

        # Adjust bev_region to match the coordinate system expected by H_img_to_bev
        bev_region = {
            'z_min': float(config['xmin']),  # forward direction
            'z_max': float(config['xmax']),
            'x_min': float(config['ymin']),  # lateral direction
            'x_max': float(config['ymax'])
        }

        for idx, img_pair in enumerate(img_pairs_to_process):
            left_img_path = img_pair[0]
            right_img_path = img_pair[1]

            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)


            # generate H_img_to_bev matrix
            (ref_height, ref_width) = (1080, 1920)
            H_img_to_bev = EvalSVO.H_img_to_bev(K, bev_region, bev_size, ground_height)
            ipm_m = np.concatenate([H_img_to_bev.flatten(), [ref_height, ref_width]])
            
            ipm_left_img = EvalSVO.generate_ipm_image(left_img, K, bev_region, bev_size, ground_height)
            
            dest_folder = os.path.join(GT_dir, f"{idx}")
            ImgS3Handler.write_img_data_to_folder(dest_folder, 
                                                  left_img, 
                                                  right_img, 
                                                  ipm_left_img, 
                                                  ipm_m, 
                                                  filename=str(Path(left_img_path).parent))

    @staticmethod
    def generate_GT_train_test(config_path: str):
        
        logger = get_logger("ImgS3Handler")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        s3_config = config['s3_data_handler']
        base_dir = s3_config['base_dir']
        folders_to_sample = s3_config['folders_to_sample']
        num_images_to_sample = s3_config['num_images_to_sample']

        GT_test = os.path.join(base_dir, "GT-test")
        os.makedirs(GT_test, exist_ok=True)
        assert not (os.path.exists(GT_test) and os.listdir(GT_test))
        
        img_pairs_to_process = ImgS3Handler.generate_sample_img_pairs(
            base_dir=base_dir,
            folders_to_sample=folders_to_sample[:2],
            num_images_to_sample=num_images_to_sample
        )

        ImgS3Handler.write_img_data_to_GT(
            GT_dir=GT_test,
            img_pairs_to_process=img_pairs_to_process,
            config_path=config_path)
    
    @staticmethod
    def generate_model_dataset(config_path: str):
        
        logger = get_logger("ImgS3Handler")
        
        with open(config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        s3_config = config['s3_data_handler']
        base_dir = s3_config['base_dir']
        aws_dir = os.path.join(base_dir, "GT-aws")


        ImgS3Handler.fetch_s3_svo_images(
            s3_uri=s3_config['s3_uri'],
            aws_dir=aws_dir
        )

        ImgS3Handler.generate_GT_train_test(config_path=config_path)

        GT_test_dir = os.path.join(base_dir, "GT-test")
        model_folder = os.path.join(base_dir, "model-dataset")

        ModelDataHandler._restructure_GT_folder(GT_test_dir, model_folder)
        ModelDataHandler._flip_masks(os.path.join(model_folder, 'seg-masks-mono'),\
                           os.path.join(model_folder, 'seg-masks-mono'))
        
        ModelDataHandler._populate_json(os.path.join(model_folder, 'dataset.json'), model_folder, model_folder, model_folder)


        

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help='Path to config file')
#     args = parser.parse_args()
    
#     # with open(args.config, 'r') as f:
#     #     config = yaml.safe_load(f)
    
#     # s3_config = config['s3_data_handler']
#     # base_dir = s3_config['base_dir']
#     # aws_dir = os.path.join(base_dir, "GT-aws")

#     # ImgS3Handler.fetch_s3_svo_images(
#     #     s3_uri=s3_config['s3_uri'],
#     #     aws_dir=aws_dir
#     # )

#     # ImgS3Handler.generate_GT_train_test(
#     #     base_dir=base_dir,
#     #     folders_to_sample=s3_config['folders_to_sample'],
#     #     num_images_to_sample=s3_config['num_images_to_sample']
#     # )

#     ImgS3Handler.generate_model_dataset(
#         config_path=args.config
#     )

# if __name__ == "__main__":
#     main()