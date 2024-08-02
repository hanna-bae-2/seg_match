import os 
import os.path as osp 
import argparse 
from mmseg.apis import inference_segmentor, init_segmentor 
import torch 
import mmcv 
import cv2 
import numpy as np 
from tqdm import tqdm 
from glob import glob 
from nets.semseg.utils import get_semantic_dict 


def get_conf_dict():
    maps = {
        0 : 1, # Volatile 
        1 : 1, # Long-term 
        2 : 0.2, # Dynamic 
        3 : 0.5 # Short-term 
    }
    return maps 

def create_segment_mask(img_path, model, semantic_dict, conf_dict, seg_path):
    # directory pass the images 
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(512, 512))

    # segment by the pretrained model
    result = inference_segmentor(model=model, img=img)
    seg_mask = result[0].astype(np.uint8) +1
    
    # semantic dict (categorized)
    categorized_mask = np.zeros_like(seg_mask)

    for label, category in semantic_dict.items():
        categorized_mask[seg_mask == label] = category

    # confidence mask (stability value)
    conf_mask = np.zeros_like(categorized_mask)
    
    for label, confidence in conf_dict.items():
        conf_mask[categorized_mask == label] = confidence
    
    conf_ext = np.expand_dims(conf_mask, axis=-1)
    mask_img = img * conf_ext 
    filename = osp.basename(img_path)
    path = f'mask_result/{filename}'
    output_path = osp.join(seg_path, path)
    cv2.imwrite(output_path, mask_img)
    print(f'create img {output_path}')


def process_folder(img_dir):
    seg_path = os.getcwd() 
    config_file = 'nets/semseg/configs/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k.py'
    checkpoint_file = 'weights/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # init segmentation model 
    model = init_segmentor(osp.join(seg_path, config_file), osp.join(seg_path, checkpoint_file), device = device)
    
    # get semantic and confidence 
    semantic_dict = get_semantic_dict()
    conf_dict = get_conf_dict()

    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith(('.png','.jpg', '.jpeg')):
            img_path = osp.join(img_dir, file_name)
            create_segment_mask(img_path, model, semantic_dict, conf_dict, seg_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create mask images')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to input image')
    
    args = parser.parse_args()
    process_folder(args.img_dir)
