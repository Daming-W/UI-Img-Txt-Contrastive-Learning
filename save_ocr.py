import pytesseract
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import csv 
from tqdm import tqdm
import glob
import os
from google.cloud import vision

def ocr2save(args):

    with open(args.csv_file_path, 'w') as csvfile:
        csvfile_writer = csv.writer(csvfile)

        if args.dataset_name=='rico':
            img_type = '.jpg'
        else:
            img_type = '.png'

        for img_path in glob.glob(os.path.join(args.img_folder_path,'*'+img_type)):
            with open(img_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations
            for text in texts:
                print('\n"{}"'.format(text.description))
            #csvfile_writer.writerow([img_path,string])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='rico', help="dataset name rico or clarity")  
parser.add_argument("--img_folder_path", type=str, default="/root/autodl-tmp/RICO/combined/")
 
parser.add_argument("--csv_file_path", type=str, default='/root/autodl-tmp/RICO/ocr.csv', help="target csv file path for writing in ocr result")   
args = parser.parse_args()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/root/autodl-tmp/abiding-state-380503-ff79ff173b6f.json'

client = vision.ImageAnnotatorClient()

if args.dataset_name == 'clarity':
    args.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
    args.csv_file_path = '/root/autodl-nas/Clarity-Data/ocr.csv'

elif args.dataset_name == 'rico':
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/ocr.csv'

ocr2save(args)
