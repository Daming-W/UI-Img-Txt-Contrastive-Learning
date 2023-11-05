import os
import json
import glob
from tqdm import tqdm
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--json_folder_path", type=str, default='/root/autodl-tmp/RICO/semantic_annotations/', help="directory of original json folder")    
parser.add_argument("--csv_file_path", type=str, default='/root/autodl-tmp/data/RicoGPT.csv', help="dir parsed multilabel csv")   
args = parser.parse_args()

def json2csv(args):
    # get all captioned index from json dir
    for caption_json_file in tqdm(glob.glob(os.path.join(args.json_folder_path,'*.json'))):
        index_list.append(int(caption_json_file.split('/')[-1].split('.')[0]))
    # load bbox and captions per image
    for index in index_list:
        bbox_list,caption_list = [],[]
        json_path = os.path.join(self.json_folder_path,index+'.json')
        json_data = json.load(open(json_path))
        # loop each caption and bbox and write in csv
        for key in json_data:
            with open(args.csv_file_path,'a') as csv_file:
                writer = csv.writer(csv_file)
                # writin with [id,bbox/global,caption]
                writer.writerow([int(index),key,json_data[key]])

json2csv(args)