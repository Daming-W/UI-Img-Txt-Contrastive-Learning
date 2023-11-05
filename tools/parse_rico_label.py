import os
import json
import glob
from tqdm import tqdm
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--json_folder_path", type=str, default='/root/autodl-tmp/RICO/semantic_annotations/', help="directory of original json folder")
parser.add_argument("--target_json_path", type=str, default='/root/autodl-tmp/RICO/annotations.json', help="directory of target json file" )
parser.add_argument("--clean_type", type=str, default='cutoff', help="type of data cleaning")      
parser.add_argument("--csv_file_path", type=str, default='/root/autodl-tmp/RICO/cls.csv', help="dir parsed multilabel csv")   
args = parser.parse_args()

label_dict={'Checkbox':0, 'Advertisement':1, 'List Item':2, 'Number Stepper':3, 
            'Image':4, 'Toolbar':5, 'Multi-Tab':6, 'Card':7, 'Text':8,  
            'Drawer':9, 'Button Bar':10, 'Map View':11, 'Date Picker':12, 'Text Button':13, 
            'Web View':14, 'Pager Indicator':15, 'Input':16, 'Slider':17, 'Video':18, 'Modal':19,  
            'Radio Button':20, 'Bottom Navigation':21, 'Icon':22, 'On/Off Switch':23, 'Background Image':24
}

label_dict_c ={'Checkbox':0, 'List Item':1, 'Number Stepper':2, 
            'Image':3, 'Toolbar':4, 'Multi-Tab':5, 'Card':6, 'Text':7,  
            'Drawer':8, 'Button Bar':9, 'Map View':10, 'Date Picker':11, 'Text Button':12, 
            'Pager Indicator':13, 'Input':14, 'Slider':15, 'Video':16, 'Modal':17,  
            'Radio Button':18, 'Bottom Navigation':19, 'Icon':20, 'On/Off Switch':21
}


# input: json_path->the path of each specific json file
# output: pair_list->annotations of one image with style:
# [img_id, [label text, label id, xmin, ymin, xmax, ymax] * N(annotations contained in this img)]
def parse_json(json_path):
    with open(json_path,'r') as f: temp = json.loads(f.read())
    bbox,bbox_list,label_list=[],[],[]
    #parse VH to single dicts
    def parse_dict(dictionary, previous=None):
        previous = previous[:] if previous else []
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                if isinstance(value, dict): 
                    for d in parse_dict(value,  previous + [key]):
                        yield d
                elif isinstance(value, list) or isinstance(value, tuple):
                    for k,v in enumerate(value):
                        for d in parse_dict(v, previous + [key] + [[k]]):
                            yield d
                else: yield previous + [key, value]
        else: yield previous + [dictionary]
    #keep dicts with label or bounds info
    for i in list(parse_dict(temp)):
        if 'componentLabel' in i:
            label_list.append([i[-1],i[:-2]])

    #match bound and label by childern id
    json_id = (os.path.split(json_path)[-1]).split(".")[0]
    pair_list =set()
    for label in label_list:
        #pair.append(label_dict[label[0]])
        pair_list.add(label[0])
    
    return int(json_id),pair_list

'''
# test parsed result of a json sample 
json_path = '/root/autodl-tmp/RICO/semantic_annotations/15.json'
pair_list = parse_json(json_path)
print(pair_list)
'''

# main worker
def preprocess(args):
    label_set=set()
    label_list=[]
    with open(args.csv_file_path, 'w') as csvfile:
        csvfile_writer = csv.writer(csvfile)
        for json_path in tqdm(glob.glob(os.path.join(args.json_folder_path,'*.json'))):
            idx, pair_list = parse_json(json_path)
            if len(pair_list)>0:
                csvfile_writer.writerow([idx,list(pair_list)])


preprocess(args)