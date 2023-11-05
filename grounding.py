import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os

# setup args
parser = argparse.ArgumentParser()
parser.add_argument("--do_compare", type=bool, default =False,
                    help="if true, show both clip orig and ft to compare")
parser.add_argument("--image_folder_path", type=str, default='/root/autodl-tmp/RICO/combined/',
                    help="single image path or 4 images directory (grid)")                
parser.add_argument("--image_name", type=str,
                    help="single image path or 4 images directory (grid)")
parser.add_argument("--gpu_id", type=str, default='cpu',
                    help="GPU id to work on, \'cpu\'.")
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--distill_num", type=int, default=0,
                    help="Number of iterative masking")
parser.add_argument("--attack_type", type=str, default=None,
                    help="attack type: \"snow\", \"fog\"")
parser.add_argument("--sentence", type=str, default='',
                    help="input text")
parser.add_argument("--vis_res_folder_path", type=str, default='/root/autodl-tmp/UI_ITC/vis_res/',
                    help="path of vis results")                  
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ask for inputing and tokenize
id = input(f'image id: ')
args.image_name = str(id) + '.jpg'
bbox=[]
for i in range(4):
    num = input(f'bbox coord{i}:')
    if ',' in num: 
        num_list = num.split(',')
        bbox = [int(item) for item in num_list]
        break
    else:
        bbox.append(int(num))
print(bbox)

args.sentence = str(input(f'sentence (if checking tapbility, then enter TAP to continue): '))
if args.sentence == 'TAP':
    args.sentence = 'a clickable element'
text_tokens = clip_modified.tokenize(args.sentence)

# setup device
if args.gpu_id != 'cpu':
    args.gpu_id = int(args.gpu_id)

# get model and load weights 
itc_model, preprocess, preprocess_aug = clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
#model, target_layer, reshape_transform = getCLIP(model_name=args.clip_model_name, gpu_id=args.gpu_id)
itc_model.to(device)
#itc_model.load_state_dict(torch.load('/root/autodl-tmp/UI_ITC/checkpoints/ckp_ricogpt_rn50_0415_epoch40.pt'))
print('model loaded')

# Image read target image
image_path=args.image_folder_path + args.image_name
if os.path.isfile(image_path):
    image = Image.open(image_path).convert('RGB')            
    # resize to match caption
    image = image.resize((1440, 2560))
    image = image.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
    image.save("etst.jpg")
    image = preprocess(image).unsqueeze(0)
else:
    images = []
    for f in os.listdir(image_path):
        images.append(Image.open(os.path.join(image_path, f)))

def itc_predict(model,image,text_tokens):
    image = image.cuda(non_blocking=True)
    text_tokens = text_tokens.cuda(non_blocking=True)
    # computing cos sim
    logits_per_image, logits_per_text = model(image,text_tokens)
    print(f'logits per image : {logits_per_image.item()}')
    print(f'logits per text  : {logits_per_text.item()}')

itc_predict(itc_model,image,text_tokens)