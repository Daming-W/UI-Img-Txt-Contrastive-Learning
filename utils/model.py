import clip_modified
from pytorch_grad_cam_modified import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, EigenGradCAM, GuidedBackpropReLUModel, LayerCAM 
from pytorch_grad_cam_modified import GradCAM_original, ScoreCAM_original, GradCAMPlusPlus_original, XGradCAM_original, EigenCAM_original, EigenGradCAM_original, GuidedBackpropReLUModel_original, LayerCAM_original
from torchvision.models import resnet50, resnet101
import torch
import timm


def getCLIP(model_name, gpu_id):
    reshape_transform = None
    if model_name == "RN50":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.layer4[-1]
    elif model_name == "RN101":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.layer4[-1]
    elif model_name == "RN50x4":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.layer4[-1]
    elif model_name == "RN50x16":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.layer4[-1]
    elif model_name == "ViT-B/32":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.transformer.resblocks[-1].ln_1
        reshape_transform = reshapeTransform7
    elif model_name == "ViT-B/16":
        model, preprocess,preprocess_aug = clip_modified.load(model_name, device = gpu_id, jit = False)
        target_layer = model.visual.transformer.resblocks[-1].ln_1
        reshape_transform = reshapeTransform14
    elif model_name == "RN50-pretrained":
        model = resnet50(pretrained=True).to(gpu_id)
        target_layer = model.layer4[-1]
    elif model_name == "DeiT-pretrained":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(gpu_id)
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshapeTransform_original
    elif model_name == "ViT-B/16-pretrained":
        model = timm.create_model("vit_base_patch16_224", pretrained=True).to(gpu_id)
        model.eval()
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshape_transform_vitb16
    elif model_name == "ViT-B/32-pretrained":
        model = timm.create_model("vit_base_patch32_224", pretrained=True).to(gpu_id)
        model.eval()
        target_layer = model.blocks[-1].norm1
        reshape_transform = reshape_transform_vitb32
    elif model_name == "RN101-pretrained":
        model = resnet101(pretrained=True).to(gpu_id)
        target_layer = model.layer4[-1]
    return model, target_layer, reshape_transform

def reshapeTransform7(x):
    x = x[1:, : , :]
    x = x.permute(1,2,0) #N * D * L-1 (width * height)
    x = x.reshape(x.shape[0], x.shape[1], 7, 7)
    return x

def reshapeTransform14(x):
    x = x[1:, : , :]
    x = x.permute(1,2,0) #N * D * L-1 (width * height)
    x = x.reshape(x.shape[0], x.shape[1], 14, 14)
    return x

def reshape_transform_vitb16(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vitb32(tensor, height=7, width=7):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getCAM(model_name, model, target_layer, gpu_id, reshape_transform):
    if model_name == "GradCAM":
        cam = GradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAMPlusPlus":
        cam = GradCAMPlusPlus(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "XGradCAM":
        cam = XGradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "ScoreCAM":
        cam = ScoreCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenCAM":
        cam = EigenCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenGradCAM":
        cam = EigenGradCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GuidedBackpropReLUModel":
        cam = GuidedBackpropReLUModel(model=model, gpu_id=gpu_id)
    elif model_name == "LayerCAM":
        cam = LayerCAM(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAM_original":
        cam = GradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GradCAMPlusPlus_original":
        cam = GradCAMPlusPlus_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "XGradCAM_original":
        cam = XGradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "ScoreCAM_original":
        cam = ScoreCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenGradCAM_original":
        cam = EigenGradCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "EigenCAM_original":
        cam = EigenCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    elif model_name == "GuidedBackpropReLUModel_original":
        cam = GuidedBackpropReLUModel_original(model=model, gpu_id=gpu_id)
    elif model_name == "LayerCAM_original":
        cam = LayerCAM_original(model=model, target_layer=target_layer, gpu_id=gpu_id, reshape_transform = reshape_transform)
    return cam

def reshapeTransform_original(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getFineTune(model_name, model, out_feature):
    if model_name == "RN50-pretrained":
        model.fc = torch.nn.Linear(in_features=2048, out_features = out_feature)
    elif model_name == "ViT-B/16-pretrained":
        model.head = torch.nn.Linear(in_features=768, out_features = out_feature)
    elif model_name == "ViT-B/32-pretrained":
        model.head = torch.nn.Linear(in_features=768, out_features = out_feature)
    elif model_name == "RN101-pretrained":
        model.fc = torch.nn.Linear(in_features=2048, out_features = out_feature)
    # for para in model.parameters():
    #     para.require_grad = False
    print(model)

    return model


######################################
### image caption model estimation ###
######################################
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs

class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoLSTM, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

def get_cnn_lstm(args):
    return CNNtoLSTM(args.embed_size, args.hidden_size, args.vocab_size, args.lstm_num_layers)