import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import workspace_utils as wsu


def build_model(output_size, hidden_layer, class_to_idx): 
    '''
    assemble pieces into a model
    '''
    model = models.vgg16(pretrained=True)
    input_size = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size,hidden_layer)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_layer,output_size)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    return model

def load_checkpoint(filepath):
    '''
    load saved model from filepath 
    '''
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['output_size'], 
                        checkpoint['hidden_layers'], 
                        checkpoint['mapping'])  
    
    model.load_state_dict(checkpoint['state_dict'])
    
    loaded = {'model': model, 
              'optimizer': checkpoint['optimizer'], 
              'epochs': checkpoint['epochs']}
    
    return loaded

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    size = 265, 265
    image.thumbnail(size)
    
    width, height = image.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    im = image.crop((left, top, right, bottom))    
    
    color_range = 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(im)/color_range
    
    normalized_image = (np_image - mean)/std
    normalized_im = normalized_image.transpose((2,0,1))
    
    return normalized_im


def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    array = process_image(image)
    tensor = torch.FloatTensor(array)
    model.eval()
    
    device = torch.device('cuda' if (torch.cuda.is_available())&(gpu==True) else 'cpu')
    print(device)
    model = model.to(device)
    tensor = tensor.to(device)
    input_ = tensor.unsqueeze_(0)
    
    output = model.forward(input_)
    predictions = torch.exp(output).topk(topk)
    
    probabilities, indexes = predictions
    probabilities = list(probabilities.cpu()[0].detach().numpy())
    indexes = list(indexes.cpu()[0].detach().numpy())
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = [idx_to_class[i] for i in indexes]
    return probabilities, classes

def name_classify(probabilities, classes, mapping):
    '''
    print the name of predicted class and probabilities
    '''
    for index, class_ in enumerate(classes):
        print(mapping[str(class_)], '\t', probabilities[index])
        

def main():
    parser = argparse.ArgumentParser(description='Basic usage: python predict.py /path/to/image checkpoint')
    parser.add_argument('image_path', type=str, default='flowers/test/101/image_07952.jpg', help='path to image')
    parser.add_argument('load_model', type=str, default='checkpoint', help='path to saved model')
    parser.add_argument('--top_k', type=int, default=5, help='number of predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to category names')
    parser.add_argument('--gpu', action='store_true', default=False, help='cuda or cpu')
    args=parser.parse_args()
    
    gpu_available = args.gpu
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    loaded_checkpoint = load_checkpoint(args.load_model+'.pth')
    model = loaded_checkpoint['model']
    
    probabilities, classes = predict(args.image_path, model, gpu_available, args.top_k)
    name_classify(probabilities, classes, cat_to_name)

if __name__ == "__main__":
    main()