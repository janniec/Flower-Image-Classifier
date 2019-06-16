import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
from collections import OrderedDict
import os
import re
import workspace_utils as wsu


def prep_data(data_dir):
    '''
    set up training torch data, 
    set up validation torch data, 
    set up testing torch data
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform = train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform = test_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloader['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    dataloader['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    
    return image_datasets, dataloader

def build_model(output_size, hidden_layer, class_to_idx, architecture='vgg'): 
    '''
    assemble elements into a model
    '''
    pretrained_models = {'alexnet':models.alexnet(pretrained=True)
                         , 'vgg':models.vgg16(pretrained=True)
                         , 'resnet':models.resnet50(pretrained=True)
                         , 'squeezenet':models.squeezenet1_1(pretrained=True)
                         , 'densenet':models.densenet169(pretrained=True)
                         , 'inception':models.inception_v3(pretrained=True)
    }
    
    
    model = pretrained_models[architecture]
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

def validation(model, validloader, criterion, gpu):
    '''
    test accuracy of model
    '''
    test_loss = 0
    accuracy = 0
    
    device = torch.device('cuda' if (torch.cuda.is_available())&(gpu==True) else 'cpu')
    model.to(device)
    
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu):
    '''
    train the model on validation data
    '''
    epochs = epochs
    print_every = print_every
    steps = 0
    
    device = torch.device('cuda' if (torch.cuda.is_available())&(gpu==True) else 'cpu')
    model.to(device)
    print('device:', device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, gpu)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Step: {}.. ".format(steps),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train()
    model.load_state_dict
    model.to('cpu')
    
    return model

def check_accuracy_on_test(testloader, criterion, model, gpu):
    '''
    train the model on testing data
    '''
    model.eval()
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, gpu)
        print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)))
        print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
        
def save_checkpoint(filepath, model, epochs, optimizer, output_size, hidden_layer):
    '''
    save model to filepath
    '''
    checkpoint = {'output_size': output_size,
              'hidden_layers': hidden_layer,
              'state_dict': model.state_dict(), 
              'mapping': model.class_to_idx, 
              'epochs': epochs, 
              'optimizer': optimizer.state_dict 
             }
    torch.save(checkpoint, filepath)
    
def strip_arch_name(string):
    '''
    process string from args.arch to a type of pretrained pytorch model
    '''
    string = string.split('_')[0].lower()
    arch = re.findall(r"[a-z]+", string, re.I)[0]
    return arch 

def main():
    parser = argparse.ArgumentParser(description= 'Basic usage: python train.py data_directory')
    parser.add_argument('data_dir', type=str, default='flowers', help='data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='directory to save model checkpoint')
    parser.add_argument('--arch',  type=str, default='vgg16', help='pretrained model')  
    parser.add_argument('--learning_rate', type=int, default=0.001, help='learning rate')   
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden layer')
    parser.add_argument('--epochs', type=int, default=6, help='epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='cuda or cpu')
    args = parser.parse_args()
    
    gpu_available = args.gpu
    image_datasets, dataloader = prep_data(args.data_dir)
    
    output_size = len(os.listdir(args.data_dir + '/train'))
    class_to_idx = image_datasets['train'].class_to_idx
    
    architecture = strip_arch_name(args.arch)

    model = build_model(output_size, args.hidden_units, class_to_idx, architecture)  
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    print_every = 40
    
    with wsu.active_session():
        model = do_deep_learning(model, dataloader['train'], dataloader['valid'], args.epochs, print_every, criterion, optimizer, gpu_available)
    with wsu.active_session():
        check_accuracy_on_test(dataloader['test'], criterion, model, gpu_available)
        
    save_checkpoint(args.save_dir, model, args.epochs, optimizer, output_size, args.hidden_units)

if __name__ == "__main__":
    main()