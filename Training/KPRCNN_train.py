# !/usr/bin/env python
"""Training script for pigeon data"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import os
import argparse
import numpy as np

sys.path.append("./")
sys.path.append("Repositories/i-muppet/")
sys.path.append("Utils/")
sys.path.append("Repositories/i-muppet/utils")
sys.path.append("Repositories/i-muppet/models")


from misc_utils import load_config, create_folder,display_dataset
from Network_utils import load_network

import JSONReader
from Utils.engine import train_one_epoch, getValidationLoss
from Torchutils import collate_fn
import MultiPigeonDataset


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        required=False,
                        default='./Configs/KPRCNN_3dpop.yaml',
                        help="Input: Specify the path to the configuration file.")
    parser.add_argument("--display_data",
                        action='store_true',
                        help="Visualization: Display images and targets of data.")

    args = parser.parse_args()

    return args


def main(args):

    ################
    ### Training ###
    ################

    print('[INFO] prepare training...')

    # load config
    config = load_config(args.config)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    print("Device: %s" %device)
    #################
    ### Variables ###
    species = config.species  # ['pigeon', 'cowbird', 'mouse']

    ## Pigeons
    session_train = config.dataset.training_session
    # Set pretrained_model to None if needed
    pretrained_model = config.dataset.pretrained_model
    flip_probability = config.data_augmentation.flip_probability
    gray_scale = config.data_augmentation.gray_scale
    scale_percentages_range = config.data_augmentation.scale_percentages_range
    dataset_Path = config.dataset.training_data_path
    # Annotation_JSON = config.dataset.JSON_path


    num_workers = config.num_workers

    brightness = config.data_augmentation.brightness
    contrast = config.data_augmentation.contrast
    saturation = config.data_augmentation.saturation
    hue = config.data_augmentation.hue
    # float: 0 gives blurred image, 1 gives original image, 2 increases sharpness by factor 2
    sharpness_factor = config.data_augmentation.sharpness_factor
    sharpness_prob = config.data_augmentation.sharpness_prob  # probability
    batch_size_train = config.hyperparameters.batch_size_train  # Batch size for training
    learning_rate = config.hyperparameters.learning_rate  # Learning rate of optimizer
    momentum = config.hyperparameters.momentum  # Momentum of optimizer
    weight_decay = config.hyperparameters.weight_decay  # Weight decay of optimizer (L2 penalty).
    step_size = config.hyperparameters.step_size  # Period of learning rate decay
    gamma = config.hyperparameters.gamma  # Multiplicative factor of learning rate decay. Default: 0.1
    num_epochs = config.hyperparameters.num_epochs  # Number of epochs
    print_freq = config.print_freq  # Iterations printed
    weight_path = './Data/Muppet_Weights/'
    #################

    # create 'my_weights' folder
    create_folder(weight_path)
    # set where to store trained weights
    save_model_at = os.path.join(weight_path, args.config[args.config.rfind('/') + 1:args.config.rfind('.')] + '.pt')

    print('\nSpecies:', config.species)

    print('Training session:', session_train)
    if pretrained_model is not None:
        # set path of pre-trained model
        pretrained_model = os.path.join('./Data/Weights', pretrained_model + '.pt')
        print('Pre-trained model:', pretrained_model)
    else:
        pass

    print('New model will be stored at', save_model_at, '\n')

    ### Load dataset ###
    print('[INFO] load data...')

    if args.display_data:
        transform = transforms.Compose(  # to display images
            [transforms.ToTensor(),
             transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
             transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_prob),
             transforms.Grayscale(3)
             ])
    else:
        if gray_scale == 1: #black and white
            transform = transforms.Compose(  # for training
                [transforms.ToTensor(),
                transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_prob),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Grayscale(3)
                ])
            
        else:
            transform = transforms.Compose(  # for training
                [transforms.ToTensor(),
                transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
                transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_prob),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    Annotation_JSON = os.path.join(dataset_Path,"Annotation","Train-2D.json")
    trainset = MultiPigeonDataset.MultiPigeonDataset(dataset_Path, 
                 Annotation_JSON, 
                 Type = "Train",
                 flip_probability=flip_probability, 
                 scale_percentages_range=scale_percentages_range,
                 transform=transform)
    # import ipdb;ipdb.set_trace()
    ##Validation Set
    Annotation_JSON_Val = os.path.join(dataset_Path,"Annotation","Val-2D.json")

    valset = MultiPigeonDataset.MultiPigeonDataset(dataset_Path, 
                Annotation_JSON_Val, 
                Type = "Val",
                flip_probability=flip_probability, 
                scale_percentages_range=scale_percentages_range,
                transform=transform)

    
    trainloader = DataLoader(trainset,
                             batch_size=batch_size_train,
                             shuffle=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)

    valloader = DataLoader(valset,
                             batch_size=batch_size_train,
                             shuffle=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)


    if args.display_data:
        # Display image and targets.
        display_dataset(trainloader, species=species, check_left_right=False)
    else:
        ### Load Keypoint R-CNN model ###
        net = load_network(network_name='KeypointRCNN',
                                         looking_for_object=species,
                                         eval_mode=False,
                                         pre_trained_model=pretrained_model,
                                         device=device
                                         )

        ### Define optimizer and learning rate scheduler ###
        print('[INFO] define optimizer and learning rate scheduler...')

        # construct an optimizer
        # torch.cuda.empty_cache()
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # import ipdb;ipdb.set_trace()
        ### Training ###
        LowestLoss = np.inf
        LowestSavePath = os.path.join(weight_path,"temp")


        print('[INFO] start training...\n')
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # train for one epoch, printing every 'print_freq' iterations
            train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=print_freq)
            # update the learning rate
            lr_scheduler.step()
            
            ValLoss = getValidationLoss(net,valloader,device)
            if ValLoss < LowestLoss:
                ###Val Loss is lowest, save it
                if os.path.exists(LowestSavePath):
                    os.remove(LowestSavePath) #remove previous one
                LowestSavePath = os.path.join(weight_path, "%s_CheckpointE%i_Loss%s_Lowest.pt"%(session_train,epoch,ValLoss))
                torch.save(net.state_dict(), LowestSavePath)
                LowestLoss = ValLoss

            if epoch % 10 == 0:
                ##Save checkpoint
                SavePath = os.path.join(weight_path, "%s_CheckpointE%i.pt"%(session_train,epoch))
                torch.save(net.state_dict(), SavePath)

        ### Save model ###
        # Save state_dict only
        print('\n[INFO] save our trained model...')
        torch.save(net.state_dict(), save_model_at)

    print('[INFO] finished')


if __name__ == '__main__':

    my_args = parse_args()
    print("args: {}".format(my_args))

    main(my_args)
