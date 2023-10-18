"""Custom network utils, small modifications"""


import torchvision
import torch
from torchvision import transforms
import cv2
import numpy as np


import PigeonMetaData



def load_network(network_name='KeypointRCNN', looking_for_object='pigeon', eval_mode=False, pre_trained_model=None,
                 device=torch.device('cpu')):
    print("[INFO] loading network...")

    if network_name == 'KeypointRCNN':
        net = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=PigeonMetaData.IS_COCO_INSTANCE,  # pretrained on COCO or not train2017
            progress=True,
            num_classes=2,  # number of output classes including background
            pretrained_backbone=True,  # backbone pretrained on Imagenet
            trainable_backbone_layers=0,
            num_keypoints=len(PigeonMetaData.PIGEON_KEYPOINT_NAMES)
        )
        # load (pre)trained model
        if pre_trained_model is not None:
            if looking_for_object == 'person':
                pass
            else:
                print('[INFO] Load pre-trained weights...')
                net.load_state_dict(torch.load(pre_trained_model))
    elif network_name == 'MaskRCNN':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,  # pretrained on COCO train2017
            progress=True,
            num_classes=91,  # number of output classes including background
            pretrained_backbone=True,  # backbone pretrained on Imagenet
            trainable_backbone_layers=0
        )
        # do not load (pre)trained model since I do not have any right now
    else:
        print('[ERROR] !!! select available network !!!')
        assert False
    net.to(device)
    if eval_mode:
        net.eval()

    return net

def image_cv_to_rgb_tensor(cv_image, scaling=True):

    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # bgr to rgb
    if scaling:  # [0, 255] scaled to [0, 1]
        image = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )(image)
    else:  # RGB tensor with data_type = uint8
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

    return image


def normalize_tensor_image(tensor_image, mean, std):
    image = transforms.Compose(
        [
            transforms.Normalize(mean, std)
        ]
    )(tensor_image)

    return image


def image_to_device(image, device):

    return image.to(device)


def infer_one_image(network, image):

    with torch.inference_mode():
        return network([image])[0]


def transform_predictions(predictions):

    bboxes = predictions['boxes']
    labels = predictions['labels']
    if 'keypoints' in list(predictions):
        keypoints = predictions['keypoints']  # [N,K,3]
    else:
        keypoints = None
    scores = predictions['scores']
    if 'masks' in list(predictions):
        masks = predictions['masks']  # [N, 1, H, W]
    else:
        masks = None

    return scores, labels, bboxes, keypoints, masks


