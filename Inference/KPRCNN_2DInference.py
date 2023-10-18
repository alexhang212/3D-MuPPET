""" Do inference on i-muppet from video input"""
import torchvision
import torch
from torchvision import transforms
import cv2
import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("Utils")


import Network_utils

from PigeonMetaData import PIGEON_KEYPOINT_NAMES
import VisualizeUtil


def LoadNetwork(WeightsPath,device):

    network = Network_utils.load_network(network_name='KeypointRCNN', 
                           looking_for_object='pigeon', 
                           eval_mode=True, pre_trained_model=WeightsPath,
                            device=device)
    
    return network

def ProcessImage(frame,device):
    frame = Network_utils.image_cv_to_rgb_tensor(frame)

    frame = Network_utils.normalize_tensor_image(
        tensor_image=frame,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
    frame = Network_utils.image_to_device(image=frame, device=device)
    
    return frame

def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid

    
def VisualizeAll(frame,result,imsize ,score_threshold=0.1, lines = True):
    """Visualize all i-muppet output"""
    ColourList = [ (255, 0 , 0 ),(0,255,0),(0,0,255),(63,133,205),(255,255,0),(255, 0 , 255),(128,0,128),(203,192,255),(0, 255, 255)]
    
    
    ##BBox first:
    BoxNumpy = result["boxes"].to('cpu').numpy()
    for x in range(BoxNumpy.shape[0]):
        if result["scores"][x].item() < score_threshold:
            continue
        x1,x2,y1,y2 = BoxNumpy[x, ...]
        cv2.rectangle(frame,(round(x1),round(x2)),(round(y1),round(y2)),[255,0,0],3)
        
    ##Keypoints:
    KpNumpy = result["keypoints"].to('cpu').numpy()
    for x in range(KpNumpy.shape[0]):
        if result["scores"][x].item() < score_threshold:
            continue
        
        for k in range(KpNumpy.shape[1]):
            point = (round(KpNumpy[x,k,0]),round(KpNumpy[x,k,1]))            
            if IsPointValid(imsize, point):
                cv2.circle(frame,point,2,ColourList[k], -1)

        if lines:
            PointsDict = {key:KpNumpy[x][j] for j, key in enumerate(PIGEON_KEYPOINT_NAMES)}

            VisualizeUtil.PlotLine(PointsDict,"leftEye","nose",[0,0,255],frame)
            VisualizeUtil.PlotLine(PointsDict,"rightEye","nose",[0,0,255],frame)
            VisualizeUtil.PlotLine(PointsDict,"beak","nose",[0,0,255],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftEye","rightEye",[0,0,255],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","topKeel",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"topKeel","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"leftShoulder","tail",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"tail","rightShoulder",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"tail","bottomKeel",[0,255,0],frame)
            VisualizeUtil.PlotLine(PointsDict,"bottomKeel","topKeel",[0,255,0],frame)
                            


    return frame
        
    

def RunInference(network, InputVideo,startFrame,device):
    
    cap = cv2.VideoCapture(InputVideo)
    cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    imsize = (cap.get(3),cap.get(4))
    counter=startFrame

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    TotalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    while(cap.isOpened()):

        ret, frame = cap.read()
        # frame = cv2.imread("/home/alexchan/Documents/Pigeon3DTrack/tempRaw.jpg")
        # print(counter)

        if ret == True:
            InferImage = frame.copy()
            # InferImage = cv2.cvtColor(InferImage, cv2.COLOR_BGR2GRAY)
            InferImage = ProcessImage(InferImage,device)
            
            ##inference:
            with torch.inference_mode():
                result = network([InferImage])[0]
            # import ipdb;ipdb.set_trace()
            frame = VisualizeAll(frame,result,imsize)
                
            cv2.imshow('Frame',frame)
            # cv2.imwrite(img = frame, filename = "temp.jpg")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()
    

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Input Video, path to input video")
    parser.add_argument("--weight",
                        type=str,
                        default= "Weights/KPRCNN_3DPOP_Best.pt",
                        help="Path to pre-trained weight")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":

    arg = ParseArgs()
    InputVideo = arg.input
    WeightsPath = arg.weight

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device =  torch.device('cpu')

    network = LoadNetwork(WeightsPath,device)
    
    RunInference(network, InputVideo,startFrame=0,device=device)