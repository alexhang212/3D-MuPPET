""" 
inference on single video for MaskRCNN (FROM DETECTRON) + DLC WITH SORT

two step, pretrained MaskRCNN, then DLC to open up GPU

"""
import cv2 
import torch
import argparse

import sys
sys.path.append("Repositories/DeepLabCut/")
sys.path.append("Repositories/DeepLabCut-live")

import deeplabcut as dlc
from dlclive import DLCLive, Processor

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import shutil

from torchvision.transforms import transforms as transforms

sys.path.append("Repositories/sort/")
from sort import *



import detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]




def Process_Crop(Crop, CropSize):
    """Crop image and pad, if too big, will scale down """
    # import ipdb;ipdb.set_trace()
    if Crop.shape[0] > CropSize[0] or Crop.shape[1] > CropSize[1]: #Crop is bigger, scale down
        ScaleProportion = min(CropSize[0]/Crop.shape[0],CropSize[1]/Crop.shape[1])
        
        width_scaled = int(Crop.shape[1] * ScaleProportion)
        height_scaled = int(Crop.shape[0] * ScaleProportion)
        Crop = cv2.resize(Crop, (width_scaled,height_scaled), interpolation=cv2.INTER_LINEAR)  # resize image

        # Points2D = {k:[v[0]*ScaleProportion,v[1]*ScaleProportion] for k,v in Points2D.items()}
    else:
        ScaleProportion = 1
        
    if Crop.shape[0] %2 ==0:
        #Shape is even number
        YPadTop = int((CropSize[1] - Crop.shape[0])/2)
        YPadBot = int((CropSize[1] - Crop.shape[0])/2)
    else:
        YPadTop = int( ((CropSize[1] - Crop.shape[0])/2)-0.5)
        YPadBot = int(((CropSize[1] - Crop.shape[0])/2)+0.5)
    ##Padding:
    if Crop.shape[1] %2 ==0:
        #Shape is even number
        XPadLeft = int((CropSize[0] - Crop.shape[1])/2)
        XPadRight= int((CropSize[0] - Crop.shape[1])/2)
    else:
        XPadLeft =  int(((CropSize[0] - Crop.shape[1])/2)-0.5)
        XPadRight= int(((CropSize[0] - Crop.shape[1])/2)+0.5)



    OutImage = cv2.copyMakeBorder(Crop, YPadTop,YPadBot,XPadLeft,XPadRight,cv2.BORDER_CONSTANT,value=[0,0,0])
    
    return OutImage,ScaleProportion, YPadTop,XPadLeft



def DLCInference(Crop,dlc_liveObj,CropSize):
    """Inference for DLC"""

    ###Scale crop if image bigger than cropsize
    # import ipdb;ipdb.set_trace()
    if Crop.shape[0] > CropSize[0] or Crop.shape[1] > CropSize[1]: #Image bigger than crop size, scale down
        ScaleRatio = min([CropSize[0]/Crop.shape[0], CropSize[1]/Crop.shape[1]])
        ScaleWidth = round(Crop.shape[1] * ScaleRatio)
        ScaleHeight = round(Crop.shape[0]*ScaleRatio)
        resizedCrop = cv2.resize(Crop, (ScaleWidth,ScaleHeight), interpolation=cv2.INTER_LINEAR)  # resize image
        ScaleUpRatio = 1/ScaleRatio #ratio to scale keypoints back up to original
        # import ipdb;ipdb.set_trace()
    else:
        resizedCrop = Crop
        ScaleUpRatio = 1
    # cv2.imwrite(filename="tempresize.jpg", img=resizedCrop)
    # cv2.imwrite(filename="temp.jpg", img=Crop)
    if dlc_liveObj.sess == None: #if first time, init
        DLCPredict2D = dlc_liveObj.init_inference(resizedCrop)

    DLCPredict2D= dlc_liveObj.get_pose(resizedCrop)
    DLCPredict2D[:,0] = DLCPredict2D[:,0]*ScaleUpRatio
    DLCPredict2D[:,1] = DLCPredict2D[:,1]*ScaleUpRatio

    return DLCPredict2D


def VisualizeAll(frame, box, DLCPredict2D,MeanConfidence,ScaleBBox, imsize):
    """Visualize all stuff"""
    colourList = [(255,255,0),(255,0 ,255),(128,0,128),(203,192,255),(0, 255, 255),(255, 0 , 0 ),(63,133,205),(0,255,0),(0,0,255)]
    ##Order: Lshoulder, Rshoulder, topKeel,botKeel,Tail,Beak,Nose,Leye,Reye
    ##Points:
    PlotPoints = []
    for x,point in enumerate(DLCPredict2D):
        roundPoint = [round(point[0]+box[0]),round(point[1]+box[1])]
        cv2.circle(frame,roundPoint,1,colourList[x], 5)
        PlotPoints.append(roundPoint)

    cv2.rectangle(frame,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),[255,0,0],3)

    return frame, PlotPoints


def InferenceLoopMaskRCNN(InputVideo,predictor,startFrame=0,TotalFrames =-1,ScaleBBox=1,Dilate=5):
    """Loop through video for SAM, save framewise info"""
    transform = transforms.Compose([transforms.ToTensor()])

    cap = cv2.VideoCapture(InputVideo)
    # cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    imsize = (int(cap.get(3)),int(cap.get(4)))
    counter=startFrame

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)


    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    FrameResultList = []

    if os.path.exists("IntermediateFolder2/"):
        shutil.rmtree('IntermediateFolder2/')

    os.mkdir("IntermediateFolder2/")


    for i in tqdm(range(TotalFrames), desc = "MaskRCNN Inference:"):
        ret, frame = cap.read()
        FrameDict = {}
        if ret == True:
            InferFrame = frame.copy()
            outputs = predictor(InferFrame)["instances"].to("cpu")


            BirdIndex = np.where(outputs.pred_classes.numpy() == 14)[0] #14 is ID for bird
            BirdBBox = outputs.pred_boxes[BirdIndex].tensor.numpy()
            BirdMasks = (outputs.pred_masks>0.95).numpy()[BirdIndex]

            for x in range(BirdBBox.shape[0]):

                bbox = list(BirdBBox[x])
                Mask = BirdMasks[x]>0
                Mask = np.array(Mask,dtype=np.uint8)

                if Dilate > 0:
                    DilateKernel = np.ones((Dilate,Dilate),np.uint8)
                    Mask = cv2.dilate(Mask,DilateKernel,iterations = 3) 

                # import ipdb;ipdb.set_trace()
                Mask = np.array(Mask,dtype=np.uint8)
                Mask = Mask.reshape(imsize[1],imsize[0],1)
                Crop = cv2.bitwise_and(frame, frame, mask=Mask)


                ##change box to XYWH to scale up
                bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
                ScaleWidth = ((ScaleBBox * bbox[2])/2)-(bbox[2]/2)
                ScaleHeight = ((ScaleBBox * bbox[3])/2)-(bbox[3]/2)

                x1 = round(bbox[0]-ScaleWidth) if round(bbox[0]-ScaleWidth)>0 else 0
                y1 = round(bbox[1]-ScaleHeight)if round(bbox[1]-ScaleHeight)>0 else 0
                x2 = round(bbox[0]+bbox[2]+ScaleWidth) if round(bbox[0]+bbox[2]+ScaleWidth) < imsize[0] else imsize[0]
                y2 = round(bbox[1]+bbox[3]+ScaleHeight)if round(bbox[1]+bbox[3]+ScaleHeight) < imsize[1] else imsize[1]
                bbox = [x1,y1,x2,y2]
                BirdCrop = Crop[y1:y2,x1:x2] #bbox is XYWH
                ImgName = os.path.join("IntermediateFolder2" , "F%s_Obj%s.jpg"%(counter,x))
                cv2.imwrite(img = BirdCrop, filename=ImgName)
                FrameDict.update({x:{"ObjIndex":x,"Path":ImgName, "bbox":bbox}})

            FrameResultList.append(FrameDict)
            # import ipdb;ipdb.set_trace()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # counter += 1
            break
        counter += 1
    cap.release()
    # cv2.destroyAllWindows()

    return FrameResultList
    
def VisualizeTracks(frame,TrackingOut):
    """Visualize SORT tracking output"""

    for x in range(TrackingOut.shape[0]):
        Point = (round(TrackingOut[x][0]),round(TrackingOut[x][1])-5)

        frame = cv2.putText(frame,"ID: %s" %str(int(TrackingOut[x][4])),Point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, [255,0,0],3, cv2.LINE_AA)

    return frame


def InferenceLoopDLC(FrameResultList,dlc_liveObj, InputVideo,CropSize,startFrame=0,TotalFrames =1800,ScaleBBox=1,DLCThreshold=0.5):
    """Inference Loop for DLC, reading in mask info"""

    # import ipdb;ipdb.set_trace()
    VideoName = os.path.basename(InputVideo).split(".")[0]
    cap = cv2.VideoCapture(InputVideo)
    # cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    imsize = (int(cap.get(3)),int(cap.get(4)))
    counter=startFrame

    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    Sort_tracker = Sort(max_age = 20, min_hits = 5)
    # import ipdb;ipdb.set_trace()
    # DeepSort_Tracker = tracker.Tracker(nn_matching.NearestNeighborDistanceMetric("euclidean", 0.5))

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    out = cv2.VideoWriter(filename=os.path.join("PigeonWild_Sample_%s.mp4"%VideoName), apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    OutDict = {}
    for i in tqdm(range(TotalFrames)):
        FrameOutputDict = {"2DKeyList": [], "BBoxList" : []}

        ret, frame = cap.read()
        # print(counter)
        # import ipdb;ipdb.set_trace()
        if ret == True:
            InferFrame = frame.copy()
            InferFrame = InferFrame
            # InferFrame = torch.tensor(InferFrame).to("cuda")
            ##Segment everything:
            FrameDict = FrameResultList[i]
            # show_anns(InferFrame, results)

            TrackingArrayList = []
            # DetectionsList = []


            for k, objDict in FrameDict.items():
                BirdCrop = cv2.imread(objDict["Path"])
                bbox = objDict["bbox"]
                DLCPredict2D= DLCInference(BirdCrop,dlc_liveObj,CropSize)
                MeanConfidence = DLCPredict2D[:,2].mean()
                # print(MeanConfidence)


                if MeanConfidence > DLCThreshold: #if mean keypoint confidence is higher than this threshold, consider bird

                    bbox.append(MeanConfidence)
                    #SORT:
                    TrackingArrayList.append(bbox)
                    FrameOutputDict["BBoxList"].append(bbox)

                    frame, PlotPoints = VisualizeAll(frame, bbox, DLCPredict2D,MeanConfidence,ScaleBBox,imsize)
                    FrameOutputDict["2DKeyList"].append(np.array(PlotPoints))

            if len(FrameOutputDict["BBoxList"]) == 0: #if nothing detected
                FrameOutputDict["BBoxTracked"] = np.array([])
                out.write(frame)
                counter += 1
                continue

            TrackingArray = np.array(TrackingArrayList)
            TrackingOut = Sort_tracker.update(TrackingArray)

            FrameOutputDict["BBoxTracked"] = TrackingOut

            # import ipdb;ipdb.set_trace()
            
            cv2.imshow('Frame',frame)
            OutDict.update({counter :FrameOutputDict })


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # counter += 1
            break
        out.write(frame)
        counter += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    shutil.rmtree("./IntermediateFolder2")

    



def RunInference(predictor,dlc_liveObj, InputVideo,CropSize,startFrame=0,TotalFrames =-1,ScaleBBox=1,DLCThreshold=0.5,Dilate=5):
    """
    Params:
    Dilate: dilate/ increase mask size
    DLCThreshold: what threshold of mean DLC confidence to accept as bird
    ScaleBBox: scale the bbox bigger
    TotalFrames: total frames to run
    StartFrame: frame to start from

    """
    
    FrameResultList = InferenceLoopMaskRCNN(InputVideo,predictor,startFrame=startFrame,TotalFrames =TotalFrames,ScaleBBox=ScaleBBox,Dilate=Dilate)
    del predictor
    torch.cuda.empty_cache()
    # import ipdb;ipdb.set_trace()
    InferenceLoopDLC(FrameResultList,dlc_liveObj, InputVideo,CropSize,startFrame=startFrame,TotalFrames =TotalFrames,ScaleBBox=ScaleBBox,DLCThreshold=DLCThreshold)




def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Input Video, path to input video")
    parser.add_argument("--DLCweight",
                        type=str,
                        default= "Weights/DLC_Mask/",
                        help="Path to pre-trained weight for exported DLC model directory")

    arg = parser.parse_args()

    return arg




if __name__ == "__main__":
    args = ParseArgs()
    VidPath = args.input
    ExportModelPath = args.DLCweight
    CropSize = (320,320)

    dlc_proc = Processor()
    dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)

    ###Detectron:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    RunInference(predictor,dlc_liveObj, VidPath,CropSize)

