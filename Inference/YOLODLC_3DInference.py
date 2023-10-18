""" inference on single video for YOLO + DLC"""
import cv2 
from ultralytics import YOLO
import torch
from tqdm import tqdm
import numpy as np
import argparse

import sys
sys.path.append("Repositories/DeepLabCut-live")


sys.path.append("Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

sys.path.append("Utils")
from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation
import VisualizeUtil

import math
import os
import deeplabcut as dlc
from dlclive import DLCLive, Processor

PIGEON_KEYPOINT_NAMES = ['bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail','hd_beak', 'hd_leftEye', 'hd_rightEye', 'hd_nose']



   
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



def DLCInference(InferFrame,box,dlc_liveObj,CropSize):
    """Inference for DLC"""
    box = [0 if val < 0 else val for val in box] #f out of screen, 0


    Crop = InferFrame[round(box[1]):round(box[3]),round(box[0]):round(box[2])]


    if dlc_liveObj.sess == None: #if first time, init
        DLCPredict2D = dlc_liveObj.init_inference(Crop)

    DLCPredict2D= dlc_liveObj.get_pose(Crop)

    return DLCPredict2D

def getColor(keyPoint):
    if keyPoint.endswith("beak"):
        return (255, 0 , 0 )
    elif keyPoint.endswith("nose"):
        return (63,133,205)
    elif keyPoint.endswith("leftEye"):
        return (0,255,0)
    elif keyPoint.endswith("rightEye"):
        return (0,0,255)
    elif keyPoint.endswith("leftShoulder"):
        return (255,255,0)
    elif keyPoint.endswith("rightShoulder"):
        return (255, 0 , 255)
    elif keyPoint.endswith("topKeel"):
        return (128,0,128)
    elif keyPoint.endswith("bottomKeel"):
        return (203,192,255)
    elif keyPoint.endswith("tail"):
        return (0, 255, 255)
    else:
        return (0,165,255)
        # return (0,255,0)

def VizualizeAll(frame, counter,VisCam,Boxes,Point3DDict,imsize,SubjectList, Lines = True):
    """Visualize all on visualize cam"""

    # import ipdb;ipdb.set_trace()
    for Subject in SubjectList:
        PointsDict = {}

        SubjectDict = {k:v for k,v in Point3DDict.items() if k.startswith(Subject)}
        if len(SubjectDict) == 0:
            continue

        Points3DArr = np.array(list(SubjectDict.values()))
        PointsNames = list(SubjectDict.keys())
        Allimgpts, jac = cv2.projectPoints(Points3DArr, VisCam.rvec, VisCam.tvec, VisCam.camMat, VisCam.distCoef)


        for i in range(len(Allimgpts)):
            pts = Allimgpts[i]
            if np.isnan(pts[0][0]) or math.isinf(pts[0][0]) or math.isinf(pts[0][1]):
                continue
            #######
            point = (round(pts[0][0]),round(pts[0][1]))
            if IsPointValid(imsize,point):
                colour = getColor(PointsNames[i])
                cv2.circle(frame,point,1,colour, -1)
            
            PointsDict.update({PointsNames[i]:point})
    
        ##Plot Lines:
        if Lines:
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
                            

    ##Plot BBox:
    for box in Boxes:
        # import ipdb;ipdb.set_trace()
        cv2.rectangle(frame,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),[255,0,0],3)
      
    return frame
    

def GetBBoxOverlap(BBox1,BBox2):
    """
    Calculate bounding box overlap between 2 boxes, inputted as list, as [x1,x2,y1,y2]
    
    Outputs percentage overlap
    """
    dx = min(BBox1[2], BBox2[2]) - max(BBox1[0], BBox2[0])
    dy = min(BBox1[3], BBox2[3]) - max(BBox1[1], BBox2[1])
    if (dx>=0) and (dy>=0):
        #Area of ind 1:
        Area1 = (BBox1[2]-BBox1[0])*(BBox1[3]-BBox1[1])
        if Area1 == 0:
            return 0
        return (dx*dy)/Area1
    else:
        return 0
    
def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid


def MatchID_BBox(GT_BBox,Pred_BBoxList,confidence_threshold=0.5,i=None):
    """Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap"""
    
    OutDict = {}
    AssingedIndex = []
    

    #Just match all ground truth to all
    for key,val in GT_BBox.items() :
        FoundMatch = False
        PercentOverlapList  = [GetBBoxOverlap(Pred_BBox,val) for Pred_BBox in Pred_BBoxList]
        # import ipdb;ipdb.set_trace()
        if len(PercentOverlapList) ==0 or sum(PercentOverlapList) == 0: #no bbox predicted or no overlap
            continue

        while FoundMatch == False:            
            MaxIndex = np.argmax(PercentOverlapList)
            if MaxIndex in AssingedIndex:
                if sum(PercentOverlapList) == 0: ##all possible match changed to 0, no match
                    break
                PercentOverlapList[MaxIndex] = 0
                # print(i) #debugging purposes
                
                continue
                #continue looping if the bbox has low score or is already assinged.
            else:
                OutDict.update({key:MaxIndex})
                AssingedIndex.append(MaxIndex)
                FoundMatch = True

    return OutDict


def RunYOLOLoop(YOLOModel, DatasetPath,SequenceNum,VisualizeIndex,startFrame,TotalFrames,ScaleBBox):
    """Get YOLO boxes"""
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopDataset()
    
    VisCam = SequenceObj.camObjects[VisualizeIndex]
    
    NumInd = len(SequenceObj.Subjects)

    counter=startFrame

    CamParamDict = {} #dictionary for camera parameters
    CamNames = []
    for cam in SequenceObj.camObjects:
        CamNames.append(cam.CamName)

    ##Setup video capture objects
    capList = []
    for cam in SequenceObj.camObjects:    
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)
        
    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 

        
    imsize = (int(cap.get(3)),int(cap.get(4)))
    # out = cv2.VideoWriter(filename="YOLODLC3D_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    OutbboxList = []

    for i in tqdm(range(TotalFrames), desc = "Running YOLO:"):
        FramebboxDict = {} #Points Dictionary for this frame

        FrameList = []
        BBoxList = []
        #read frames
        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)

        for x in range(len(SequenceObj.camObjects)): #for each cam
            InferFrame = FrameList[x].copy()
            InferFrame = InferFrame
            # InferFrame = torch.tensor(InferFrame).to("cuda")
            results = YOLOModel(InferFrame, imgsz=3840)
            ##Filter for birds:
            classID = [key for key,val in results[0].names.items() if val == "bird"][0]
            # frame = results[0].plot()
            DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()

            
            # bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
            bbox = results[0].boxes.xywh.cpu().numpy().tolist()
            ##Filter birds only:
            bbox = [box for x,box in enumerate(bbox) if DetectedClasses[x] == classID]


            bbox = [[box[0],box[1],box[2]*ScaleBBox,box[3]*ScaleBBox] for box in bbox] #scale width and height
            ##convert back to xyxy:
            bbox = [[box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] for box in bbox]
            BBoxList.append(bbox)
            # FramebboxDict.update({CamNames[x]:BBoxList})
            
        OutbboxList.append(BBoxList)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1


    for cap in capList:
        ret, frame = cap.read()
        FrameList.append(frame)

    return OutbboxList

def RunDLCLoop(dlc_liveObj,OutbboxList,SequenceNum,DatasetPath,CropSize,startFrame,TotalFrames,VisualizeIndex = 0,ScaleBBox=1):
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopDataset()
    
    VisCam = SequenceObj.camObjects[VisualizeIndex]
    
    NumInd = len(SequenceObj.Subjects)

    cv2.namedWindow("Window",cv2.WINDOW_NORMAL)
    counter=startFrame

    CamParamDict = {} #dictionary for camera parameters
    CamNames = []
    for cam in SequenceObj.camObjects:
        ##Camera params:
        CamParamDict.update({cam.CamName:{
            "R":cam.rvec,
            "T":cam.tvec,
            "cameraMatrix":cam.camMat,
            "distCoeffs":cam.distCoef
            
        }})
        CamNames.append(cam.CamName)

    ##Setup video capture objects
    capList = []
    for cam in SequenceObj.camObjects:    
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)
        
    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 

        
    imsize = (int(cap.get(3)),int(cap.get(4)))
    out = cv2.VideoWriter(filename="YOLODLC3D_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    for i in tqdm(range(TotalFrames)):
        PointsDict = {} #Points Dictionary for this frame

        FrameList = []
        BBoxList = []
        #read frames
        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)

        for x in range(len(SequenceObj.camObjects)): #for each cam
            InferFrame = FrameList[x].copy()
            InferFrame = InferFrame

            bbox = OutbboxList[i][x]
            # BBoxList.append(bbox)
            # import ipdb;ipdb.set_trace()

            GT_BBox = {}
            for bird in SequenceObj.Subjects:
                top, bot = SequenceObj.camObjects[x].GetBBoxData(SequenceObj.camObjects[x].BBox,counter,bird)
                outList = [top[0],top[1],bot[0],bot[1]]
                if np.any(np.isnan(outList)):
                    continue
                GT_BBox.update({bird:outList})
                
            MatchDict = MatchID_BBox(GT_BBox,bbox)
            CamDict = {}
            for k,v in MatchDict.items():
                box = bbox[v]
                #Corresponding 2D Points based on matching:
                DLCPredict2D= DLCInference(InferFrame,box,dlc_liveObj,CropSize)
                CamDict.update({"%s_%s"%(k,PIGEON_KEYPOINT_NAMES[j]):[DLCPredict2D[j,0]+box[0],DLCPredict2D[j,1]+box[1]] for j in range(DLCPredict2D.shape[0])})    
                
            PointsDict.update({CamNames[x]: CamDict})
            

        # import ipdb;ipdb.set_trace()
        TriangTool = BundleAdjustmentTool_Triangulation(CamNames,CamParamDict)
        TriangTool.PrepareInputData(PointsDict)
        Point3DDict = TriangTool.run()

        # import ipdb;ipdb.set_trace()
        frame = FrameList[VisualizeIndex]
        Boxes = OutbboxList[i][VisualizeIndex]

        frame = VizualizeAll(frame, counter,VisCam,Boxes,Point3DDict,imsize,SequenceObj.Subjects)

        cv2.imshow('Window',frame)
        # import ipdb;ipdb.set_trace()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        counter += 1

    for cap in capList:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    

def RunInference3DPOP(YOLOModel,dlc_liveObj,SequenceNum,DatasetPath,CropSize,startFrame=0,TotalFrames=1800,VisualizeIndex = 0,ScaleBBox=1):
    OutbboxList = RunYOLOLoop(YOLOModel, DatasetPath,SequenceNum,VisualizeIndex,startFrame,TotalFrames,ScaleBBox)
    del YOLOModel
    torch.cuda.empty_cache()
    RunDLCLoop(dlc_liveObj,OutbboxList,SequenceNum,DatasetPath,CropSize,startFrame,TotalFrames,VisualizeIndex,ScaleBBox)


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the 3DPOP dataset")
    parser.add_argument("--seq",
                        type=int,
                        required=True,
                        help="Sequence Number of 3D POP")
    parser.add_argument("--YOLOweight",
                        type=str,
                        default= "Weights/YOLO_Barn.pt",
                        help="Path to pre-trained weight for YOLO model")
    parser.add_argument("--DLCweight",
                        type=str,
                        default= "Weights/DLC_Barn/",
                        help="Path to pre-trained weight for exported DLC model directory")

    arg = parser.parse_args()

    return arg

if __name__ == "__main__":
    args = ParseArgs()
    YOLOPath = args.YOLOweight
    DatasetPath = args.dataset
    SequenceNum = args.seq
    ExportModelPath = args.DLCweight

    CropSize = (320,320)

    YOLOModel = YOLO(YOLOPath)

    dlc_proc = Processor()
    dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)
    
    RunInference3DPOP(YOLOModel,dlc_liveObj,SequenceNum,DatasetPath,CropSize,startFrame=0,TotalFrames = 1800,VisualizeIndex = 0,ScaleBBox=1)

