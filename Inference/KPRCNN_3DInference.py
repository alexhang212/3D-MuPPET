"""
Run i-muppet, then triangulate for 3D pose estimation
Inference script, reads in video from 3DPOP then does triangulate then reproject to 1 view
"""

import torchvision
import torch
from torchvision import transforms
import cv2
import numpy as np
import os
import math
import argparse

import sys
sys.path.append("./Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

import sys
sys.path.append("./")
sys.path.append("Utils")
import VisualizeUtil

import Network_utils
from PigeonMetaData import PIGEON_KEYPOINT_NAMES
from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation
from tqdm import tqdm
import pickle
from HungarianAlgorithm import hungarian_algorithm

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

def MatchID_BBox_Hungarian(GT_BBox,result,confidence_threshold=0.5,i=None):
    """Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap and hungarian algorithm"""

    
    Pred_BBoxList = result["boxes"].to('cpu').numpy().tolist()
    
    IndexBelowThreshold = np.where(result["scores"].to("cpu").numpy() < confidence_threshold)[0].tolist()
    

    Filtered_PredBBox = [box for j, box in enumerate(Pred_BBoxList) if j not in IndexBelowThreshold ]
    # GT_BBoxList = [val for val in GT_BBox.values()]

    
    OutDict = {}
    AssingedIndex = []
    NameList = []
    PercentOverlapMatrixList = []
    
    for key,val in GT_BBox.items() :
        NameList.append(key)
        PercentOverlapList  = [GetBBoxOverlap(Pred_BBox,val) for Pred_BBox in Pred_BBoxList]
        PercentOverlapMatrixList.append(PercentOverlapList)
    # import ipdb;ipdb.set_trace()

    if len(GT_BBox) != len(Filtered_PredBBox):
        ###If ground truth and predicted bbox doesnt have same length,
        ### Remove ground truth bbox with most mean error, probably cant match that
        for x in range((len(GT_BBox)-len(Filtered_PredBBox))):
            MaxOverlap = [np.array(row).max() for row in PercentOverlapMatrixList ]
            IndexMinMax = np.argmin(np.array(MaxOverlap)) #find index of which GT has lowest max overlap
            PercentOverlapMatrixList.pop(IndexMinMax)
            NameList.pop(IndexMinMax)

    ##Get distance Matrix
    PercentOverlapMat = 1-np.array(PercentOverlapMatrixList) #1 - overlap matrix, to for hungarian to minimize 
    MatchOut = hungarian_algorithm(PercentOverlapMat)
    OutDict = {NameList[index[0]]:index[1] for index in MatchOut }
    
    return OutDict

def MatchID_BBox(GT_BBox,result,confidence_threshold=0.5,i=None):
    """Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap"""

    
    Pred_BBoxList = result["boxes"].to('cpu').numpy().tolist()
    
    IndexBelowThreshold = np.where(result["scores"].to("cpu").numpy() < confidence_threshold)[0].tolist()
    

    # Filtered_PredBBox = [box for j, box in enumerate(Pred_BBoxList) if j not in IndexBelowThreshold ]
    # GT_BBoxList = [val for val in GT_BBox.values()]
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
            if MaxIndex in IndexBelowThreshold or MaxIndex in AssingedIndex:
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
    ###Viszualize all
    ##Reproject points
    # Points3DArr = np.array(list(Point3DDict.values()))
    # PointsNames = list(Point3DDict.keys())
    # Allimgpts, jac = cv2.projectPoints(Points3DArr, VisCam.rvec, VisCam.tvec, VisCam.camMat, VisCam.distCoef)

    # import ipdb;ipdb.set_trace()
    for Subject in SubjectList:
        PointsDict = {}

        SubjectDict = {k:v for k,v in Point3DDict.items() if k.startswith(Subject)}
        Points3DArr = np.array(list(SubjectDict.values()))
        PointsNames = list(SubjectDict.keys())
        try:
            Allimgpts, jac = cv2.projectPoints(Points3DArr, VisCam.rvec, VisCam.tvec, VisCam.camMat, VisCam.distCoef)
        except:
            continue

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
    
    
    
    

def Inference3DPOP(model, SequenceNum,DatasetPath,device,startFrame = 0,confidence_threshold=0.5,VisualizeIndex = 0):
    """Read a sequence from 3D pop then do inference with i-muppet + triangulate"""
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopDataset()
    
    VisCam = SequenceObj.camObjects[VisualizeIndex]
    
    # TotalFrames = len(VisCam.Keypoint2D.index)
    TotalFrames = 9000
    NumInd = len(SequenceObj.Subjects)
    
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


    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    counter = startFrame
    
    ##Setup video capture objects
    capList = []
    for cam in SequenceObj.camObjects:    
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)
        
    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 

        
    imsize = (int(cap.get(3)),int(cap.get(4)))
    out = cv2.VideoWriter(filename="KPRCNN3D_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    Points3DList = []

    for i in tqdm(range(TotalFrames)):
        PointsDict = {} #Points Dictionary for this frame

        FrameList = []
        BBoxList = []

        #read frames
        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)
        
        for x in range(len(SequenceObj.camObjects)):
            Img = FrameList[x].copy()
            Img= ProcessImage(Img,device)
            with torch.inference_mode():
                result = model([Img])[0]

            GT_BBox = {}
            for bird in SequenceObj.Subjects:
                top, bot = SequenceObj.camObjects[x].GetBBoxData(SequenceObj.camObjects[x].BBox,counter,bird)
                outList = [top[0],top[1],bot[0],bot[1]]
                if np.any(np.isnan(outList)):
                    continue
                GT_BBox.update({bird:outList})
                
            MatchDict = MatchID_BBox_Hungarian(GT_BBox,result,confidence_threshold=confidence_threshold)
            CamDict = {}
            
            Key2DPred = result["keypoints"].to("cpu").numpy()
            # import ipdb;ipdb.set_trace()
            ScoresList = result["scores"].to("cpu").numpy().tolist()
            boxesList = result["boxes"].to("cpu").numpy().tolist()
            
            BBoxList.append([box for k,box in enumerate(boxesList) if ScoresList[k]>confidence_threshold ])

            for k,v in MatchDict.items():
                #Corresponding 2D Points based on matching:
                Key2D = Key2DPred[v]
                
                CamDict.update({"%s_%s"%(k,PIGEON_KEYPOINT_NAMES[j]):[Key2D[j,0],Key2D[j,1]] for j in range(Key2D.shape[0])})    
                
            PointsDict.update({CamNames[x]: CamDict})
            
        
        TriangTool = BundleAdjustmentTool_Triangulation(CamNames,CamParamDict)
        TriangTool.PrepareInputData(PointsDict)
        Point3DDict = TriangTool.run()
        Points3DList.append(Point3DDict)
        
        ##Visualize
        frame = FrameList[VisualizeIndex]
        Boxes = BBoxList[VisualizeIndex]
        # import ipdb;ipdb.set_trace()
        frame = VizualizeAll(frame, counter,VisCam,Boxes,Point3DDict,imsize,SequenceObj.Subjects)
    
        cv2.imshow('Window',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # torch.cuda.empty_cache()
        out.write(frame)
        counter += 1
        # cv2.imwrite(img=frame, filename="Sample3D.jpg")
        
    out.release()

    return Points3DList

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
    parser.add_argument("--weight",
                        type=str,
                        default= "Weights/KPRCNN_3DPOP_Best.pt",
                        help="Path to pre-trained weight")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()
    
    DatasetPath = args.dataset
    SequenceNum = args.seq
    WeightsPath = args.weight
    
    CamNames = ["Cam1","Cam2","Cam3","Cam4"]
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device =  torch.device('cpu')

    network = LoadNetwork(WeightsPath,device)
    Points3DList = Inference3DPOP(network, SequenceNum,DatasetPath,device,startFrame = 0,VisualizeIndex = 0)