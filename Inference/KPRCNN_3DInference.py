"""
Run i-muppet, then triangulate for 3D pose estimation
Inference script, reads in video from 3DPOP then does triangulate then reproject to 1 view
"""

import torch
import cv2
import numpy as np
import math
import argparse

import sys
sys.path.append("Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

import sys
sys.path.append("./")
sys.path.append("Utils")
import VisualizeUtil

import Network_utils
from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation, BundleAdjustmentTool_Triangulation_Filter
from tqdm import tqdm

import itertools
import HungarianAlgorithm

import copy
from scipy.spatial import distance_matrix

sys.path.append("Repositories/sort/")
from sort import *
PIGEON_KEYPOINT_NAMES = ['hd_beak', 'hd_leftEye', 'hd_rightEye', 'hd_nose', 'bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail']

def GetEucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""
    
    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")
    
    return EucDist

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


def MatchID_BBox(GT_BBox,result,confidence_threshold=0.5,i=None):
    """Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap"""

    
    Pred_BBoxList = result["boxes"].to('cpu').numpy().tolist()
    
    IndexBelowThreshold = np.where(result["scores"].to("cpu").numpy() < confidence_threshold)[0].tolist()
    
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


def MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,DminThresh = 200,confidence_threshold=0.5):
    """
    Matching algorithm to get correspondeces based on reprojection
    Dmin: based on Huang et al 2020, minimum threshold, after that finish matching
    """
    
    ##Get all combinations of cameras
    CamNamePairs = list(itertools.combinations(CamNames,2))
    # CamIndexPairs = [(CamNames.index(a), CamNames.index(b)) for a,b in CamNamePairs]
    CamPairDict = {}
    
    #For each possible camera pair and for each possible detection pair, triangulate and reproject
    ## to get reprojection error. Hungarian algorithm for each pair to then get pairs with min cost (error)

    for CamPair in CamNamePairs:#For each possible camera pair
        Cam1KP = MatchingDict[CamPair[0]]["Keypoints"]
        Cam2KP = MatchingDict[CamPair[1]]["Keypoints"]

        TempCamNames = [CamPair[0],CamPair[1]]

        ObjectPairs = list(itertools.product(range(len(Cam1KP)),range(len(Cam2KP))))
        ErrorArray = np.full((len(Cam1KP),len(Cam2KP)),0, dtype=np.float32) #left(rows) is KP1, top(columns) is KP2

        Point3DDict = {}
        for IndexPair in ObjectPairs:
            #Prepare data:
            Cam1Dict = {"%s"%(PIGEON_KEYPOINT_NAMES[j]):[Cam1KP[IndexPair[0]][j,0],Cam1KP[IndexPair[0]][j,1]] for j in range(Cam1KP[IndexPair[0]].shape[0])}
            Cam2Dict = {"%s"%(PIGEON_KEYPOINT_NAMES[j]):[Cam2KP[IndexPair[1]][j,0],Cam2KP[IndexPair[1]][j,1]] for j in range(Cam2KP[IndexPair[1]].shape[0])}

            Point2DDict = {TempCamNames[0]:Cam1Dict, TempCamNames[1]:Cam2Dict}

            TriangTool = BundleAdjustmentTool_Triangulation(TempCamNames,CamParamDict)
            TriangTool.PrepareInputData(Point2DDict)
            All3DPoints = TriangTool.Points3DArr[TriangTool.PointIndexArr]
            Reproject2D = TriangTool.Reproject(All3DPoints)
            Point3DDict.update({IndexPair:TriangTool.Points3DArr})

            ErrorList = [GetEucDist(TriangTool.Points2DArr[x],Reproject2D[x]) for x in range(TriangTool.Points2DArr.shape[0])]
            ErrorArray[IndexPair[0],IndexPair[1]] = np.array(ErrorList).mean()


        #Feed Error array into hungarian algorithm to find best match
        FinalMatches = HungarianAlgorithm.hungarian_algorithm(ErrorArray)
        FinalMatch3DPoints = [Point3DDict[IndexPair] for IndexPair in FinalMatches]
        CamPairDict.update({CamPair:{"Matches":FinalMatches,"3DPoints":FinalMatch3DPoints }})

    ### Implementing Huang et al 2020 for dynamic matching
    ### First get matrix of differences between all 3d point pairs (with corresponding camera and index pairs)
    ### Find the min in that matrix, then start a global matched set
    ### if any set already has any camera/index pair, just add to the set, if not start a new set
    ### if a camera already exist in a set (with different index), use the existing one and throw this out

    PointsList = []
    IndexList = []
    for camPair, val in CamPairDict.items():
        for x in range(len(val["Matches"])):
            PointsList.append(val["3DPoints"][x][7].tolist()) #bot keel instead of beak
            IndexList.append([(camPair[0],val["Matches"][x][0]),(camPair[1],val["Matches"][x][1])])
    
    #Pairwise distances of all points
    PairDistances = distance_matrix(PointsList,PointsList)

    np.fill_diagonal(PairDistances, np.inf)
    Dmin = 0
    GlobalMatchedList = []

    while Dmin < DminThresh:
        # print("yo")
        MinIndex = np.unravel_index(PairDistances.argmin(),PairDistances.shape )
        Dmin = PairDistances[MinIndex[0],MinIndex[1]]
        PointCamPairs = set(IndexList[MinIndex[0]]+ IndexList[MinIndex[1]])

        existSetIndex = set([x for x,Subset in enumerate(GlobalMatchedList) for pair in Subset if pair in PointCamPairs ])
        #indexes in global list where there is overlap cam index pair

        if len(existSetIndex)==0: ##no set already matched, create new set
            GlobalMatchedList.append(PointCamPairs)
        else: 
            if len(existSetIndex)>1:
                PairDistances[MinIndex[0],MinIndex[1]] = np.inf #remove the already matched pair
                continue
            
            MatchedSet = GlobalMatchedList[list(existSetIndex)[0]]
            PresentCamNames = [pair[0] for pair in PointCamPairs]
            for pair in MatchedSet:
                if pair[0] in PresentCamNames: ##cam already matched here
                    ##to find which one of the pairs shares cam name:
                    MatchedPair = [subPair for subPair in PointCamPairs if subPair[0] == pair[0]]
                    [PointCamPairs.discard(subPair) for subPair in MatchedPair]

            GlobalMatchedList[list(existSetIndex)[0]].update(PointCamPairs)
                    
        PairDistances[MinIndex[0],MinIndex[1]] = np.inf #remove the already matched pair

    # import ipdb;ipdb.set_trace()


    ##Convert Global Matching list back to dictionary format, with an arbitiary index for each individual

    FinalCamDict = {key:{} for key in CamNames}

    # import ipdb;ipdb.set_trace()
    #Go through each cluster
    for x in range(len(GlobalMatchedList)): 
        for cam in CamNames:
            IndexList = [pair[1] for pair in GlobalMatchedList[x] if pair[0] == cam]
            if len(IndexList) == 0:
                CamIndex = None
            else:
                CamIndex = IndexList[0]

            FinalCamDict[cam].update({x:CamIndex})
    # import ipdb;ipdb.set_trace()
    # FinalCamDict["Cam4"].values()
    return FinalCamDict



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
        # import ipdb;ipdb.set_trace()
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
    

    ###Visualize Tracks
    Points3DKeel = {key.split("_")[0]: val for key,val in Point3DDict.items() if "bp_bottomKeel" in key}

    IDs3D = list(Points3DKeel.keys())
    points3D = np.array(list(Points3DKeel.values()))
    Allimgpts, jac = cv2.projectPoints(points3D, VisCam.rvec, VisCam.tvec, VisCam.camMat, VisCam.distCoef)


    for j in range(Allimgpts.shape[0]):
        # import ipdb;ipdb.set_trace()
        
        Point = (round(Allimgpts[j][0][0])-30,round(Allimgpts[j][0][1])-70)
        if not IsPointValid(imsize, Point):
            continue
        ##Plot rectangle around text
        x,y = Point
        text = "ID: %s"%str(int(IDs3D[j]))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text_size, _ = cv2.getTextSize(text, font, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, Point, (x + text_w, y + text_h), [255,255,255], -1)

        frame = cv2.putText(frame,text,(x,y+text_h), font, 1, [255,0,0],1, cv2.LINE_AA)


    return frame
    

    
def VisualizeTracks(frame,TrackingOut):
    """Visualize SORT tracking output"""

    for x in range(TrackingOut.shape[0]):
        Point = (round(TrackingOut[x][0]),round(TrackingOut[x][1])-5)

        frame = cv2.putText(frame,"ID: %s" %str(int(TrackingOut[x][4])),Point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [255,0,0],1, cv2.LINE_AA)

    return frame

def Visualize3DTracks(frame,TrackerOut,VisCam):
    """Visualize 3D tracker tracking output"""

    points3D = np.array(list(TrackerOut.values()))
    IDs3D = list(TrackerOut.keys())
    Allimgpts, jac = cv2.projectPoints(points3D, VisCam.rvec, VisCam.tvec, VisCam.camMat, VisCam.distCoef)


    for j in range(Allimgpts.shape[0]):
        # import ipdb;ipdb.set_trace()
        
        Point = (round(Allimgpts[j][0][0])-30,round(Allimgpts[j][0][1])-70)

        ##Plot rectangle around text
        x,y = Point
        text = "ID: %s"%str(int(IDs3D[j]))
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text_size, _ = cv2.getTextSize(text, font, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, Point, (x + text_w, y + text_h), [255,255,255], -1)

        frame = cv2.putText(frame,text,(x,y+text_h), font, 1, [255,0,0],1, cv2.LINE_AA)

    return frame
    

def Inference3DPOP(model, SequenceNum,DatasetPath,device,startFrame = 0,TotalFrames = 1800,confidence_threshold=0.5,VisualizeIndex = 0):
    """Read a sequence from 3D pop then do inference with i-muppet + triangulate"""
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    
    VisCam = SequenceObj.camObjects[VisualizeIndex]
    
    # TotalFrames = len(VisCam.Keypoint2D.index)
    # TotalFrames = 250
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

    Tracker3DOutDict = {}

    ##Setup video capture objects
    capList = []
    for cam in SequenceObj.camObjects:    
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)
        
    # for cap in capList:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 

        
    imsize = (int(cap.get(3)),int(cap.get(4)))
    out = cv2.VideoWriter(filename="BarnTracking_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    Points3DDict = {}

    Sort_trackerList = [Sort(max_age = 10) for b in range(len(CamNames))]

    AllMatched = False
    Rematched = False
    GlobalMatchedDict = {cam:{} for cam in CamNames} #global matching dictionary

    ### 2D tracking result files
    Detection2DOutDict = {Cam:{} for Cam in CamNames}
    Tracking2DOutDict = {Cam:{} for Cam in CamNames}
    Points2DDict = {}


    for i in tqdm(range(TotalFrames)):

        PointsDict = {} #Points Dictionary for this frame

        FrameList = []
        BBoxList = []

        #read frames
        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)
        
        if ret == False: #end of vid
            break
        if SequenceNum == 59 and i<90:
            continue

        # if AllMatched == False:
        ##if correspondences not yet matched:
        MatchingDict = {}

        for x in range(len(SequenceObj.camObjects)):
            Img = FrameList[x].copy()
            Img= ProcessImage(Img,device)
            with torch.inference_mode():
                result = model([Img])[0]
            
            # import ipdb;ipdb.set_trace()

                
            # MatchDict = MatchID_BBox(GT_BBox,result,confidence_threshold=confidence_threshold)
            CamDict = {}
            
            Key2DPred = result["keypoints"].to("cpu").numpy()
            # import ipdb;ipdb.set_trace()
            ScoresList = result["scores"].to("cpu").numpy().tolist()
            boxesList = result["boxes"].to("cpu").numpy().tolist()
            
            # if AllMatched == False:
            ##Get top 10 (or N) individuals with highest scores 
            NumInd = len(SequenceObj.Subjects)
            Top10Index = sorted(range(len(ScoresList)), key=lambda i: ScoresList[i])[-NumInd:]

            FilteredBBox = [box for k,box in enumerate(boxesList) if k in Top10Index]
            FilteredKP = [pt for k,pt in enumerate(Key2DPred) if k in Top10Index]
            
            MatchingDict[CamNames[x]] = {"BBox":FilteredBBox, "Keypoints" : FilteredKP}
            BBoxList.append(FilteredBBox)


            FilteredScores = [score for k,score in enumerate(ScoresList) if k in Top10Index]
            ##combine bbox with the score
            # import ipdb;ipdb.set_trace()

            CombinedBBoxScores = [box + [score] for box,score in zip(FilteredBBox,FilteredScores)]

            Detection2DOutDict[CamNames[x]][i] = CombinedBBoxScores



        ##Tracking
        TrackingOutList = []
        for x in range(len(CamNames)):
            TrackArray = np.array(BBoxList[x])
            TrackingOut = Sort_trackerList[x].update(TrackArray)
            MatchingDict[CamNames[x]]["TrackedBBox"] = TrackingOut
            TrackingOutList.append(TrackingOut)
            # import ipdb;ipdb.set_trace()
            Tracking2DOutDict[CamNames[x]][i] = TrackingOut

        if Rematched == False: #first frame or rematching, only one loop
            # import ipdb;ipdb.set_trace()
            MatchDict = MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,confidence_threshold=confidence_threshold)
            # import ipdb;ipdb.set_trace()
            for x, cam in enumerate(CamNames):
                # import ipdb;ipdb.set_trace()

                for k,v in MatchDict[cam].items():
                    if v == None:
                        continue
                    ###Check if global id is already in global matched dict
                    CurrentBoxSum = round(np.array(MatchingDict[cam]["BBox"][v]).sum())
                    CurrentBBoxSumDistance = [abs(sum(TrackingOutList[x][y,:4])-CurrentBoxSum) for y in range(len(TrackingOutList[x]))]
                    CurrentIndex = CurrentBBoxSumDistance.index(min(CurrentBBoxSumDistance))
                    CurrentSortIndex = TrackingOutList[x][CurrentIndex,4]
                    # import ipdb;ipdb.set_trace()
                    # SortIndex = TrackingOutList[x][v][4]
                    GlobalMatchedDict[cam].update({k:CurrentSortIndex})

            # import ipdb;ipdb.set_trace()
            AssingedSum = sum([len(v) for v in GlobalMatchedDict.values()])
            if AssingedSum == len(CamNames)*NumInd:
                AllMatched = True

            Rematched = True

        if AllMatched == False: #all matches not found yet
            # import ipdb;ipdb.set_trace()
            MatchDict = MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,confidence_threshold=confidence_threshold)
            # import ipdb;ipdb.set_trace()
            for x, cam in enumerate(CamNames):
                # import ipdb;ipdb.set_trace()

                for k,v in MatchDict[cam].items():
                    if v == None:
                        continue
                    ###Check if global id is already in global matched dict
                    CurrentBoxSum = round(np.array(MatchingDict[cam]["BBox"][v]).sum())
                    CurrentBBoxSumDistance = [abs(sum(TrackingOutList[x][y,:4])-CurrentBoxSum) for y in range(len(TrackingOutList[x]))]
                    CurrentIndex = CurrentBBoxSumDistance.index(min(CurrentBBoxSumDistance))
                    CurrentSortIndex = TrackingOutList[x][CurrentIndex,4]
                
                    if CurrentSortIndex in GlobalMatchedDict[cam].values():
                        continue
                    else:
                        ##find missing global indicies:
                        MissingGlobal = list(set(range(NumInd)).difference(list(GlobalMatchedDict[cam].keys())))
                        ##Get global bbox from cam 1:
                        MatchedLocalIndexList = []
                        for MissedIndex in MissingGlobal:
                            MissedSortIndex = GlobalMatchedDict[CamNames[1]][MissedIndex] #for cam 1, which sort index it is
                            BBoxSum = TrackingOutList[1][TrackingOutList[1][:,4].tolist().index(MissedSortIndex),:4].sum() #get sum of the index from cam 1
                            ##find local index from cam 1:
                            BBoxSumDistance = [abs(sum(MatchingDict[CamNames[1]]["BBox"][y])-BBoxSum) for y in range(len(TrackingOutList[x]))]
                            LocalIndex = BBoxSumDistance.index(min(BBoxSumDistance))
                            MatchedLocalIndex = [key1 for key1,val1 in MatchDict[CamNames[1]].items() if val1 == LocalIndex][0]
                            MatchedLocalIndexList.append(MatchedLocalIndex)
                        # import ipdb;ipdb.set_trace()

                        if k in MatchedLocalIndexList: #found index for this individual in this frame
                            GlobalIndex = MissingGlobal[MatchedLocalIndexList.index(k)]
                            GlobalMatchedDict[cam].update({GlobalIndex:CurrentSortIndex})

                        else:
                            continue
                    # import ipdb;ipdb.set_trace()
                    # SortIndex = TrackingOutList[x][v][4]

            # import ipdb;ipdb.set_trace()
            AssingedSum = sum([len(v) for v in GlobalMatchedDict.values()])
            if AssingedSum == len(CamNames)*NumInd:
                AllMatched = True

        #Get Corresponding 2D Points based on global matching dict:
        for x, cam in enumerate(CamNames):
            CamDict = {}

            for key, val in GlobalMatchedDict[cam].items():
                try:
                    # import ipdb;ipdb.set_trace()
                    TrackIndex = TrackingOutList[x][:,4].tolist().index(val)
                    TrackedBoxesSum = round(TrackingOutList[x][TrackIndex,:4].sum())
                    RealBBoxSumDistance = [abs(sum(MatchingDict[cam]["BBox"][y])-TrackedBoxesSum) for y in range(len(MatchingDict[cam]["BBox"]))]

                    #attempt to match bbox of tracked and predicted, summed the values and compared diff
                    RealBBoxIndex = RealBBoxSumDistance.index(min(RealBBoxSumDistance))
                    # import ipdb;ipdb.set_trace()

                except:
                    print("skipped a cam")
                    continue
                Key2D = MatchingDict[cam]["Keypoints"][RealBBoxIndex]
        
                CamDict.update({"%s_%s"%(key,PIGEON_KEYPOINT_NAMES[j]):[Key2D[j,0],Key2D[j,1]] for j in range(Key2D.shape[0])})    
        
            PointsDict.update({cam:CamDict})
        # import ipdb;ipdb.set_trace()
        Points2DDict[i] = PointsDict.copy()
        TriangTool = BundleAdjustmentTool_Triangulation_Filter(CamNames,CamParamDict)
        TriangTool.PrepareInputData(PointsDict)
        Point3DDict = TriangTool.run()
        Points3DDict[i] = Point3DDict.copy()


        ##Do tracking in 3D:
        Points3D = {key.split("_")[0]: val for key,val in Point3DDict.items() if "bp_bottomKeel" in key}

        Tracker3DOutDict[i] = Points3D.copy()
        # import ipdb;ipdb.set_trace()

        ##Visualize
        frame = FrameList[VisualizeIndex]
        Boxes = BBoxList[VisualizeIndex]

        # import ipdb;ipdb.set_trace()
        SubjectList = [str(n) for n in range((NumInd))]
        frame = VizualizeAll(frame, counter,VisCam,Boxes,Point3DDict,imsize,SubjectList)
        # frame = Visualize3DTracks(frame,TrackerOut,VisCam)

        cv2.imshow('Window',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # torch.cuda.empty_cache()
        out.write(frame)
        counter += 1
        # cv2.imwrite(img=frame, filename="Sample3D.jpg")
        # print(GlobalMatchedDict)
        
    out.release()

    return Points3DDict, Tracker3DOutDict, Detection2DOutDict, Tracking2DOutDict,Points2DDict


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
    Inference3DPOP(network, SequenceNum,DatasetPath,device,startFrame = 0,TotalFrames=1800,VisualizeIndex = 0)

