import torch
import cv2
import numpy as np
import os
import sys
import argparse

sys.path.append("Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

sys.path.append("./")
sys.path.append("Utils")

from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation_Filter
from tqdm import tqdm
import pickle

from EvalUtils import GetEucDist, MatchingAlgorithm
from ultralytics import YOLO

sys.path.append("Repositories/sort/")
from sort import *

sys.path.append("Repositories/DeepLabCut/")
sys.path.append("Repositories/DeepLabCut-live")

from dlclive import DLCLive, Processor

PIGEON_KEYPOINT_NAMES = ['bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail','hd_beak', 'hd_nose','hd_leftEye', 'hd_rightEye']
KEEL_INDEX = 3


def RunYOLOLoop(YOLOModel, DatasetPath,SequenceNum,startFrame,TotalFrames,ScaleBBox):
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")

    counter = startFrame

    CamNames = []
    for cam in SequenceObj.camObjects:
        CamNames.append(cam.CamName)

    capList = []
    for cam in SequenceObj.camObjects:
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)

    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    OutbboxList = []
    OutConfList = []

    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(TotalFrames), desc = "Running YOLO:"):
        FrameList = []
        BBoxList = []
        ConfidenceList = []
        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)

        if ret == False:
            break

        for x in range(len(SequenceObj.camObjects)):
            InferFrame = FrameList[x].copy()
            results = YOLOModel(InferFrame, imgsz=3840, verbose = False)
            classID = [key for key,val in results[0].names.items() if val == "bird"][0]
            DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()

            bbox = results[0].boxes.xywh.cpu().numpy().tolist()
            bbox = [box for x,box in enumerate(bbox) if DetectedClasses[x] == classID]
            bbox = [[box[0],box[1],box[2]*ScaleBBox,box[3]*ScaleBBox] for box in bbox]
            bbox = [[box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] for box in bbox]
            BBoxList.append(bbox)

            ConfList = results[0].boxes.conf.cpu().numpy().tolist()
            ConfidenceList.append(ConfList)

        OutbboxList.append(BBoxList)
        OutConfList.append(ConfidenceList)
        counter += 1

    return OutbboxList, OutConfList


def DLCInference(InferFrame,box,dlc_liveObj,CropSize):
    box = [0 if val < 0 else val for val in box]

    Crop = InferFrame[round(box[1]):round(box[3]),round(box[0]):round(box[2])]

    if dlc_liveObj.sess == None:
        DLCPredict2D = dlc_liveObj.init_inference(Crop)

    DLCPredict2D= dlc_liveObj.get_pose(Crop)

    DLCPredict2DList = [[DLCPredict2D[j,0]+box[0],DLCPredict2D[j,1]+box[1]] for j in range(DLCPredict2D.shape[0])]

    return DLCPredict2DList


def RunDLCLoop(dlc_liveObj,OutbboxList,OutConfList,SequenceNum,DatasetPath,CropSize,startFrame,TotalFrames):
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")

    NumInd = len(SequenceObj.Subjects)

    CamParamDict = {}
    CamNames = []
    for cam in SequenceObj.camObjects:
        CamParamDict.update({cam.CamName:{
            "R":cam.rvec,
            "T":cam.tvec,
            "cameraMatrix":cam.camMat,
            "distCoeffs":cam.distCoef
        }})
        CamNames.append(cam.CamName)

    counter = startFrame

    Tracker3DOutDict = {}

    capList = []
    for cam in SequenceObj.camObjects:
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)

    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for cap in capList:
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    Points3DDict = {}

    Sort_trackerList = [Sort(max_age = 10) for b in range(len(CamNames))]

    AllMatched = False
    Rematched = False
    GlobalMatchedDict = {cam:{} for cam in CamNames}

    Detection2DOutDict = {Cam:{} for Cam in CamNames}
    Tracking2DOutDict = {Cam:{} for Cam in CamNames}
    Points2DDict = {}

    for i in tqdm(range(len(OutbboxList))):

        PointsDict = {}
        FrameList = []
        BBoxList = []

        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)

        if ret == False:
            break

        MatchingDict = {}

        for x in range(len(SequenceObj.camObjects)):
            Img = FrameList[x].copy()
            boxesList = OutbboxList[i][x]
            ScoresList = OutConfList[i][x]

            Key2DPredList = []
            for box in boxesList:
                DLCPredict2D= DLCInference(Img,box,dlc_liveObj,CropSize)
                Key2DPredList.append(DLCPredict2D)

            Key2DPred = np.array(Key2DPredList)

            NumInd = len(SequenceObj.Subjects)
            Top10Index = sorted(range(len(ScoresList)), key=lambda i: ScoresList[i])[-NumInd:]

            FilteredBBox = [box for k,box in enumerate(boxesList) if k in Top10Index]
            FilteredKP = [pt for k,pt in enumerate(Key2DPred) if k in Top10Index]

            MatchingDict[CamNames[x]] = {"BBox":FilteredBBox, "Keypoints" : FilteredKP}
            BBoxList.append(FilteredBBox)

            FilteredScores = [score for k,score in enumerate(ScoresList) if k in Top10Index]
            CombinedBBoxScores = [box + [score] for box,score in zip(FilteredBBox,FilteredScores)]
            Detection2DOutDict[CamNames[x]][i] = CombinedBBoxScores

        TrackingOutList = []
        for x in range(len(CamNames)):
            TrackArray = np.array(BBoxList[x])
            if len(TrackArray) == 0:
                Tracking2DOutDict[CamNames[x]][i] = np.nan
                TrackingOutList.append(np.nan)
                continue
            TrackingOut = Sort_trackerList[x].update(TrackArray)
            MatchingDict[CamNames[x]]["TrackedBBox"] = TrackingOut
            TrackingOutList.append(TrackingOut)
            Tracking2DOutDict[CamNames[x]][i] = TrackingOut

        if Rematched == False:
            MatchDict = MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,PIGEON_KEYPOINT_NAMES,KEEL_INDEX)
            for x, cam in enumerate(CamNames):
                for k,v in MatchDict[cam].items():
                    if v == None:
                        continue
                    CurrentBoxSum = round(np.array(MatchingDict[cam]["BBox"][v]).sum())
                    CurrentBBoxSumDistance = [abs(sum(TrackingOutList[x][y,:4])-CurrentBoxSum) for y in range(len(TrackingOutList[x]))]
                    CurrentIndex = CurrentBBoxSumDistance.index(min(CurrentBBoxSumDistance))
                    CurrentSortIndex = TrackingOutList[x][CurrentIndex,4]
                    GlobalMatchedDict[cam].update({k:CurrentSortIndex})

            AssingedSum = sum([len(v) for v in GlobalMatchedDict.values()])
            if AssingedSum == len(CamNames)*NumInd:
                AllMatched = True

            Rematched = True

        if AllMatched == False:
            MatchDict = MatchingAlgorithm(MatchingDict,CamNames,CamParamDict,PIGEON_KEYPOINT_NAMES,KEEL_INDEX)

            if len(MatchDict) > 0:
                for x, cam in enumerate(CamNames):
                    for k,v in MatchDict[cam].items():
                        if v == None:
                            continue
                        CurrentBoxSum = round(np.array(MatchingDict[cam]["BBox"][v]).sum())
                        CurrentBBoxSumDistance = [abs(sum(TrackingOutList[x][y,:4])-CurrentBoxSum) for y in range(len(TrackingOutList[x]))]
                        CurrentIndex = CurrentBBoxSumDistance.index(min(CurrentBBoxSumDistance))
                        CurrentSortIndex = TrackingOutList[x][CurrentIndex,4]

                        if CurrentSortIndex in GlobalMatchedDict[cam].values():
                            continue
                        else:
                            MissingGlobal = list(set(range(NumInd)).difference(list(GlobalMatchedDict[cam].keys())))
                            MatchedLocalIndexList = []
                            for MissedIndex in MissingGlobal:
                                MissedSortIndex = GlobalMatchedDict[CamNames[0]][MissedIndex]
                                BBoxSum = TrackingOutList[0][TrackingOutList[0][:,4].tolist().index(MissedSortIndex),:4].sum()
                                BBoxSumDistance = [abs(sum(MatchingDict[CamNames[0]]["BBox"][y])-BBoxSum) for y in range(len(TrackingOutList[x]))]
                                LocalIndex = BBoxSumDistance.index(min(BBoxSumDistance))
                                MatchedLocalIndex = [key1 for key1,val1 in MatchDict[CamNames[0]].items() if val1 == LocalIndex][0]
                                MatchedLocalIndexList.append(MatchedLocalIndex)

                            if k in MatchedLocalIndexList:
                                GlobalIndex = MissingGlobal[MatchedLocalIndexList.index(k)]
                                GlobalMatchedDict[cam].update({GlobalIndex:CurrentSortIndex})
                            else:
                                continue

                AssingedSum = sum([len(v) for v in GlobalMatchedDict.values()])
                if AssingedSum == len(CamNames)*NumInd:
                    AllMatched = True

        for x, cam in enumerate(CamNames):
            CamDict = {}

            for key, val in GlobalMatchedDict[cam].items():
                try:
                    TrackIndex = TrackingOutList[x][:,4].tolist().index(val)
                    TrackedBoxesSum = round(TrackingOutList[x][TrackIndex,:4].sum())
                    RealBBoxSumDistance = [abs(sum(MatchingDict[cam]["BBox"][y])-TrackedBoxesSum) for y in range(len(MatchingDict[cam]["BBox"]))]
                    RealBBoxIndex = RealBBoxSumDistance.index(min(RealBBoxSumDistance))
                except:
                    print("skipped a cam")
                    continue
                Key2D = MatchingDict[cam]["Keypoints"][RealBBoxIndex]
                CamDict.update({"%s_%s"%(key,PIGEON_KEYPOINT_NAMES[j]):[Key2D[j,0],Key2D[j,1]] for j in range(Key2D.shape[0])})

            PointsDict.update({cam:CamDict})

        Points2DDict[i] = PointsDict.copy()

        TriangTool = BundleAdjustmentTool_Triangulation_Filter(CamNames,CamParamDict)
        TriangTool.PrepareInputData(PointsDict)
        Point3DDict = TriangTool.run()
        Points3DDict[i] = Point3DDict.copy()

        Points3D = {key.split("_")[0]: val for key,val in Point3DDict.items() if "bp_bottomKeel" in key}
        Tracker3DOutDict[i] = Points3D.copy()

    return Points3DDict, Tracker3DOutDict, Detection2DOutDict, Tracking2DOutDict, Points2DDict


def RunInference3DPOP(YOLOModel,dlc_liveObj,SequenceNum,DatasetPath,CropSize,startFrame=0,TotalFrames = 1800,ScaleBBox=1):
    OutbboxList,OutConfList = RunYOLOLoop(YOLOModel, DatasetPath,SequenceNum,startFrame,TotalFrames,ScaleBBox)
    del YOLOModel
    torch.cuda.empty_cache()
    Points3DDict,Tracker3DOutDict,Detection2DOutDict, Tracking2DOutDict,Points2DDict = RunDLCLoop(dlc_liveObj,OutbboxList,OutConfList,SequenceNum,DatasetPath,CropSize,startFrame,TotalFrames)

    return Points3DDict,Tracker3DOutDict,Detection2DOutDict, Tracking2DOutDict,Points2DDict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/media/alexchan/WD Elements AE/Pop3D-Dataset", help="Path to the Pop3D dataset")
    args = parser.parse_args()

    AllSequences = [11,1,2,5]
    for SequenceNum in AllSequences:
        print("Sequence: %s"%SequenceNum)

        OutDir = "EvaluationData"
        ModelName = "YOLODLC"

        if os.path.exists(os.path.join(OutDir,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SequenceNum))):
            continue

        YOLOPath = "Weights/YOLO_Barn.pt"
        DatasetPath = args.dataset_path

        ExportModelPath = "Weights/DLC_Barn"
        CropSize = (320,320)

        YOLOModel = YOLO(YOLOPath)

        dlc_proc = Processor()
        dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)

        Points3DDict,Tracker3DOutDict,Detection2DOutDict, Tracking2DOutDict,Points2DDict = RunInference3DPOP(YOLOModel,dlc_liveObj,SequenceNum,DatasetPath,CropSize,
                                                                                                              startFrame=0,TotalFrames = 250,ScaleBBox=1)

        pickle.dump(Points3DDict,open(os.path.join(OutDir,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Tracker3DOutDict,open(os.path.join(OutDir,"SeqEval_3DTracker_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))

        pickle.dump(Detection2DOutDict,open(os.path.join(OutDir,"SeqEval_Detection2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Tracking2DOutDict,open(os.path.join(OutDir,"SeqEval_Tracking2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Points2DDict,open(os.path.join(OutDir,"SeqEval_Points2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
