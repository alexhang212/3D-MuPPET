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

import Network_utils
from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation_Filter
from tqdm import tqdm
import pickle

from EvalUtils import GetEucDist, MatchingAlgorithm

sys.path.append("Repositories/sort/")
from sort import *

PIGEON_KEYPOINT_NAMES = ['hd_beak', 'hd_leftEye', 'hd_rightEye', 'hd_nose', 'bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail']
KEEL_INDEX = 7


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


def Inference3DPOP(model, SequenceNum,DatasetPath,device,confidence_threshold=0.5):
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")

    TotalFrames = 250
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

    Points3DDict = {}
    Tracker3DOutDict = {}

    capList = []
    for cam in SequenceObj.camObjects:
        cap = cv2.VideoCapture(cam.VideoPath)
        capList.append(cap)

    Sort_trackerList = [Sort(max_age = 10) for b in range(len(CamNames))]

    AllMatched = False
    Rematched = False
    GlobalMatchedDict = {cam:{} for cam in CamNames}

    Detection2DOutDict = {Cam:{} for Cam in CamNames}
    Tracking2DOutDict = {Cam:{} for Cam in CamNames}
    Points2DDict = {}

    for i in tqdm(range(TotalFrames)):

        PointsDict = {}
        FrameList = []
        BBoxList = []

        for cap in capList:
            ret, frame = cap.read()
            FrameList.append(frame)

        if ret == False:
            break
        if SequenceNum == 59 and i<90:
            continue

        MatchingDict = {}

        for x in range(len(SequenceObj.camObjects)):
            Img = FrameList[x].copy()
            Img = ProcessImage(Img,device)
            with torch.inference_mode():
                result = model([Img])[0]

            Key2DPred = result["keypoints"].to("cpu").numpy()
            ScoresList = result["scores"].to("cpu").numpy().tolist()
            boxesList = result["boxes"].to("cpu").numpy().tolist()

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
                            MissedSortIndex = GlobalMatchedDict[CamNames[1]][MissedIndex]
                            BBoxSum = TrackingOutList[1][TrackingOutList[1][:,4].tolist().index(MissedSortIndex),:4].sum()
                            BBoxSumDistance = [abs(sum(MatchingDict[CamNames[1]]["BBox"][y])-BBoxSum) for y in range(len(TrackingOutList[x]))]
                            LocalIndex = BBoxSumDistance.index(min(BBoxSumDistance))
                            MatchedLocalIndex = [key1 for key1,val1 in MatchDict[CamNames[1]].items() if val1 == LocalIndex][0]
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/media/alexchan/WD Elements AE/Pop3D-Dataset", help="Path to the Pop3D dataset")
    args = parser.parse_args()

    AllSequences = [1,2,5,11]
    for SequenceNum in AllSequences:
        OutDir = "EvaluationData"
        ModelName = "KPRCNN"

        DatasetPath = args.dataset_path
        WeightsPath = "Weights/KPRCNN_3DPOP_Best.pt"

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        network = LoadNetwork(WeightsPath,device)
        Points3DDict,Tracker3DOutDict,Detection2DOutDict, Tracking2DOutDict,Points2DDict = Inference3DPOP(network, SequenceNum,DatasetPath,device)

        pickle.dump(Points3DDict,open(os.path.join(OutDir,"./SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Tracker3DOutDict,open(os.path.join(OutDir,"./SeqEval_3DTracker_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))

        pickle.dump(Detection2DOutDict,open(os.path.join(OutDir,"./SeqEval_Detection2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Tracking2DOutDict,open(os.path.join(OutDir,"./SeqEval_Tracking2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
        pickle.dump(Points2DDict,open(os.path.join(OutDir,"SeqEval_Points2D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
