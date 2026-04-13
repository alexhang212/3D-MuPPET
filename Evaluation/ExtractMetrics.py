import numpy as np
import argparse

import sys
sys.path.append("./")
sys.path.append("Utils")

import HungarianAlgorithm

import pickle
import math
from tqdm import tqdm
import itertools
from glob import glob
import os
from scipy.spatial.distance import cdist
import statistics
import pandas as pd
from Utils import Head_Angle_Eval
from EvalUtils import GetEucDist

sys.path.append("Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

PIGEON_KEYPOINT_NAMES = ["hd_beak","hd_nose","hd_leftEye","hd_rightEye","bp_leftShoulder","bp_rightShoulder","bp_topKeel","bp_bottomKeel","bp_tail"]

def GetPCK(PointDist,MaxDist):
    PCK10 = 0
    PCK05 = 0

    PercentageofMax = PointDist/MaxDist

    if PercentageofMax < 0.1:
        PCK10 = 1

    if PercentageofMax < 0.05:
        PCK05 = 1

    return PCK10, PCK05

def GetMedian(ErrorList):
    ErrorList = [x for x in ErrorList if x == x]
    Out = statistics.median(ErrorList)
    return Out


def GetPCKSum(PCKList):
    PCKList = [x for x in PCKList if x == x]
    return (sum(PCKList)/len(PCKList))*100


def GetRMSE(ErrorList):
    ErrorList = [x for x in ErrorList if x == x]
    Out = np.sqrt(np.mean(np.array(ErrorList)**2))
    return Out

def MatchID(SequenceObj,CamObj, Predictions, counter = None):
    if counter == None:
        counter = 0

    MatchedDict = {}

    while True:
        if counter not in Predictions:
            counter +=1
            continue

        FramePred = Predictions[counter]
        GTDict = {}
        for bird in SequenceObj.Subjects:
            GTDict[bird] = list(CamObj.Read3DKeypointData(CamObj.Keypoint3D, counter, bird, Keypoints = ["bp_bottomKeel"]).values())[0]
        if np.isnan(list(GTDict.values())).any():
            counter += 1
            continue

        try:
            PredDict = {k.split("_")[0]:v.tolist() for k,v in FramePred.items() if "bp_bottomKeel" in k}
        except:
            counter +=1
            continue

        PredNP = np.array(list(PredDict.values()))
        GTNP = np.array(list(GTDict.values()))

        DistanceMatrix = cdist(GTNP, PredNP)

        Matches = HungarianAlgorithm.hungarian_algorithm(DistanceMatrix)
        BirdIDs = list(GTDict.keys())
        PredIDs = list(PredDict.keys())

        MatchedDict = {}
        for match in Matches:
            MatchedDict[BirdIDs[match[0]]] = PredIDs[match[1]]

        break

    return MatchedDict

def RMSESummaryDict(RMSEDictList,Keypoints, filter = False):
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    IndividualFilterCounter = 0
    TotalIndividualsCounter = 0

    for i in range(len(RMSEDictList)):
        FrameDict = RMSEDictList[i]

        for PointsDict in FrameDict.values():
            TotalIndividualsCounter += 1
            MeanVal = np.array(list(PointsDict.values())).mean()
            if filter:
                if MeanVal > filter:
                    IndividualFilterCounter += 1
                    continue

            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)

    print(IndividualFilterCounter)
    print("Total Individuals: %s"%TotalIndividualsCounter)

    return PerKeypointDict, AllPointsList


def AngleSummaryDict(AngleDictList, filter = False):
    Types = ["yaw","pitch","roll"]

    PerDimDict = {}
    for key in Types:
        PerDimDict[key] = []

    AllAngleList = []
    IndividualFilterCounter = 0
    TotalIndividualsCounter = 0

    for i in range(len(AngleDictList)):
        FrameDict = AngleDictList[i]

        for PointsDict in FrameDict.values():
            TotalIndividualsCounter += 1
            MeanVal = np.array(list(PointsDict.values())).mean()

            for k,v in PointsDict.items():
                PerDimDict[k].append(abs(v))
                AllAngleList.append(abs(v))

    return PerDimDict, AllAngleList


def DoEval3D(DatasetPath, SeqNum,Predictions3D):
    SequenceObj = Trial.Trial(DatasetPath,SeqNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    CamObj = SequenceObj.camObjects[0]
    FrameNums = Predictions3D.keys()

    MatchedDict = {sub:sub for sub in SequenceObj.Subjects}

    EucErrorList3D = []
    PCK05List3D = []
    PCK10List3D = []
    AngleList3D = []

    for i in tqdm(FrameNums):

        FramePred3D = Predictions3D[i]
        EucErrorDict3D = {}
        PCK05Dict3D = {}
        PCK10Dict3D = {}
        AngleDict3D = {}

        for GTID, PredID in MatchedDict.items():
            Bird3DGT = CamObj.Read3DKeypointData(CamObj.Keypoint3D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)
            Bird3DPred = {"_".join(k.split("_")[2:4]):v for k,v in FramePred3D.items() if k.startswith(PredID)}

            DistList = []
            for pair in itertools.product(list(Bird3DGT.values()),repeat=2):
                DistList.append(GetEucDist(pair[0],pair[1]))

            MaxDist = max(DistList)
            BirdEucErrorDict3D = {}
            BirdPCK10Dict3D = {}
            BirdPCK05Dict3D = {}

            for kp in PIGEON_KEYPOINT_NAMES:
                if np.isnan(np.array(list(Bird3DGT.values()))).any():
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                elif kp not in Bird3DPred:
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                else:
                    GTval = Bird3DGT[kp]
                    PredVal = Bird3DPred[kp]
                    if np.isnan(PredVal).any() or np.isnan(GTval).any():
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        PointDist = GetEucDist(GTval,PredVal)
                        PCK10,PCK05 = GetPCK(PointDist,MaxDist)
                    if PointDist > 100:
                        print(i)

                BirdEucErrorDict3D[kp] = PointDist
                BirdPCK10Dict3D[kp] = PCK10
                BirdPCK05Dict3D[kp] = PCK05

            EucErrorDict3D[GTID] = BirdEucErrorDict3D
            PCK10Dict3D[GTID] = BirdPCK10Dict3D
            PCK05Dict3D[GTID] = BirdPCK05Dict3D

            GT_R, GT_T = Head_Angle_Eval.DefineObj(Bird3DGT)
            Pred_R, Pred_T = Head_Angle_Eval.DefineObj(Bird3DPred)

            if GT_R is np.nan or Pred_R is np.nan:
                continue

            IdentityMat = np.identity(3)
            TranslationOrigin = np.array([0,0,0]).reshape(3,1)
            Inv_GT_R, Inv_GT_T = Head_Angle_Eval.computeExtrinsic(IdentityMat, TranslationOrigin,GT_R, np.array(GT_T).reshape(3,1))
            Inv_Pred_R, Inv_Pred_T = Head_Angle_Eval.computeExtrinsic( IdentityMat, TranslationOrigin,Pred_R, np.array(Pred_T).reshape(3,1))

            DiffR, DiffT = Head_Angle_Eval.computeExtrinsic(Inv_GT_R,Inv_GT_T,Inv_Pred_R , Inv_Pred_T)

            DiffEuler =  Head_Angle_Eval.rotationMatrixToEulerAngles(DiffR)
            DiffEulerDeg = [math.degrees(angle) for angle in DiffEuler]
            AngleDict3D[GTID] = {"pitch":DiffEulerDeg[0],"roll":DiffEulerDeg[1],"yaw":DiffEulerDeg[2]}

        EucErrorList3D.append(EucErrorDict3D)
        PCK10List3D.append(PCK10Dict3D)
        PCK05List3D.append(PCK05Dict3D)
        AngleList3D.append(AngleDict3D)

    return EucErrorList3D,PCK10List3D,PCK05List3D,AngleList3D


def DoEval(DatasetPath, SeqNum,Predictions3D,Predictions2D):
    SequenceObj = Trial.Trial(DatasetPath,SeqNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    CamObj = SequenceObj.camObjects[0]
    FrameNums = Predictions3D.keys()

    MatchedDict = MatchID(SequenceObj,CamObj, Predictions3D)
    EucErrorList3D = []
    PCK05List3D = []
    PCK10List3D = []
    EucErrorList2D = []
    PCK05List2D = []
    PCK10List2D = []
    AngleList3D = []

    for i in tqdm(FrameNums):
        FramePred3D = Predictions3D[i]
        FramePred2D = Predictions2D[i]
        EucErrorDict3D = {}
        PCK05Dict3D = {}
        PCK10Dict3D = {}
        EucErrorDict2D = {}
        PCK05Dict2D = {}
        PCK10Dict2D = {}
        AngleDict3D = {}

        for GTID, PredID in MatchedDict.items():

            Bird3DGT = CamObj.Read3DKeypointData(CamObj.Keypoint3D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)
            Bird3DPred = {"_".join(k.split("_")[1:3]):v for k,v in FramePred3D.items() if k.startswith(PredID)}

            DistList = []
            for pair in itertools.product(list(Bird3DGT.values()),repeat=2):
                DistList.append(GetEucDist(pair[0],pair[1]))

            MaxDist = max(DistList)
            BirdEucErrorDict3D = {}
            BirdPCK10Dict3D = {}
            BirdPCK05Dict3D = {}
            AngleDict3D = {}

            for kp in PIGEON_KEYPOINT_NAMES:

                if kp not in Bird3DPred:
                    PointDist = np.nan
                    PCK10 = np.nan
                    PCK05 = np.nan
                else:
                    GTval = Bird3DGT[kp]
                    PredVal = Bird3DPred[kp]
                    if np.isnan(PredVal).any() or np.isnan(GTval).any():
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        PointDist = GetEucDist(GTval,PredVal)
                        PCK10,PCK05 = GetPCK(PointDist,MaxDist)

                BirdEucErrorDict3D[kp] = PointDist
                BirdPCK10Dict3D[kp] = PCK10
                BirdPCK05Dict3D[kp] = PCK05

            for camObj in SequenceObj.camObjects:
                CamName = camObj.CamName
                if CamName not in FramePred2D:
                    continue

                Bird2DGT = camObj.Read2DKeypointData(camObj.Keypoint2D, i, GTID,Keypoints = PIGEON_KEYPOINT_NAMES,StripName=True)

                Bird2DPred = {"_".join(k.split("_")[1:3]):v for k,v in FramePred2D[CamName].items() if k.startswith(PredID)}
                Bird2DBBox = camObj.GetBBoxData(camObj.BBox ,i,GTID )

                MaxDist = max([Bird2DBBox[1][0]-Bird2DBBox[0][0], Bird2DBBox[1][1]-Bird2DBBox[0][1]])
                BirdEucErrorDict2D = {}
                BirdPCK10Dict2D = {}
                BirdPCK05Dict2D = {}

                for kp in PIGEON_KEYPOINT_NAMES:
                    if kp not in Bird2DPred:
                        PointDist = np.nan
                        PCK10 = np.nan
                        PCK05 = np.nan
                    else:
                        GTval = Bird2DGT[kp]
                        PredVal = Bird2DPred[kp]
                        if np.isnan(PredVal).any() or np.isnan(GTval).any():
                            PointDist = np.nan
                            PCK10 = np.nan
                            PCK05 = np.nan
                        else:
                            PointDist = GetEucDist(GTval,PredVal)
                            PCK10,PCK05 = GetPCK(PointDist,MaxDist)

                    BirdEucErrorDict2D[kp] = PointDist
                    BirdPCK10Dict2D[kp] = PCK10
                    BirdPCK05Dict2D[kp] = PCK05

                EucErrorDict2D["%s_%s"%(CamName,GTID)] = BirdEucErrorDict2D
                PCK10Dict2D["%s_%s"%(CamName,GTID)] = BirdPCK10Dict2D
                PCK05Dict2D["%s_%s"%(CamName,GTID)] = BirdPCK05Dict2D

            EucErrorDict3D[GTID] = BirdEucErrorDict3D
            PCK10Dict3D[GTID] = BirdPCK10Dict3D
            PCK05Dict3D[GTID] = BirdPCK05Dict3D

            GT_R, GT_T = Head_Angle_Eval.DefineObj(Bird3DGT)
            Pred_R, Pred_T = Head_Angle_Eval.DefineObj(Bird3DPred)

            if GT_R is np.nan or Pred_R is np.nan:
                continue

            IdentityMat = np.identity(3)
            TranslationOrigin = np.array([0,0,0]).reshape(3,1)
            Inv_GT_R, Inv_GT_T = Head_Angle_Eval.computeExtrinsic(IdentityMat, TranslationOrigin,GT_R, np.array(GT_T).reshape(3,1))
            Inv_Pred_R, Inv_Pred_T = Head_Angle_Eval.computeExtrinsic( IdentityMat, TranslationOrigin,Pred_R, np.array(Pred_T).reshape(3,1))

            DiffR, DiffT = Head_Angle_Eval.computeExtrinsic(Inv_GT_R,Inv_GT_T,Inv_Pred_R , Inv_Pred_T)

            DiffEuler =  Head_Angle_Eval.rotationMatrixToEulerAngles(DiffR)
            DiffEulerDeg = [math.degrees(angle) for angle in DiffEuler]
            AngleDict3D[GTID] = {"pitch":DiffEulerDeg[0],"roll":DiffEulerDeg[1],"yaw":DiffEulerDeg[2]}

        EucErrorList3D.append(EucErrorDict3D)
        PCK10List3D.append(PCK10Dict3D)
        PCK05List3D.append(PCK05Dict3D)
        AngleList3D.append(AngleDict3D)

        EucErrorList2D.append(EucErrorDict2D)
        PCK10List2D.append(PCK10Dict2D)
        PCK05List2D.append(PCK05Dict2D)

    return EucErrorList3D,PCK10List3D,PCK05List3D, EucErrorList2D, PCK10List2D,PCK05List2D,AngleList3D

def PCKSummaryDict(PCKDictList, Keypoints):
    PerKeypointDict = {}
    for key in Keypoints:
        PerKeypointDict[key] = []

    AllPointsList = []
    for i in range(len(PCKDictList)):
        FrameDict = PCKDictList[i]

        for PointsDict in FrameDict.values():
            for k,v in PointsDict.items():
                PerKeypointDict[k].append(v)
                AllPointsList.append(v)

    return PerKeypointDict, AllPointsList

def GetSummaryCSV(Models,AllSequences, Type = "3D"):
    EucDFDict = {}
    PCK10DFDict = {}
    PCK05DFDict = {}
    MedianDFDict = {}
    AnglesDFDict = {}

    counter = 0

    for ModelName in Models:
        AllEucErrorList = []
        AllPCK10List = []
        AllPCK05List = []
        AllAngleList = []

        for SeqNum in AllSequences:
            if Type == "3D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"rb"))
                AngleList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_3D_Angle.p"%(ModelName,SeqNum)),"rb"))

            elif Type == "2D":
                EucErrorList = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"rb"))
                PCK10List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"rb"))
                PCK05List = pickle.load(open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"rb"))
                AngleList = []

            AllEucErrorList.extend(EucErrorList)
            AllPCK10List.extend(PCK10List)
            AllPCK05List.extend(PCK05List)
            AllAngleList.extend(AngleList)

        ModelEucDict = {"Model":ModelName}
        PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
        ModelEucDict["Overall"] = GetRMSE(AllPointsListEuc)
        for Key,Val in PerKeypointDictEuc.items():
            RMSE = GetRMSE(Val)
            ModelEucDict.update({Key:RMSE})
        EucDFDict[counter] = ModelEucDict

        ModelMedDict = {"Model":ModelName}
        PerKeypointDictEuc, AllPointsListEuc = RMSESummaryDict(AllEucErrorList,PIGEON_KEYPOINT_NAMES,filter=False)
        ModelMedDict.update({"Overall":GetMedian(AllPointsListEuc)})
        for Key,Val in PerKeypointDictEuc.items():
            RMSE = GetMedian(Val)
            ModelMedDict.update({Key:RMSE})
        MedianDFDict[counter]= ModelMedDict

        ModelPCK10Dict = {"Model":ModelName}
        PerKeypointDictPCK10, AllPointsListPCK10 = PCKSummaryDict(AllPCK10List,PIGEON_KEYPOINT_NAMES)
        ModelPCK10Dict.update({"Overall":GetPCKSum(AllPointsListPCK10)})
        for Key,Val in PerKeypointDictPCK10.items():
            PCK = GetPCKSum(Val)
            ModelPCK10Dict.update({Key:PCK})
        PCK10DFDict[counter] = ModelPCK10Dict

        ModelPCK05Dict = {"Model":ModelName}
        PerKeypointDictPCK05, AllPointsListPCK05 = PCKSummaryDict(AllPCK05List,PIGEON_KEYPOINT_NAMES)
        ModelPCK05Dict.update({"Overall":GetPCKSum(AllPointsListPCK05)})
        for Key,Val in PerKeypointDictPCK05.items():
            PCK = GetPCKSum(Val)
            ModelPCK05Dict.update({Key:PCK})
        PCK05DFDict[counter] = ModelPCK05Dict

        if Type == "3D":
            ModelAngleDict = {"Model":ModelName}
            PerDimDictAngle, AllPointsListAngle = AngleSummaryDict(AllAngleList)
            ModelAngleDict.update({"Overall_RMSE":GetRMSE(AllPointsListAngle)})
            ModelAngleDict.update({"Overall_Median":GetMedian(AllPointsListAngle)})

            for Key,Val in PerDimDictAngle.items():
                RMSE = GetRMSE(Val)
                Median = GetMedian(Val)
                ModelAngleDict.update({"%s_RMSE"%Key:RMSE})
                ModelAngleDict.update({"%s_Median"%Key:Median})

            AnglesDFDict[counter] = ModelAngleDict

        counter += 1

    EucDF = pd.DataFrame.from_dict(EucDFDict,orient= "index")
    EucDF.to_csv("./EucErrorSummary%s.csv"%Type)

    PCK10DF = pd.DataFrame.from_dict(PCK10DFDict,orient= "index")
    PCK10DF.to_csv("./PCK10Summary%s.csv"%Type)

    PCK05DF = pd.DataFrame.from_dict(PCK05DFDict,orient= "index")
    PCK05DF.to_csv("./PCK05Summary%s.csv"%Type)

    MedianDF = pd.DataFrame.from_dict(MedianDFDict,orient= "index")
    MedianDF.to_csv("./MedianSummary%s.csv"%Type)

    if Type == "3D":
        AnglesDF = pd.DataFrame.from_dict(AnglesDFDict,orient= "index")
        AnglesDF.to_csv("./AngleSummary%s.csv"%Type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/media/alexchan/WD Elements AE/Pop3D-Dataset", help="Path to the Pop3D dataset")
    args = parser.parse_args()

    EvalDir = "EvaluationData/"
    PickleFiles = [os.path.basename(file) for file in glob(EvalDir + "/*.p")]
    DatasetPath = args.dataset_path

    Files3D = [file for file in PickleFiles if "Kalman3D" in file]

    for file3d in Files3D:

        _, _, ModelName, SeqNum= file3d.split("_")
        SeqNum = SeqNum.split(".")[0].split("Seq")[1]

        print("Seq: %s, Model: %s"%(SeqNum,ModelName))

        if ModelName == "ltohp":
            Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
            EucErrorList3D,PCK10List3D,PCK05List3D,AngleList3D = DoEval3D(DatasetPath, SeqNum,Predictions3D)
            pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
            pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
            pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))
            pickle.dump(AngleList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_Angle.p"%(ModelName,SeqNum)),"wb"))
            continue

        Predictions3D= pickle.load(open(os.path.join(EvalDir,file3d ), "rb"))
        Predictions2D = pickle.load(open(os.path.join(EvalDir,"SeqEval_Points2D_%s_Seq%s.p"%(ModelName,SeqNum) ), "rb"))

        EucErrorList3D,PCK10List3D,PCK05List3D, EucErrorList2D, PCK10List2D,PCK05List2D,AngleList3D = DoEval(DatasetPath, SeqNum,Predictions3D,Predictions2D)

        pickle.dump(EucErrorList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_EucError.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK10List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK10.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK05List3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_PCK05.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(AngleList3D, open(os.path.join(EvalDir,"%s_Seq%s_3D_Angle.p"%(ModelName,SeqNum)),"wb"))

        pickle.dump(EucErrorList2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_EucError.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK10List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK10.p"%(ModelName,SeqNum)),"wb"))
        pickle.dump(PCK05List2D, open(os.path.join(EvalDir,"%s_Seq%s_2D_PCK05.p"%(ModelName,SeqNum)),"wb"))

    Models3D = ["YOLOVit","YOLODLC","KPRCNN","ltohp"]
    Models2D = ["YOLOVit","YOLODLC","KPRCNN"]

    AllSequences = [11,1,2,5]

    GetSummaryCSV(Models3D,AllSequences, Type = "3D")
    GetSummaryCSV(Models2D,AllSequences, Type = "2D")
