"""
Prepare tracking evaluation files from barn test set tracking output

"""

import os
import sys
sys.path.append("../MAAP3D-3DPOP")
from POP3D_Reader import Trial
import pickle
import numpy as np
sys.path.append("Utils")

import HungarianAlgorithm, BBoxOverlap
from scipy.spatial import distance as dist

from tqdm import tqdm
import argparse

def MatchGTIDs(CamObj,TrackerOutDict, BirdIDs):
    """Given Tracking dict and ground truth, get a dictionary of corresponding GT IDs to index"""

    #Get bot keel point per ID:
    ##Look for first frame with no NAs for Keel point:
    for i in range(len(CamObj.Keypoint3D.index)):
        
        KeelPointDict = {}
        for bird in BirdIDs:
            KeelPoint = CamObj.Read3DKeypointData(CamObj.Keypoint3D,i,bird)["%s_bp_bottomKeel"%bird]
            KeelPointDict[bird] = KeelPoint.copy()

        if np.isnan(np.array(list(KeelPointDict.values()))).any():
            continue
        else:
            MatchingIndex = i
            print("Matching ground truth IDs based on frame index %s"%i)
            break
    
    MatchTracking = TrackerOutDict[MatchingIndex]

    DiffMatrix = dist.cdist(list(KeelPointDict.values()),list(MatchTracking.values()))

    MatchedPairs = HungarianAlgorithm.hungarian_algorithm(DiffMatrix)

    OutMatchingDict = {}
    for pair in MatchedPairs:
        OutMatchingDict.update({list(KeelPointDict.keys())[pair[0]]:list(MatchTracking.keys())[pair[1]]})
    

    return OutMatchingDict

def  MatchGTBBoxIDs(CamObj,CamTracking,BirdIDs):
    """
    Given Tracking dict and ground truth, get a dictionary of corresponding GT IDs to index
    Version for BBox
    """

    #Get bot keel point per ID:
    ##Look for first frame with no NAs for Keel point:
    for i in range(len(CamObj.BBox.index)):
        bboxDict = {}
        for bird in BirdIDs:
            bbox = CamObj.GetBBoxData(CamObj.BBox,i,bird)
            bboxDict[bird] = [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]

        if np.isnan(np.array(list(bboxDict.values()))).any():
            continue
        else:
            MatchingIndex = i
            print("Matching ground truth IDs based on frame index %s"%i)
            
    
            # Tracking data:
            FrameTrackData = CamTracking[i]
            TrackBBoxList = []
            TrackIDList = []
            for track in FrameTrackData:
                TrackBBoxList.append([track[0],track[1],track[2],track[3]])
                TrackIDList.append(track[4])

            break

    OutMatchingDict = BBoxOverlap.MatchID_BBox_Hungarian_Basic(bboxDict,TrackBBoxList )

    return OutMatchingDict, TrackIDList


def GetEvaluate3DCSVs(OutDir,SequenceObj,TrackerOutDict):


    OutSubDir = os.path.join(OutDir,"%s_3D"%SequenceObj.TrialName)
    if not os.path.exists(OutSubDir):
        os.mkdir(OutSubDir)
        os.mkdir(os.path.join(OutSubDir, "gt"))


    GTPath = os.path.join(OutSubDir,"gt","gt.txt")
    DetectPath = os.path.join(OutSubDir,"track.txt")
    # TrackerPath =  os.path.join(OutDir,"track.txt")
    

    CamObj = SequenceObj.camObjects[0]

    ##Get matching of GT IDs based on first frame
    OutMatchingDict = MatchGTIDs(CamObj,TrackerOutDict,SequenceObj.Subjects)


    ###Interpolate points
    Keypoint3DInterpolate = CamObj.Keypoint3D.interpolate(methods="linear")


    with open(GTPath, "w") as GTFile:
        with open(DetectPath, "w") as DTFile:
            for i in tqdm(range(len(TrackerOutDict))):
                if i not in TrackerOutDict:
                    continue

                # import ipdb;ipdb.set_trace()

                # frameNum = i+1
                frameNum = i+90

                ##GT txt:
                PointsList = []
                IDList = []
                for bird in SequenceObj.Subjects:
                    # point = CamObj.Read3DKeypointData(Keypoint3DInterpolate,i,bird)["%s_bp_bottomKeel"%bird]
                    point = CamObj.Read3DKeypointData(Keypoint3DInterpolate,frameNum,bird)["%s_bp_bottomKeel"%bird]

                    CorrespondingID = OutMatchingDict[bird]
                    PointsList.append(point)
                    IDList.append(CorrespondingID)

                if np.isnan(np.array(PointsList)).any():
                    continue
                
                for x in range(len(PointsList)):
                    GTFile.write("%s,%s,%s,%s,%s\n"%(frameNum,IDList[x],PointsList[x][0]+10000,PointsList[x][1]+10000,PointsList[x][2]+10000))


                ###Detection txt:
                FrameTrackDict = TrackerOutDict[i]
                for ID, point in FrameTrackDict.items():
                    DTFile.write("%s,%s,%s,%s,%s\n"%(frameNum,ID,point[0]+10000,point[1]+10000,point[2]+10000))


def GetEvaluate2DCSVs(OutDir,SequenceObj,Detection2D,Tracking2D,cam):
    OutSubDir = os.path.join(OutDir,"%s_%s_2D"%(SequenceObj.TrialName,cam))
    if not os.path.exists(OutSubDir):
        os.mkdir(OutSubDir)
        os.mkdir(os.path.join(OutSubDir,"gt"))
        os.mkdir(os.path.join(OutSubDir,"det"))


    CamDetection = Detection2D[cam]
    CamTracking = Tracking2D[cam]

    # import ipdb;ipdb.set_trace()

    GTPath = os.path.join(OutSubDir,"gt","gt.txt")
    DetectPath = os.path.join(OutSubDir,"det","det.txt")
    TrackerPath =  os.path.join(OutSubDir,"track.txt")
    InfoPath = os.path.join(OutSubDir,"seqinfo.ini")

    #Write info file:
    with open(InfoPath, "w") as infoFile:
        infoFile.writelines("[Sequence]\n")
        infoFile.writelines("name=%s_%s\n"%(SequenceObj.TrialName,cam))
        infoFile.writelines("imDir=img1\n")
        infoFile.writelines("frameRate=30\n")
        infoFile.writelines("seqLength=%s\n"%len(CamDetection))
        infoFile.writelines("imWidth=3840\n")
        infoFile.writelines("imHeight=2160\n")
        infoFile.writelines("imExt=.jpg\n")



    CamObj = [c for c in SequenceObj.camObjects if c.CamName==cam][0]

    ##Get matching of GT IDs based on first frame
    OutMatchingDict, TrackIDList = MatchGTBBoxIDs(CamObj,CamTracking,SequenceObj.Subjects)


    with open(GTPath, "w") as GTFile:
        with open(DetectPath, "w") as DTFile:
            with open(TrackerPath, "w") as TRFile:
                for i in tqdm(range(len(CamDetection))):
                    if i not in CamDetection:
                        continue
                    frameNum = i+1

                    ##GT txt:
                    for bird in SequenceObj.Subjects:
                        # GTbbox = CamObj.GetBBoxData(CamObj.BBox,i,bird)
                        GTbbox = CamObj.GetBBoxData(CamObj.BBox,frameNum,bird)

                        CorrespondingIDIndex = OutMatchingDict[bird]
                        CorrespondingID = int(TrackIDList[CorrespondingIDIndex])

                        if np.isnan(np.array(GTbbox)).any():
                            print("yo")
                            continue
                        #convert to xyhw
                        GTxyhw =  [GTbbox[0][0],GTbbox[0][1],GTbbox[1][0]-GTbbox[0][0],GTbbox[1][1]-GTbbox[0][1]]

                        GTFile.write("%s,%s,%s,%s,%s,%s,1,-1,-1,-1\n"%(frameNum,CorrespondingID,GTxyhw[0],GTxyhw[1],GTxyhw[2],GTxyhw[3]))


                    ##Tracker txt:
                    FrameTrack = CamTracking[i]
                    for track in FrameTrack:
                        ##Change from xyxy to xywh
                        trackBox = [track[0],track[1],track[2]-track[0],track[3]-track[1]]
                        TRFile.write("%s,%s,%s,%s,%s,%s,1,-1,-1,-1\n"%(frameNum,int(track[4]),trackBox[0],trackBox[1],trackBox[2],trackBox[3]))

                    ###Detection txt:
                    FrameDetect = CamDetection[i]
                    for DetectBox in FrameDetect:
                        Detectxyhw =  [DetectBox[0],DetectBox[1],DetectBox[2]-DetectBox[0],DetectBox[3]-DetectBox[1]]

                        DTFile.write("%s,-1,%s,%s,%s,%s,%s,-1,-1,-1\n"%(frameNum,Detectxyhw[0],Detectxyhw[1],Detectxyhw[2],Detectxyhw[3],DetectBox[4]))


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
    parser.add_argument("--input",
                        type=str,
                        default= "./",
                        help="Input Directory to tracking output")
    parser.add_argument("--output",
                        type=str,
                        default= "./",
                        help="Output Directory to tracking")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()

    DataDir = args.input
    DatasetPath = args.dataset
    CamNames = ["Cam1","Cam2","Cam3","Cam4"]
    SequenceNum = args.seq

    DetectionsDict = pickle.load(open(os.path.join(DataDir,"Dynamic_Point3D_Seq%s.p"%SequenceNum),"rb"))
    TrackerOutDict = pickle.load(open(os.path.join(DataDir,"Dynamic_3DTracker_Seq%s.p"%SequenceNum),"rb"))
    Detection2D  = pickle.load(open(os.path.join(DataDir,"./Dynamic_Detection2D_Seq%s.p"%SequenceNum), "rb"))
    Tracking2D = pickle.load(open(os.path.join(DataDir,"./Dynamic_Tracking2D_Seq%s.p"%SequenceNum), "rb"))
    
    
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = False, Type = "Test")


    OutDir = os.path.join(DataDir, "TrackingEval_Seq%s"%SequenceNum)
    if not os.path.exists(OutDir):
        os.mkdir(OutDir)


    GetEvaluate3DCSVs(OutDir,SequenceObj,TrackerOutDict)

    for cam in CamNames:
        GetEvaluate2DCSVs(OutDir,SequenceObj,Detection2D,Tracking2D,cam)

