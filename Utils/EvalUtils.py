import math
import itertools
import numpy as np

from BundleAdjustmentTool import BundleAdjustmentTool_Triangulation
import HungarianAlgorithm
from scipy.spatial import distance_matrix


def GetEucDist(Point1, Point2):
    if len(Point1) == 3 & len(Point2) == 3:
        EucDist = math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2))
    elif len(Point1) == 2 & len(Point2) == 2:
        EucDist = math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        raise Exception("point input size error")
    return EucDist


def MatchingAlgorithm(MatchingDict, CamNames, CamParamDict, keypoint_names, keel_index, DminThresh=200):
    CamNamePairs = list(itertools.combinations(CamNames, 2))
    CamPairDict = {}

    for CamPair in CamNamePairs:
        Cam1KP = MatchingDict[CamPair[0]]["Keypoints"]
        Cam2KP = MatchingDict[CamPair[1]]["Keypoints"]

        if len(Cam1KP) != len(Cam2KP):
            continue

        TempCamNames = [CamPair[0], CamPair[1]]

        ObjectPairs = list(itertools.product(range(len(Cam1KP)), range(len(Cam2KP))))
        ErrorArray = np.full((len(Cam1KP), len(Cam2KP)), 0, dtype=np.float32)

        Point3DDict = {}
        for IndexPair in ObjectPairs:
            Cam1Dict = {"%s" % (keypoint_names[j]): [Cam1KP[IndexPair[0]][j, 0], Cam1KP[IndexPair[0]][j, 1]] for j in range(Cam1KP[IndexPair[0]].shape[0])}
            Cam2Dict = {"%s" % (keypoint_names[j]): [Cam2KP[IndexPair[1]][j, 0], Cam2KP[IndexPair[1]][j, 1]] for j in range(Cam2KP[IndexPair[1]].shape[0])}

            Point2DDict = {TempCamNames[0]: Cam1Dict, TempCamNames[1]: Cam2Dict}

            TriangTool = BundleAdjustmentTool_Triangulation(TempCamNames, CamParamDict)
            TriangTool.PrepareInputData(Point2DDict)
            All3DPoints = TriangTool.Points3DArr[TriangTool.PointIndexArr]
            Reproject2D = TriangTool.Reproject(All3DPoints)
            Point3DDict.update({IndexPair: TriangTool.Points3DArr})

            ErrorList = [GetEucDist(TriangTool.Points2DArr[x], Reproject2D[x]) for x in range(TriangTool.Points2DArr.shape[0])]
            ErrorArray[IndexPair[0], IndexPair[1]] = np.array(ErrorList).mean()

        FinalMatches = HungarianAlgorithm.hungarian_algorithm(ErrorArray)
        FinalMatch3DPoints = [Point3DDict[IndexPair] for IndexPair in FinalMatches]
        CamPairDict.update({CamPair: {"Matches": FinalMatches, "3DPoints": FinalMatch3DPoints}})

    if len(CamPairDict) == 0:
        return {}

    PointsList = []
    IndexList = []
    for camPair, val in CamPairDict.items():
        for x in range(len(val["Matches"])):
            PointsList.append(val["3DPoints"][x][keel_index].tolist())
            IndexList.append([(camPair[0], val["Matches"][x][0]), (camPair[1], val["Matches"][x][1])])

    PairDistances = distance_matrix(PointsList, PointsList)
    np.fill_diagonal(PairDistances, np.inf)
    Dmin = 0
    GlobalMatchedList = []

    while Dmin < DminThresh:
        MinIndex = np.unravel_index(PairDistances.argmin(), PairDistances.shape)
        Dmin = PairDistances[MinIndex[0], MinIndex[1]]
        PointCamPairs = set(IndexList[MinIndex[0]] + IndexList[MinIndex[1]])

        existSetIndex = set([x for x, Subset in enumerate(GlobalMatchedList) for pair in Subset if pair in PointCamPairs])

        if len(existSetIndex) == 0:
            GlobalMatchedList.append(PointCamPairs)
        else:
            if len(existSetIndex) > 1:
                PairDistances[MinIndex[0], MinIndex[1]] = np.inf
                continue

            MatchedSet = GlobalMatchedList[list(existSetIndex)[0]]
            PresentCamNames = [pair[0] for pair in PointCamPairs]
            for pair in MatchedSet:
                if pair[0] in PresentCamNames:
                    MatchedPair = [subPair for subPair in PointCamPairs if subPair[0] == pair[0]]
                    [PointCamPairs.discard(subPair) for subPair in MatchedPair]

            GlobalMatchedList[list(existSetIndex)[0]].update(PointCamPairs)

        PairDistances[MinIndex[0], MinIndex[1]] = np.inf

    FinalCamDict = {key: {} for key in CamNames}

    for x in range(len(GlobalMatchedList)):
        for cam in CamNames:
            IndexList = [pair[1] for pair in GlobalMatchedList[x] if pair[0] == cam]
            if len(IndexList) == 0:
                CamIndex = None
            else:
                CamIndex = IndexList[0]
            FinalCamDict[cam].update({x: CamIndex})

    return FinalCamDict
