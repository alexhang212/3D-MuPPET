import numpy as np

import sys
sys.path.append("./")
sys.path.append("Utils")

import pickle
import math
from tqdm import tqdm
from glob import glob
import os
import pandas as pd

from pykalman import KalmanFilter
from natsort import natsort


def RunKalman(ColVals):
    InitialStateList = []
    FirstValIndex = np.where(~np.isnan(ColVals))[0][0]
    for col in range(ColVals.shape[1]):
        InitialStateList.append(ColVals[FirstValIndex,col])

    initial_state_mean = [InitialStateList[0],0,InitialStateList[1],0,InitialStateList[2],0]

    transition_matrix = [[1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]]

    MaskedData = np.ma.array(ColVals, mask = np.isnan(ColVals))

    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    )

    InitializeNum = 2
    kf1 = kf1.em(MaskedData[:InitializeNum], n_iter=5)

    (filtered_state_means, filtered_state_covariances)  = kf1.filter(MaskedData[:InitializeNum])
    x_new = np.zeros((3, MaskedData.shape[0]))
    x_new[0,0:InitializeNum] = filtered_state_means[:,0]
    x_new[1,0:InitializeNum] = filtered_state_means[:,2]
    x_new[2,0:InitializeNum] = filtered_state_means[:,4]

    filtered_state_means = filtered_state_means[-1]
    filtered_state_covariances = filtered_state_covariances[-1]

    ShiftCounter = 1
    for i in range(MaskedData.shape[0]):
        if i < InitializeNum:
            continue

        (filtered_state_means, filtered_state_covariances)  = kf1.filter_update(filtered_state_means, filtered_state_covariances, MaskedData[i])
        x_new[0,i] = filtered_state_means[0]
        x_new[1,i] = filtered_state_means[2]
        x_new[2,i] = filtered_state_means[4]
        ShiftCounter = 1

    Results = [x_new[0,:],x_new[1,:],x_new[2,:]]

    return Results


def RunInterpolation3D(Predictions3D):

    NewPredictions = {}
    for key,val in Predictions3D.items():
        NewDict = {}

        for k,v in val.items():
            if type(v) == float:
                NewDict["%s_x"%k] = np.nan
                NewDict["%s_y"%k] = np.nan
                NewDict["%s_z"%k] = np.nan
            else:
                NewDict["%s_x"%k] = v[0]
                NewDict["%s_y"%k] = v[1]
                NewDict["%s_z"%k] = v[2]
        NewPredictions[key] = NewDict

    data = pd.DataFrame.from_dict(NewPredictions,orient= "index")

    KalmanData = data.copy()
    UnqNames = natsort.natsorted(list(set([col[:-2] for col in data.columns])))

    for name in tqdm(UnqNames):
        ColNames = ["%s_x"%name,"%s_y"%name,"%s_z"%name]
        ColVals = data[ColNames].to_numpy()
        NewVals = RunKalman(ColVals)
        KalmanData[ColNames[0]] = NewVals[0]
        KalmanData[ColNames[1]] = NewVals[1]
        KalmanData[ColNames[2]] = NewVals[2]

    UnqNames = natsort.natsorted(list(set([col[:-2] for col in data.columns])))
    NewDF = pd.DataFrame(columns = UnqNames)

    for name in UnqNames:
        ColNames = ["%s_x"%name,"%s_y"%name,"%s_z"%name]
        ListVal = KalmanData[ColNames].values.tolist()
        ListVal2 = [val if not any(np.isnan(val)) else np.nan for val in ListVal]
        NewDF[name] = ListVal2

    NewDF = NewDF.applymap(np.array)
    NewDF.index = data.index
    FinalDict = NewDF.to_dict(orient="index")

    TempDict = pd.DataFrame.from_dict(Predictions3D,orient= "index")
    NANBefore = np.sum(TempDict.isna().sum().to_numpy())
    NANAfter = np.sum(NewDF.isna().sum().to_numpy())

    PercentageKPRemoved = (NANAfter-NANBefore)*100/NewDF.size
    print("Total Removed: %s%%"%((NANAfter-NANBefore)*100/NewDF.size))

    return FinalDict,PercentageKPRemoved


if __name__ == "__main__":
    EvalDir = "Data/EvaluationData"
    PickleFiles = [os.path.basename(file) for file in glob(EvalDir + "/*.p") if "Points3D" in file or "Points2D" in file]

    CamNames = ["Cam1","Cam2","Cam3","Cam4"]
    Models3D = ["YOLOVit","YOLODLC","KPRCNN","ltohp"]
    AllSequences = [1,2,5,11]

    Files3D = [file for file in PickleFiles if "Points3D" in file]
    PercentageDict = {ModelName:{} for ModelName in Models3D}

    for SeqNum in tqdm(AllSequences):
        for ModelName in Models3D:

            print("Sequence: %s, Model: %s"%(SeqNum,ModelName))

            Predictions3D= pickle.load(open(os.path.join(EvalDir,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SeqNum)),"rb"))

            KalmanOut,PercentageKPRemoved = RunInterpolation3D(Predictions3D)
            PercentageDict[ModelName][SeqNum] = PercentageKPRemoved
            pickle.dump(KalmanOut, open(os.path.join(EvalDir,"SeqEval_Kalman3D_%s_Seq%s.p"%(ModelName,SeqNum)), "wb"))

    FinalDF = pd.DataFrame.from_dict(PercentageDict)
    print("Mean % Removed:")
    print(FinalDF.apply(np.mean,axis = 0))
