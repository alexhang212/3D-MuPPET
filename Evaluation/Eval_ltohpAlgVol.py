import sys
import os
import argparse

sys.path.append("Repositories/Dataset-3DPOP")
from POP3D_Reader import Trial

from Utils import MultiPigeon3D_Dataset
from tqdm import tqdm
import math

import pickle
sys.path.append("Repositories/learnable-triangulation-pytorch/")
sys.path.append("./")

import numpy as np

import torch
from torch.utils.data import DataLoader

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet

from mvn.utils import cfg
from mvn.datasets import utils as dataset_utils

from Utils.VolumeNet import VolumetricTriangulationNet,AlgebraicTriangulationNet

PIGEON_KEYPOINT_NAMES = ['bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail','hd_beak', 'hd_nose','hd_leftEye', 'hd_rightEye']


def LoadModel(config,device):
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    model.eval()

    return model

def loadAlgDataReader(SequenceObj,frame, configAlg):
    BBoxList = [SequenceObj.camObjects[0].GetBBoxData(SequenceObj.camObjects[0].BBox,frame,bird) for bird in SequenceObj.Subjects]
    BBoxIndex = [idx for idx,bbox in enumerate(BBoxList) if not math.isnan(bbox[0][0])]

    dataset = MultiPigeon3D_Dataset.POP3D_Dataset(image_shape=(256, 256),
                 scale_bbox=1.5,
                 norm_image=True,
                 ignore_cameras=[],
                 crop=True,
                 Dataset = SequenceObj,
                 Frame = frame,
                 BBoxIndex = BBoxIndex
                 )
    dataloader = DataLoader(
        dataset,
        batch_size=len(BBoxIndex),
        num_workers=0,
                    collate_fn=dataset_utils.make_collate_fn(randomize_n_views=configAlg.dataset.train.randomize_n_views,
                                                     min_n_views=configAlg.dataset.train.min_n_views,
                                                     max_n_views=configAlg.dataset.train.max_n_views),
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
        )

    return dataloader

def loadVolDataReader(SequenceObj,frame,configVol,Points3D):
    BBoxList = [SequenceObj.camObjects[0].GetBBoxData(SequenceObj.camObjects[0].BBox,frame,bird) for bird in SequenceObj.Subjects]
    BBoxIndex = [idx for idx,bbox in enumerate(BBoxList) if not math.isnan(bbox[0][0])]

    dataset = MultiPigeon3D_Dataset.POP3D_Dataset_AlgIn(image_shape=(256, 256),
                 cuboid_side=configVol.model.cuboid_side,
                 scale_bbox=1.5,
                 norm_image=True,
                 ignore_cameras=[],
                 crop=True,
                 Dataset = SequenceObj,
                 Frame = frame,
                 BBoxIndex = BBoxIndex,
                 Points3D= Points3D
                 )
    dataloader = DataLoader(
        dataset,
        batch_size=len(BBoxIndex),
        num_workers=0,
                    collate_fn=dataset_utils.make_collate_fn(randomize_n_views=configVol.dataset.train.randomize_n_views,
                                                     min_n_views=configVol.dataset.train.min_n_views,
                                                     max_n_views=configVol.dataset.train.max_n_views),
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
        )

    return dataloader,BBoxIndex


def Inference3DPOP(Volmodel,Algmodel, SequenceNum,DatasetPath, configAlg,configVol,device):
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")

    TotalFrames = 250

    counter = 0

    Points3DDict = {}

    for i in tqdm(range(TotalFrames)):

        if SequenceNum == 59 and i<90:
            counter += 1
            continue

        DataLoader = loadAlgDataReader(SequenceObj,counter,configAlg)

        with torch.inference_mode():
            index, batch = next(enumerate(DataLoader))

            images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, configAlg)
            keypoints_3d_pred_Alg,keypoints_2d_pred,heatmap_2d_pred,confidence_pred = Algmodel(images_batch, proj_matricies_batch, batch)

            DataLoader, BBoxIndex = loadVolDataReader(SequenceObj,counter,configVol,keypoints_3d_pred_Alg)
            index, batch = next(enumerate(DataLoader))
            images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, configVol)
            keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = Volmodel(images_batch, proj_matricies_batch, batch)

            OutDict = {}
            PredKP = keypoints_3d_pred.cpu().numpy()
            for x in range(len(BBoxIndex)):
                BirdID = SequenceObj.Subjects[BBoxIndex[x]]
                BirdDict = {"%s_%s"%(BirdID,kp): PredKP[x,y,:] for y,kp in enumerate(PIGEON_KEYPOINT_NAMES)}
                OutDict.update(BirdDict)

            Points3DDict[i] = OutDict

        counter += 1

    return Points3DDict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/media/alexchan/WD Elements AE/Pop3D-Dataset", help="Path to the Pop3D dataset")
    args = parser.parse_args()

    AllSequences = [11,1,2,5]
    for SequenceNum in AllSequences:
        print("Sequence: %s"%SequenceNum)

        OutDir = "EvaluationData"
        ModelName = "ltohp"

        DatasetPath = args.dataset_path
        ConfigPathVol = "Weights/ltohp/ltohp_pigeonConfig_inferenceVol.yaml"
        ConfigPathAlg = "Weights/ltohp/ltohp_pigeonConfig_inferenceAlg.yaml"
        device = torch.device(0)

        configVol = cfg.load_config(ConfigPathVol)
        Volmodel = LoadModel(configVol,device)
        Volmodel.eval()

        configAlg = cfg.load_config(ConfigPathAlg)
        Algmodel = LoadModel(configAlg,device)
        Algmodel.eval()

        Points3DDict = Inference3DPOP(Volmodel,Algmodel, SequenceNum,DatasetPath, configAlg,configVol,device)

        pickle.dump(Points3DDict,open(os.path.join(OutDir,"SeqEval_Points3D_%s_Seq%s.p"%(ModelName,SequenceNum)), "wb"))
