""" Do inference of ltohp from reading video"""

import sys
import os
import cv2
sys.path.append("Repositories/Dataset-3DPOP")

from POP3D_Reader import Trial
from Utils import MultiPigeon3D_Dataset
from tqdm import tqdm
import math

import shutil
import argparse

import sys
sys.path.append("Repositories/learnable-triangulation-pytorch/")
sys.path.append("./")

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils
from mvn.utils.volumetric import Line3D

from Utils.VolumeNet import VolumetricTriangulationNet,AlgebraicTriangulationNet #custom network config
PIGEON_KEYPOINT_NAMES = ['bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail','hd_beak', 'hd_nose','hd_leftEye', 'hd_rightEye']



def LoadModel(config,device):
    """Load trained model to prepare for inference"""
    # is_distributed = init_distributed(args)
    master = True
    # config
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

    # import ipdb;ipdb.set_trace()

    return model

def loadAlgDataReader(SequenceObj,frame, configAlg):
    #No NA subjects:
    BBoxList = [SequenceObj.camObjects[0].GetBBoxData(SequenceObj.camObjects[0].BBox,frame,bird) for bird in SequenceObj.Subjects]
    BBoxIndex = [idx for idx,bbox in enumerate(BBoxList) if not math.isnan(bbox[0][0])]
    
    # import ipdb;ipdb.set_trace()
    dataset = MultiPigeon3D_Dataset.POP3D_Dataset(image_shape=(256, 256),
                 scale_bbox=1.5,
                 norm_image=True,
                 ignore_cameras=[],
                 crop=True,
                 Dataset = SequenceObj,
                 Frame = frame,
                 BBoxIndex = BBoxIndex
                 )
    # import ipdb;ipdb.set_trace()
    # yo = dataset.__getitem__(0)
    # yo["keypoints_3d"]
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
    #No NA subjects:
    BBoxList = [SequenceObj.camObjects[0].GetBBoxData(SequenceObj.camObjects[0].BBox,frame,bird) for bird in SequenceObj.Subjects]
    BBoxIndex = [idx for idx,bbox in enumerate(BBoxList) if not math.isnan(bbox[0][0])]
    
    # import ipdb;ipdb.set_trace()
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
    # import ipdb;ipdb.set_trace()
    # yo = dataset.__getitem__(0)
    # yo["keypoints_3d"]
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

def ComputeProjectionMatrix(rotationMatrix,translationMatrix,intrinsicMatrix):
    """
    Computes projection matrix from given rotation and translation matrices
    :param rotationMatrix: 3x1 matrix
    :param translationMatrix: 3x1 Matrix
    :param intrinsicMatrix: 3x3 matrix
    :return: 3x4 projection matrix
    """
    
    if rotationMatrix.shape == (3,3):
        cv2.Rodrigues(rotationMatrix)[0]
    
    
    rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]
    RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
    projectionMatrix = np.dot(intrinsicMatrix, RT)
    return projectionMatrix


def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid

def draw_Grid(img,VisCam,L1pts,L2pts, W1pts, W2pts):
    """
    Draw Grid
    input are points along length axis and width axis
    
    """
    
    # import ipdb;ipdb.set_trace()
    
    ##Project to 2D
    L12D, jac = cv2.projectPoints(L1pts, VisCam.rvec, VisCam.tvec,
                                    VisCam.camMat, VisCam.distCoef)
    L22D, jac = cv2.projectPoints(L2pts, VisCam.rvec, VisCam.tvec,
                                    VisCam.camMat, VisCam.distCoef)
    W12D, jac = cv2.projectPoints(W1pts, VisCam.rvec, VisCam.tvec,
                                VisCam.camMat, VisCam.distCoef)
    W22D, jac = cv2.projectPoints(W2pts, VisCam.rvec, VisCam.tvec,
                                VisCam.camMat, VisCam.distCoef)
    
    
    #convert to int
    L12D = L12D.astype(int)
    L22D = L22D.astype(int)
    W12D = W12D.astype(int)
    W22D = W22D.astype(int)

    #connect all lengths
    for i in range(len(L12D)):
        img = cv2.line(img, tuple(L12D[i][0]), tuple(L22D[i][0]), (0,0,0), 1)
    for j in range(len(W12D)):
        img = cv2.line(img, tuple(W12D[j][0]), tuple(W22D[j][0]), (0,0,0), 1)
    return img


def CustomBuildCuboid(position, sides):
    """Custom build cuboid so z is always 0"""

    primitives = []

    line_color = (255, 255, 0)

    start = position + np.array([0, 0, 0])
    ZeroStart = position + np.array([0, 0, -position[2]]) #start point with z=0
    # import ipdb;ipdb.set_trace()
    primitives.append(Line3D(ZeroStart, start + np.array([sides[0], 0, -start[2]]), color=(255, 0, 0)))
    primitives.append(Line3D(ZeroStart, start + np.array([0, sides[1], -start[2]]), color=(0, 255, 0)))
    primitives.append(Line3D(ZeroStart, start + np.array([0, 0, sides[2]]), color=(0, 0, 255)))

    #second set start from top corner, all the same except last line
    start = position + np.array([sides[0], 0, sides[2]])
    primitives.append(Line3D(start, start + np.array([-sides[0], 0, 0]), color=line_color))
    primitives.append(Line3D(start, start + np.array([0, sides[1], 0]), color=line_color))
    primitives.append(Line3D(start, start + np.array([0, 0, -start[2]]), color=line_color))

    start = position + np.array([sides[0], sides[1], 0])
    ZeroStart = position + np.array([sides[0], sides[1], -position[2]])#start point with z=0

    primitives.append(Line3D(ZeroStart, start + np.array([-sides[0], 0, -start[2]]), color=line_color))
    primitives.append(Line3D(ZeroStart, start + np.array([0, -sides[1], -start[2]]), color=line_color))
    primitives.append(Line3D(ZeroStart, start + np.array([0, 0, sides[2]]), color=line_color))

    start = position + np.array([0, sides[1], sides[2]])
    primitives.append(Line3D(start, start + np.array([sides[0], 0, 0]), color=line_color))
    primitives.append(Line3D(start, start + np.array([0, -sides[1], 0]), color=line_color))
    primitives.append(Line3D(start, start + np.array([0, 0, -start[2]]), color=line_color))

    return primitives


def VisualizeAll(frame, VisCam,keypoints_3d_pred,cuboids_pred,
                 base_points_pred,VisualizeIndex,imsize,
                 sides=350):
    """Given input frame and predictions, plot all"""
    
    ColourList = [(255,255,0),(255, 0 , 255),(128,0,128),(203,192,255),(0, 255, 255),(255, 0 , 0 ),(63,133,205),(0,255,0),(0,0,255)]
    
    
    keypoints_3d_numpy = keypoints_3d_pred.detach().cpu().numpy()
    
    for x in range(keypoints_3d_numpy.shape[0]):
        
        keypoints3D = keypoints_3d_numpy[x]
        
        Allimgpts, jac = cv2.projectPoints(keypoints3D, VisCam.rvec, VisCam.tvec,
                                           VisCam.camMat, VisCam.distCoef)
        # import ipdb;ipdb.set_trace()

        for i in range(len(Allimgpts)):            
            pts = Allimgpts[i]
            if np.isnan(pts[0][0]) or math.isinf(pts[0][0]) or math.isinf(pts[0][1]):
                continue
            #######
            point = (round(pts[0][0]),round(pts[0][1]))            

            if IsPointValid(imsize,point):
                cv2.circle(frame,point,2,ColourList[i], -1)                
                
        
        ###Get lines
        # Lines3D = cuboids_pred[x].build()
        Lines3D = CustomBuildCuboid(cuboids_pred[x].position,cuboids_pred[x].sides) #custom build cuboid that cuts off at z plane
        # import ipdb;ipdb.set_trace()
        for line in Lines3D:
            # print(line.start_point)
            # print(line.end_point)
            startEndPoints = np.array([line.start_point,line.end_point])
            Points2D, jac = cv2.projectPoints(startEndPoints, VisCam.rvec, VisCam.tvec,
                                VisCam.camMat, VisCam.distCoef)
            if np.isnan(Points2D[0][0][0]) or np.isnan(Points2D[1][0][0]) or math.isinf(Points2D[0][0][0]) or math.isinf(Points2D[1][0][0]):
                continue
            point1 = (round(Points2D[0][0][0]),round(Points2D[0][0][1]))  
            point2 = (round(Points2D[1][0][0]),round(Points2D[1][0][1]))  

            cv2.line(frame,point1, point2, line.color, line.size)
        
        ###Drawring a grid:
        # x_range = np.arange(-2000, 2001, 200)
        # y_range = np.arange(-4000, 4001, 200)
        # ##Draw a grid:
        # L1Points = np.array([np.repeat(-2000, len(y_range)),y_range,np.repeat(0, len(y_range))],dtype = np.float32)
        # L2Points = np.array([np.repeat(2000, len(y_range)),y_range,np.repeat(0, len(y_range))],dtype = np.float32)
        # W1Points = np.array([x_range,np.repeat(-4000, len(x_range)),np.repeat(0, len(x_range))],dtype = np.float32)
        # W2Points = np.array([x_range,np.repeat(4000, len(x_range)),np.repeat(0, len(x_range))],dtype = np.float32)
        
        # frame = draw_Grid(frame,VisCam,L1Points,L2Points,W1Points,W2Points)
                
    return frame



def Inference3DPOP(Volmodel,Algmodel, SequenceNum,DatasetPath,  configAlg,configVol,device,TotalFrames = 1800,VisualizeIndex = 0):
    """Read a sequence from 3D pop then do inference with ltohp"""
    SequenceObj = Trial.Trial(DatasetPath,SequenceNum)
    SequenceObj.load3DPopTrainingSet(Filter = True, Type = "Test")
    
    VisCam = SequenceObj.camObjects[VisualizeIndex]
    
    # TotalFrames = len(VisCam.Keypoint2D.index)
    # TotalFrames = 250
    NumInd = len(SequenceObj.Subjects)
    
    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    counter = 0
    
    cap = cv2.VideoCapture(VisCam.VideoPath)
    cap.set(cv2.CAP_PROP_POS_FRAMES,counter) 
    imsize = (int(cap.get(3)),int(cap.get(4)))
    out = cv2.VideoWriter(filename="ltohp_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    Points3DDict = {}

    for i in tqdm(range(TotalFrames)):


        ret, frame = cap.read()

        if ret == False:
            break

        
        ### First Run Algorithmic
        ##Read data
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
            # import ipdb;ipdb.set_trace()
            PredKP = keypoints_3d_pred.cpu().numpy()
            for x in range(len(BBoxIndex)):
                BirdID = SequenceObj.Subjects[BBoxIndex[x]]
                BirdDict = {"%s_%s"%(BirdID,kp): PredKP[x,y,:] for y,kp in enumerate(PIGEON_KEYPOINT_NAMES)}
                OutDict.update(BirdDict)
            # print(keypoints_3d_pred)
            # if len(BBoxIndex) < 10:
            #     import ipdb;ipdb.set_trace()

            frame = VisualizeAll(frame, VisCam,keypoints_3d_pred,cuboids_pred,base_points_pred,VisualizeIndex,imsize)


            # import ipdb;ipdb.set_trace()
            # frame = VisualizeAll(frame, VisCam,keypoints_3d_gt,cuboids_pred,proj_matricies_batch,VisualizeIndex,imsize)


            cv2.imshow('Window',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # torch.cuda.empty_cache()
            out.write(frame)
            Points3DDict[i] = OutDict

        counter += 1
        
    out.release()

    return Points3DDict


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
    parser.add_argument("--configVol",
                        type=str,
                        default= "Configs/ltohp_pigeonConfig_inferenceVol.yaml",
                        help="Path to ltohp config file for volumetric model")
    parser.add_argument("--configAlg",
                        type=str,
                        default= "Configs/ltohp_pigeonConfig_inferenceAlg.yaml",
                        help="Path to ltohp config file for algebraic model")
    arg = parser.parse_args()

    return arg

if __name__ == "__main__":
    args = ParseArgs()

    DatasetPath = args.dataset
    SequenceNum = args.seq
    ConfigPathVol = args.configVol
    ConfigPathAlg = args.configAlg

    device = torch.device(0)

    configVol = cfg.load_config(ConfigPathVol)
    Volmodel = LoadModel(configVol,device)
    Volmodel.eval()

    configAlg = cfg.load_config(ConfigPathAlg)
    Algmodel = LoadModel(configAlg,device)
    Algmodel.eval()
    
    Inference3DPOP(Volmodel,Algmodel, SequenceNum,DatasetPath, configAlg,configVol,device,VisualizeIndex = 2)
