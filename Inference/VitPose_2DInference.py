# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
# from argparse import ArgumentParser
import argparse

import cv2
import numpy as np

import sys
sys.path.append("Repositories/ViTPose")

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from ultralytics import YOLO


def RunInference(args,VideoPath,YOLOModel,VitConfig,Checkpoint):
    """

    Run inference on videos
    """

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        VitConfig, Checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(VideoPath)
    assert cap.isOpened(), f'Faild to load video file {VideoPath}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(VideoPath)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)

    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break

        results = YOLOModel(img)
        # results = YOLOModel(InferFrame,device="cpu")

        ##Filter for birds:
        classID = [key for key,val in results[0].names.items() if val == "bird"][0]
        # frame = results[0].plot()
        DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()
        
        # bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
        bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
        ##Filter birds only:
        bbox = [box for x,box in enumerate(bbox) if DetectedClasses[x] == classID]
        # bboxXY = [[box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] for box in bbox]


        # keep the person class bounding boxes.
        person_results = [{'bbox': box} for box in bbox]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Input Video, path to input video")
    parser.add_argument("--YOLOweight",
                        type=str,
                        default= "Weights/YOLO_Barn.pt",
                        help="Path to pre-trained weight for YOLO model")
    parser.add_argument("--VitPoseConfig",
                        type=str,
                        default= "Configs/ViTPose_huge_3dpop_256x192.py",
                        help="Path to VitPose model config")
    parser.add_argument("--VitPoseCheckpoint",
                        type=str,
                        default= "Weights/VitPose_3DPOP.pth",
                        help="VitPose Checkpoint")
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default="./",
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    arg = parser.parse_args()

    return arg





if __name__ == '__main__':
    args = ParseArgs()
    VidPath = args.input
    YOLOPath = args.YOLOweight
    VitConfig = args.VitPoseConfig
    Checkpoint = args.VitPoseCheckpoint


    YOLOModel = YOLO(YOLOPath)

    RunInference(args,VidPath,YOLOModel,VitConfig,Checkpoint)
