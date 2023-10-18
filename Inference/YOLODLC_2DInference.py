""" inference on single video for YOLO + DLC"""
import cv2 
from ultralytics import YOLO
import torch
import argparse

import sys
sys.path.append("./Repositories/DeepLabCut-live")

import deeplabcut as dlc
from dlclive import DLCLive, Processor


   
def Process_Crop(Crop, CropSize):
    """Crop image and pad, if too big, will scale down """
    # import ipdb;ipdb.set_trace()
    if Crop.shape[0] > CropSize[0] or Crop.shape[1] > CropSize[1]: #Crop is bigger, scale down
        ScaleProportion = min(CropSize[0]/Crop.shape[0],CropSize[1]/Crop.shape[1])
        
        width_scaled = int(Crop.shape[1] * ScaleProportion)
        height_scaled = int(Crop.shape[0] * ScaleProportion)
        Crop = cv2.resize(Crop, (width_scaled,height_scaled), interpolation=cv2.INTER_LINEAR)  # resize image

        # Points2D = {k:[v[0]*ScaleProportion,v[1]*ScaleProportion] for k,v in Points2D.items()}
    else:
        ScaleProportion = 1
        
    if Crop.shape[0] %2 ==0:
        #Shape is even number
        YPadTop = int((CropSize[1] - Crop.shape[0])/2)
        YPadBot = int((CropSize[1] - Crop.shape[0])/2)
    else:
        YPadTop = int( ((CropSize[1] - Crop.shape[0])/2)-0.5)
        YPadBot = int(((CropSize[1] - Crop.shape[0])/2)+0.5)
    ##Padding:
    if Crop.shape[1] %2 ==0:
        #Shape is even number
        XPadLeft = int((CropSize[0] - Crop.shape[1])/2)
        XPadRight= int((CropSize[0] - Crop.shape[1])/2)
    else:
        XPadLeft =  int(((CropSize[0] - Crop.shape[1])/2)-0.5)
        XPadRight= int(((CropSize[0] - Crop.shape[1])/2)+0.5)



    OutImage = cv2.copyMakeBorder(Crop, YPadTop,YPadBot,XPadLeft,XPadRight,cv2.BORDER_CONSTANT,value=[0,0,0])
    
    return OutImage,ScaleProportion, YPadTop,XPadLeft



def DLCInference(InferFrame,box,dlc_liveObj,CropSize):
    """Inference for DLC"""
    box = [0 if val < 0 else val for val in box] #if out of screen, 0


    Crop = InferFrame[round(box[1]):round(box[3]),round(box[0]):round(box[2])]


    if dlc_liveObj.sess == None: #if first time, init
        DLCPredict2D = dlc_liveObj.init_inference(Crop)

    DLCPredict2D= dlc_liveObj.get_pose(Crop)

    return DLCPredict2D


def VisualizeAll(frame, box, DLCPredict2D):
    """Visualize all stuff"""
    colourList = [(255,255,0),(255,0 ,255),(128,0,128),(203,192,255),(0, 255, 255),(255, 0 , 0 ),(63,133,205),(0,255,0),(0,0,255)]
    ##Order: Lshoulder, Rshoulder, topKeel,botKeel,Tail,Beak,Nose,Leye,Reye
    ##Points:
    # PlotPoints = []
    for x,point in enumerate(DLCPredict2D):
        roundPoint = [round(point[0]+box[0]),round(point[1]+box[1])]
        cv2.circle(frame,roundPoint,1,colourList[x], 3)

    cv2.rectangle(frame,(round(box[0]),round(box[1])),(round(box[2]),round(box[3])),[255,0,0],3)

    return frame


def RunInference(YOLOModel,dlc_liveObj, InputVideo,CropSize,startFrame=0,ScaleBBox=1):
    
    cap = cv2.VideoCapture(InputVideo)
    cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    imsize = (int(cap.get(3)),int(cap.get(4)))
    counter=startFrame

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    TotalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(filename="YOLODLC2D_sample.mp4", apiPreference=cv2.CAP_FFMPEG, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize = imsize)

    # while(cap.isOpened()):
    for i in range(1800):

        ret, frame = cap.read()
        # print(counter)

        if ret == True:
            InferFrame = frame.copy()
            InferFrame = InferFrame
            # InferFrame = torch.tensor(InferFrame).to("cuda")
            results = YOLOModel(InferFrame, imgsz=3840,device="cpu")
            # results = YOLOModel(InferFrame,device="cpu")

            ##Filter for birds:
            classID = [key for key,val in results[0].names.items() if val == "bird"][0]
            # frame = results[0].plot()
            DetectedClasses = results[0].boxes.cls.cpu().numpy().tolist()
            
            # bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
            bbox = results[0].boxes.xywh.cpu().numpy().tolist()
            ##Filter birds only:
            bbox = [box for x,box in enumerate(bbox) if DetectedClasses[x] == classID]


            bbox = [[box[0],box[1],box[2]*ScaleBBox,box[3]*ScaleBBox] for box in bbox] #scale width and height
            ##convert back to xyxy:
            bboxXY = [[box[0]-(box[2]/2), box[1]-(box[3]/2),box[0]+(box[2]/2),box[1]+(box[3]/2)] for box in bbox]

            # import ipdb;ipdb.set_trace()
            for box in bboxXY:
                DLCPredict2D= DLCInference(InferFrame,box,dlc_liveObj,CropSize)
                frame = VisualizeAll(frame, box, DLCPredict2D)

            out.write(frame)

            cv2.imshow('Frame',frame)
            # import ipdb;ipdb.set_trace()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    cap.release()
    cv2.destroyAllWindows()
    out.release()

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
    parser.add_argument("--DLCweight",
                        type=str,
                        default= "Weights/DLC_Barn/",
                        help="Path to pre-trained weight for exported DLC model directory")

    arg = parser.parse_args()

    return arg



if __name__ == "__main__":

    args = ParseArgs()
    VidPath = args.input
    YOLOPath = args.YOLOweight
    ExportModelPath = args.DLCweight
    CropSize = (320,320)

    YOLOModel = YOLO(YOLOPath)


    dlc_proc = Processor()
    dlc_liveObj = DLCLive(ExportModelPath, processor=dlc_proc)
    
    RunInference(YOLOModel,dlc_liveObj, VidPath,CropSize,startFrame=0,ScaleBBox=1)

