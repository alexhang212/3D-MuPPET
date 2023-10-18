# !/usr/bin/env python3
""" Bunch of utility functions to visualize data"""
import cv2
import math
import numpy as np
import re

def getColor(keyPoint):
    if keyPoint.endswith("beak"):
        return (255, 0 , 0 )
    elif keyPoint.endswith("nose"):
        return (63,133,205)
    elif keyPoint.endswith("leftEye"):
        return (0,255,0)
    elif keyPoint.endswith("rightEye"):
        return (0,0,255)
    elif keyPoint.endswith("leftShoulder"):
        return (255,255,0)
    elif keyPoint.endswith("rightShoulder"):
        return (255, 0 , 255)
    elif keyPoint.endswith("topKeel"):
        return (128,0,128)
    elif keyPoint.endswith("bottomKeel"):
        return (203,192,255)
    elif keyPoint.endswith("tail"):
        return (0, 255, 255)
    else:
        # return (0,165,255)
        return (0,255,0)


def IsPointValid(Dim, point):
    """Check if a point is valid, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if 0 <= point[0] <= Dim[0] and 0 <= point[1] <= Dim[1]:
        Valid = True
    else:
        return Valid
    return Valid

def getType(name):
    """Get type of point and filter out marker points"""
    HDpattern = re.compile(r"hd\d")
    BPpattern = re.compile(r"bp\d")

    if bool(HDpattern.search(name)) or bool(BPpattern.search(name)): #if no digit, to filter out markers
        return None
    else:
        if "hd" in name:
            return "Head"
        elif "bp" in name:
            return "Backpack"
        else:
            return None

def PlotLine(MarkerDict,Key1Short,Key2Short,Colour,img):
    """
    Plot a line in opencv between Key1 and Key2
    Input is a dictionary of points
    
    """
    
    #Get index for given keypoint
    Key1Name = [k for k in list(MarkerDict.keys()) if Key1Short in k][0]
    Key2Name = [k for k in list(MarkerDict.keys())  if Key2Short in k][0]
    
    pt1 = MarkerDict[Key1Name]
    pt2 = MarkerDict[Key2Name]

    imsize = (img.shape[1],img.shape[0])

    if np.isnan(pt1[0]) or math.isinf(pt1[0]) or math.isinf(pt1[1]):
        return None
    elif np.isnan(pt2[0]) or math.isinf(pt2[0]) or math.isinf(pt2[1]):
        return None
    elif not IsPointValid(imsize, pt1) or not IsPointValid(imsize, pt2):
        return None

    point1 = (round(pt1[0]),round(pt1[1]))
    point2 = (round(pt2[0]),round(pt2[1]))

    cv2.line(img,point1,point2,Colour,2 )

# def GetBBoxPoints(df,bird, frameNum):
#     """Get points from df of specific bird for drawing a rectangle"""
#     ObjectColumns = [name for name in df.columns if name.startswith(bird)]
#     if len(ObjectColumns) ==0:
#         return None,None

#     Index = df.index[df["frame"]==frameNum].to_list()[0]
    
#     Start = (df.loc[Index][ObjectColumns[0]],df.loc[Index][ObjectColumns[1]])
#     End = (Start[0]+df.loc[Index][ObjectColumns[2]],Start[1]+df.loc[Index][ObjectColumns[3]])

#     return Start,End


# def GetBBoxMidPoints(Start,End):
#     """Get midpoint of bounding box"""
#     MidPoint = [round(Start[0])+((round(End[0])-round(Start[0]))/2),round(Start[1])+((round(End[1])-round(Start[1]))/2)]
#     return MidPoint

def PlotBirdTrajectory(img, PointList, frames, Colour):
    PointSub = PointList[-frames:]

    for i in range(len(PointSub)-1):
        pt1 = [round(PointSub[i+1][0]),round(PointSub[i+1][1])]
        pt2 = [round(PointSub[i][0]),round(PointSub[i][1])]
        cv2.line(img,pt1,pt2,Colour,2)
