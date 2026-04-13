"""

Compute angles of head coordinate system of test set and compare with mocap

For YOLODLC

"""

import numpy as np

import sys
sys.path.append("./")

from Utils import JSONReader
import pickle
import math
import statistics
import cv2
import os

from numpy import *
from math import sqrt

from tqdm import tqdm

def findPoseFromPoints(pointsInCurrentFrameA, pointsInTargetFrameB,Name=None):
    """
    Function finds pose to transfer points from frame of reference A to frame of reference B.
    :param pointsInCurrentFrameA: 3xN Matrix of point set A
    :param pointsInTargetFrameB: 3xN Matrix of point set B
    :return: 3x3 Rotation Matrix and 3x1 Translation matrix,
    """

    pointsInCurrentFrameA = mat(pointsInCurrentFrameA)
    pointsInTargetFrameB = mat(pointsInTargetFrameB)

    assert len(pointsInCurrentFrameA) == len(pointsInTargetFrameB)

    num_rows, num_cols = pointsInCurrentFrameA.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = pointsInTargetFrameB.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(pointsInCurrentFrameA, axis=1)
    centroid_B = mean(pointsInTargetFrameB, axis=1)

    if centroid_A.shape != (3,1) and centroid_B.shape != (3,1):
        centroid_A = centroid_A.reshape(3,1)
        centroid_B = centroid_B.reshape(3,1)

    # subtract mean
    Am = pointsInCurrentFrameA - tile(centroid_A, (1, num_cols))
    Bm = pointsInTargetFrameB - tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * transpose(Bm)

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        # TEMP
        # if Name == "485_1307_hd":
        #     print("hi")
        # else:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t


def computeExtrinsic(cam1Rotation, cam1Translation, cam2Rotation, cam2Translation):
    """
    compute extrinsic matrix between the given cameras. Rotation and translation parameters supposed to bring points to
    a common coordinate space from camera space.
    Extrinsics can convert points from cam2 space to cam1 space
    Pv = R_cam1. Pc1 + t_cam1 , Pv = R_cam2 . Pc2 + t_cam2
    :param cam1Rotation: rotation ( C -> V) 3x3 matrix
    :param cam1Translation: translation (C -> V) 3x1 matrix
    :param cam2Rotation: rotation (C -> V) 3x3 matrix
    :param cam2Translation: (C -> V) 3x1 matrix
    :return: Rotation and translation ( C2 -> C1)
    """

    # Rotation
    # rotationCam2toCam1 = inverse(cam1Rotation) . cam2Rotation
    rotationCam2toCam1 = np.dot( np.linalg.inv(cam1Rotation), cam2Rotation)

    # translationCam2toCam1 = inverse(cam1Rotation) . (cam2Translation - cam1Translation)
    translationCam2toCam1 = np.dot ( np.linalg.inv(cam1Rotation), (cam2Translation-cam1Translation))

    return rotationCam2toCam1, translationCam2toCam1




def transformPoints(pointMatrix, rotationMatrix, translationMatrix):
    """
    transforms points based on given rotation and translation matrix
    :param pointMatrix : Matrix 3xN
    :param rotationMatrix : rotation matrix 3x3
    :param translationMatrix : translation matrix 3x1
    :return : Matrix 3xN
    """

    assert( pointMatrix.shape[0] == 3), "Expected dimensions of points are 3XN"
    assert (rotationMatrix.shape == (3, 3) and
            translationMatrix.shape == (3, 1)) , "Invalid dimension of rotation or tranlsation matrix"

    # P_out (3xN) = R (3x3) . P (3xN) + T (3x1)
    rotatedPoints = np.dot(rotationMatrix, pointMatrix)
    transformedPoints = np.add(rotatedPoints, translationMatrix)

    return transformedPoints


def GetMagnitude(point):
    """Return magnitute of vector"""
    return math.sqrt((point[0]**2+point[1]**2+point[2]**2))

def GetMidPoint(p1, p2):
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

def GetObjCoords(Point3D, R,T):
    """Given R and T and 3d points in world coordinate, get object coordinate"""

    ###Get points in object coordinate system:
    # import ipdb;ipdb.set_trace()
    # ObjPointsObj = np.dot((Point3D-T),R)

    Point3D = Point3D.T
    T = np.array(T).reshape(3,1)

    Translated = Point3D-T

    #test:
    ObjPointsObj = np.dot(np.linalg.inv(R),Translated).T


    return ObjPointsObj


def getRotationAngle(P, Q):
    """Get angle difference between 2 rotation matrices, from stacke exchange"""
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    return np.arccos(cos_theta) * (180/np.pi)


def DefineObj(Point3DDict,bird):
    """
    Define head object coordinate from 3 points: beak and 2 eyes
    Get rotation angle of 
    
    """
    # import ipdb;ipdb.set_trace()

    ##Prepare points
    beak = np.array(Point3DDict["%s_hd_beak"%bird], dtype=np.float64)#origin is beak
    pt1 = np.array(Point3DDict["%s_hd_leftEye"%bird], dtype=np.float64) #origin is beak
    pt2 = np.array(Point3DDict["%s_hd_rightEye"%bird], dtype=np.float64) #origin is beak
    # import ipdb;ipdb.set_trace()

    ##Get vector from eye to beak
    Vec1 = pt1-beak
    Vec2 = pt2-beak
    PlaneNormal = np.cross(Vec1,Vec2)
    NormalUnit = PlaneNormal/GetMagnitude(PlaneNormal)

    ###get vector of between eye to beak (as the y axis)
    BetweenEye = GetMidPoint(pt1,pt2)
    ForwardVec = BetweenEye-beak
    ForwardUnit = ForwardVec/GetMagnitude(ForwardVec)
    
    #get horizontal axis (x), normal is (z), mid eye to beak is y
    HorizontalAxis = np.cross(PlaneNormal,ForwardVec)
    HorizontalUnit = HorizontalAxis/GetMagnitude(HorizontalAxis)

    ###Calc angles against principle axes
    Xaxis = np.array([1,0,0])
    Yaxis = np.array([0,1,0])
    Zaxis = np.array([0,0,1])

    # HeadVecs = np.array([HorizontalUnit,ForwardUnit,NormalUnit])
    # OriginUnit = np.array([Xaxis,Yaxis,Zaxis])
    # R2, T2 = findPoseFromPoints( OriginUnit,HeadVecs)

    # ##Rotate matrix:
    # R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
    #              [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
    #              [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
    T = BetweenEye

    R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
                 [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
                 [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
   

    ###Based on Kano et al: rotate coordinate up 30 degrees
    DegreeRad = math.radians(30)
    #rotation matrix for along x axis
    RotateX = np.array([[1,0,0],
                        [0,math.cos(DegreeRad),-math.sin(DegreeRad)],
                        [0,math.sin(DegreeRad),math.cos(DegreeRad)]])

    ##rotate current rotation along x axis for 30 degrees more
    R = np.dot(R,RotateX)

    return R,T

def GetEucDist(Point1,Point2):
    """Get euclidian error, both 2D and 3D"""
    
    if len(Point1) ==3 & len(Point2) ==3:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
    elif len(Point1) ==2 & len(Point2) ==2:
        EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
    else:
        import ipdb;ipdb.set_trace()
        Exception("point input size error")
    
    return EucDist


#### Rotation matrix to Euler: from https://learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def DrawObjectOrigin(frame,camR,camT,cameraMatrix,distCoeffs,R,T):
    """Given camera parameters and a pose of a coordinate system, draw principle axis"""

    OriginalPoints = np.array([[0,0,0],
                              [100,0,0],
                              [0,100,0],
                              [0,0,100]])
    # import ipdb;ipdb.set_trace()

    TransPoints = transformPoints(OriginalPoints.T, R, np.array(T).reshape(3,1))

    Allimgpts, jac = cv2.projectPoints(TransPoints,camR,camT,cameraMatrix,distCoeffs)

    RoundedPoints = [(round(pt[0][0]),round(pt[0][1])) for pt in Allimgpts]

    frame = cv2.line(frame, RoundedPoints[0],RoundedPoints[1], [0,0,255],5)
    frame = cv2.line(frame, RoundedPoints[0],RoundedPoints[2], [255,0,0],5)
    frame = cv2.line(frame, RoundedPoints[0],RoundedPoints[3], [0,255,0],5)

    return frame

def cartesian_to_spherical(vector):
    """Converts from Cartesian coordinates to spherical coordinates.

    from Google Bard :)

    Args:
    x: The x-coordinate in Cartesian coordinates.
    y: The y-coordinate in Cartesian coordinates.
    z: The z-coordinate in Cartesian coordinates.

    Returns:
    A tuple of (r, theta, phi), where r is the radius, theta is the polar angle,
    and phi is the azimuthal angle in spherical coordinates.
    """

    x = vector[0]
    y=vector[1]
    z=vector[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y,x)
    phi = np.arccos(z / r)

    return [r, math.degrees(theta), math.degrees(phi)]


def GetVSpolar(R, T):
    """Get left and right visual field"""
    PrinciplePoint = np.array([0,1,0])
    Angle = 75

    # ##Old:
    # ##rotate beak around z axis:
    AngleRadRight = math.radians(-Angle)
    #https://stackoverflow.com/questions/14607640/rotating-a-vector-in-3d-space
    VSRight = np.array([PrinciplePoint[0]*math.cos(AngleRadRight) - PrinciplePoint[1]*math.sin(AngleRadRight),
                        PrinciplePoint[0]*math.sin(AngleRadRight) + PrinciplePoint[1]*math.cos(AngleRadRight),0])
    
    AngleRadLeft = math.radians(Angle)
    VSLeft = np.array([PrinciplePoint[0]*math.cos(AngleRadLeft) - PrinciplePoint[1]*math.sin(AngleRadLeft),
                        PrinciplePoint[0]*math.sin(AngleRadLeft) + PrinciplePoint[1]*math.cos(AngleRadLeft),0])

    AngleRadFront = math.radians(-45) #-45 down along x axis
    ShortPrinciplePoint = PrinciplePoint
    VSFront = np.array([0,ShortPrinciplePoint[1]*math.cos(AngleRadFront) - ShortPrinciplePoint[2]*math.sin(AngleRadFront),
                    ShortPrinciplePoint[1]*math.sin(AngleRadFront) + ShortPrinciplePoint[2]*math.cos(AngleRadFront)])
    #along y axis:???
    # VSFront = np.array([ShortPrinciplePoint[0]*math.cos(AngleRadFront) + ShortPrinciplePoint[2]*math.sin(AngleRadFront),ShortPrinciplePoint[1],
    #                 -ShortPrinciplePoint[0]*math.sin(AngleRadFront) + ShortPrinciplePoint[2]*math.cos(AngleRadFront)])

    ###Transform back to world coordinate
    inverse_R = np.linalg.inv(R)
    # inverse_R = R

    VSRightWorld = np.add(np.dot(VSRight,inverse_R ),T)
    VSLeftWorld = np.add(np.dot(VSLeft,inverse_R ),T)
    VSFrontWorld = np.add(np.dot(VSFront,inverse_R ),T)

    ###VS World vector:
    VSRightVec = VSRightWorld - T
    VSLeftVec = VSLeftWorld - T
    VSFrontVec = VSFrontWorld - T

    ##Get polar:
    VSR_Polar = cartesian_to_spherical(VSRightVec)
    VSL_Polar = cartesian_to_spherical(VSLeftVec)
    VSF_Polar = cartesian_to_spherical(VSFrontVec)

    return VSR_Polar,VSL_Polar,VSF_Polar


def AngleEval(YOLODLCDictList,Dataset,DatasetPath):
    PitchDiff = []
    RollDiff = []
    YawDiff = []
    MidPointDiff = []

    LeftDiff = []
    RigtDiff = []
    FrontDiff = []

    for i in tqdm(range(len(YOLODLCDictList))):
        #ground truth:
        GTDict = Dataset.Extract3D(i)

        #Predict:
        frameDict = YOLODLCDictList[i]
        ImagePath = os.path.join(DatasetPath,Dataset.GetImagePath(i)[0])

        frame = cv2.imread(ImagePath)

        for BirdID in GTDict.keys():
            BirdDictGT = {"%s_%s"%(BirdID,k):v for k,v in GTDict[BirdID].items()}
            BirdDictPred = {k:v for k,v in frameDict.items() if k.startswith(BirdID)}
            # import ipdb;ipdb.set_trace()

            GT_R, GT_T = DefineObj(BirdDictGT,BirdID)
            Pred_R, Pred_T = DefineObj(BirdDictPred,BirdID)

            ##Validate Euler angles:
            IdentityMat = np.identity(3)
            TranslationOrigin = np.array([0,0,0]).reshape(3,1)
            Inv_GT_R, Inv_GT_T = computeExtrinsic(IdentityMat, TranslationOrigin,GT_R, np.array(GT_T).reshape(3,1))
            Inv_Pred_R, Inv_Pred_T = computeExtrinsic( IdentityMat, TranslationOrigin,Pred_R, np.array(Pred_T).reshape(3,1))

            ###Find rotation between GT and Pred
            # import ipdb;ipdb.set_trace()
            DiffR, DiffT = computeExtrinsic(Inv_GT_R,Inv_GT_T,Inv_Pred_R , Inv_Pred_T)
            # DiffR2, DiffT2 = computeExtrinsic(GT_R,np.array(GT_T).reshape(3,1),Pred_R , np.array(Pred_T).reshape(3,1))

            # np.dot(GT_R,np.linalg.inv(Pred_R))
            
            # import ipdb;ipdb.set_trace()

            DiffEuler =  rotationMatrixToEulerAngles(DiffR)
            DiffEulerDeg = [math.degrees(angle) for angle in DiffEuler]

            PitchDiff.append(abs(DiffEulerDeg[0]))
            RollDiff.append(abs(DiffEulerDeg[1]))
            YawDiff.append(abs(DiffEulerDeg[2]))
            # import ipdb;ipdb.set_trace()
            MidPointDiff.append(GetMagnitude(DiffT))

            # Euler_GT =  rotationMatrixToEulerAngles(GT_R)
            # Euler_Pred =  rotationMatrixToEulerAngles(Pred_R)

            # GT_EulerDeg = [math.degrees(angle) for angle in Euler_GT ]
            # Pred_EulerDeg = [math.degrees(angle) for angle in Euler_Pred ]

            # print(Pred_EulerDeg)
            # print(GT_EulerDeg)
            # import ipdb;ipdb.set_trace()
            # # DegreeDiff = [180 - abs(abs(a - b) - 180) for a,b in zip(GT_EulerDeg,Pred_EulerDeg)]
            # PitchDiff.append(DegreeDiff[0])
            # RollDiff.append(DegreeDiff[1])
            # YawDiff.append(DegreeDiff[2])
            # MidPointDiff.append(GetEucDist(GT_T,Pred_T))


            ###Get polar coordinates of Visual Field
            GT_VSRight,GT_VSLeft,GT_VSFront = GetVSpolar(GT_R, GT_T)
            Pred_VSRight,Pred_VSLeft,Pred_VSFront = GetVSpolar(Pred_R, Pred_T)

            # print(GT_VSRight)

            VSRightError = GetEucDist([GT_VSRight[1],GT_VSRight[2]], [Pred_VSRight[1],Pred_VSRight[2]])
            VSLeftError = GetEucDist([GT_VSLeft[1],GT_VSLeft[2]], [Pred_VSLeft[1],Pred_VSLeft[2]])
            VSFrontError = GetEucDist([GT_VSFront[1],GT_VSFront[2]], [Pred_VSFront[1],Pred_VSFront[2]])

            LeftDiff.append(VSLeftError)
            RigtDiff.append(VSRightError)
            FrontDiff.append(VSFrontError)

            # import ipdb;ipdb.set_trace()
            # RotationDiff = getRotationAngle(GT_R,Pred_R)
            # AngleList.append(RotationDiff)
            # import ipdb;ipdb.set_trace()

            # camMats, distCoeffs = Dataset.GetIntrinsics(i)
            # Rvecs, Tvecs = Dataset.GetExtrinsics(i)
            # # frame = cv2.drawFrameAxes(frame,camMats[0],distCoeffs[0],GT_R,np.array(GT_T),150,10)
            # # frame = cv2.drawFrameAxes(frame,camMats[0],distCoeffs[0],Rvecs[0],Tvecs[0],150,10)
            # frame = DrawObjectOrigin(frame,Rvecs[0],Tvecs[0],camMats[0],distCoeffs[0],GT_R,GT_T)
            # frame = DrawObjectOrigin(frame,Rvecs[0],Tvecs[0],camMats[0],distCoeffs[0],Pred_R,Pred_T)

        # cv2.imshow("Window",frame)
        # cv2.waitKey(0)
            # import ipdb;ipdb.set_trace()


    # print(np.array(AngleList).mean())
    # print(statistics.median(AngleList))
    # import ipdb ;ipdb.set_trace()
    import matplotlib.pyplot as plt
    plt.hist(RollDiff, bins=100)
    plt.show()
    plt.hist(PitchDiff, bins=100)
    plt.show()
    plt.hist(YawDiff, bins=100)
    plt.show()
    plt.hist(MidPointDiff, bins=100)
    plt.show()


    print("Roll Mean error")
    print(np.array(RollDiff).mean())
    print("Pitch Mean error")
    print(np.array(PitchDiff).mean())
    print("Yaw Mean error")
    print(np.array(YawDiff).mean())
    print("Origin error")
    print(np.array(MidPointDiff).mean())

    # print("Right Mean error")
    # print(np.array(LeftDiff).mean())
    # print("Left Mean error")
    # print(np.array(RigtDiff).mean())
    # print("Front error")
    # print(np.array(FrontDiff).mean())

    # plt.hist(LeftDiff, bins=100)
    # plt.show()
    # plt.hist(RigtDiff, bins=100)
    # plt.show()
    # plt.hist(FrontDiff, bins=100)
    # plt.show()

def DefineObj(Point3DDict):
    """
    Define head object coordinate from 3 points: beak and 2 eyes
    Get rotation angle of 
    
    """
    # import ipdb;ipdb.set_trace()

    ##Prepare points
    beak = np.array(Point3DDict["hd_beak"], dtype=np.float64)#origin is beak
    pt1 = np.array(Point3DDict["hd_leftEye"], dtype=np.float64) #origin is beak
    pt2 = np.array(Point3DDict["hd_rightEye"], dtype=np.float64) #origin is beak
    # import ipdb;ipdb.set_trace()

    if np.isnan(beak).any() or np.isnan(pt1).any() or np.isnan(pt2).any():
        return np.nan, np.nan


    ##Get vector from eye to beak
    Vec1 = pt1-beak
    Vec2 = pt2-beak
    PlaneNormal = np.cross(Vec1,Vec2)
    NormalUnit = PlaneNormal/GetMagnitude(PlaneNormal)

    ###get vector of between eye to beak (as the y axis)
    BetweenEye = GetMidPoint(pt1,pt2)
    ForwardVec = BetweenEye-beak
    ForwardUnit = ForwardVec/GetMagnitude(ForwardVec)
    
    #get horizontal axis (x), normal is (z), mid eye to beak is y
    HorizontalAxis = np.cross(PlaneNormal,ForwardVec)
    HorizontalUnit = HorizontalAxis/GetMagnitude(HorizontalAxis)

    ###Calc angles against principle axes
    Xaxis = np.array([1,0,0])
    Yaxis = np.array([0,1,0])
    Zaxis = np.array([0,0,1])

    # HeadVecs = np.array([HorizontalUnit,ForwardUnit,NormalUnit])
    # OriginUnit = np.array([Xaxis,Yaxis,Zaxis])
    # R2, T2 = findPoseFromPoints( OriginUnit,HeadVecs)

    # ##Rotate matrix:
    # R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
    #              [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
    #              [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
    T = BetweenEye

    R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
                 [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
                 [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
   

    ###Based on Kano et al: rotate coordinate up 30 degrees
    DegreeRad = math.radians(30)
    #rotation matrix for along x axis
    RotateX = np.array([[1,0,0],
                        [0,math.cos(DegreeRad),-math.sin(DegreeRad)],
                        [0,math.sin(DegreeRad),math.cos(DegreeRad)]])

    ##rotate current rotation along x axis for 30 degrees more
    R = np.dot(R,RotateX)

    return R,T


if __name__ == "__main__":
    ##Evaluation dataset:
    DatasetPath =  "/media/alexchan/Extreme SSD/SampleDatasets/ImageTrainingData/N6000/"
    JSONPath = "/media/alexchan/Extreme SSD/SampleDatasets/ImageTrainingData/N6000/Annotation/Test-3D.json"
    Dataset = JSONReader.JSONReader(JSONPath,DatasetPath,Type="3D")
    
    Keypoints = ['bp_leftShoulder', 'bp_rightShoulder', 'bp_topKeel', 'bp_bottomKeel', 'bp_tail', 'hd_beak', 'hd_nose', 'hd_leftEye', 'hd_rightEye']
    
    YOLODLCDictList = pickle.load(open("./Data/Evaluation/YOLODLC3D.p", "rb"))

    AngleEval(YOLODLCDictList,Dataset,DatasetPath)