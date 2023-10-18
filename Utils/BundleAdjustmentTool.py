# !/usr/bin/env python3

"""
Python class to do bundle adjustment, both to do triangulation and tune extrinsics

Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import scipy as sc
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
from tqdm import tqdm
import math

class BundleAdjustmentTool_Triangulation:
    """Bundle adjustment Tool, for triangulation"""

    def __init__(self, CamNames, CamParams):
        """
        CamNames: list of camera names
        CamParams: Dictionary of camera parameters (intrinsic and extrinsics)
        Points2D: Detections in 2D space
        
        """
        self.CamNames = CamNames
        self.CamParamDict = CamParams
        self.PrepareCamParams(CamParams)
        
    def PrepareCamParams(self,CamParams):
        """Reads in Dictionary of camera prameters then put it in correct format"""
        
        CamRvecList = []
        CamTvecList = []
        CamMatList = []
        DistList = []
        for i in range(len(self.CamNames)):
            Cam = self.CamNames[i]
            #Extrsinics
            rvec = CamParams[Cam]["R"]
            tvec = CamParams[Cam]["T"]
            CamRvecList.append(rvec)
            CamTvecList.append(tvec)
            
            ##Intrinsics:
            CamMatList.append(CamParams[Cam]["cameraMatrix"])
            DistList.append(CamParams[Cam]["distCoeffs"])

        self.CamMatList = CamMatList
        self.DistList = DistList
        self.CamRvecList = CamRvecList
        self.CamTvecList = CamTvecList
        

    def PrepareInputData(self,PointsDictList):
        """
        input:
        PointsDictList: Dictionary named by camera names, with inner dictionary of points 
        {PointName : [x,y]}
        
        output:
        Points3D (N_point,3): 3D coordinates of unique 3D points
        PointIndex (N_Observation,): index of which 3d points a given 2d point belongs
        CamIndex (N_Observation,) : index of which camera a given 2d point belongs
        Points2D (N_Observation,2) : 2D coordinates of given feature
        """
        ##Output Lists
        Points3DList = []
        PointIndexList = []
        CamIndexList = []
        Points2DList = []        
        
        #Get all unique frames
        Cameras = list(PointsDictList.keys())
        Point2DNameList = [list(val.keys()) for val in PointsDictList.values()]
        
        #get all unique 2Dpoints:
        AllUnqPoints = sorted(list(set().union(*Point2DNameList)))
        
        Point3DIndexCounter = 0 ##rolling count for 3d point index
        PointNameList = [] ##Kep tracks of point names
        # Initial3DPoints = []
        
        for point in AllUnqPoints:
            ##Exists 2D measurements in at least 2 views:
            ExistFrameData = {}
            for x in range(len(Cameras)):
                Cam = Cameras[x]
                if point in list(PointsDictList[Cam].keys()):
                    # PointIndex = PointsDictList[Cam]["ImgIDs"].index(point)
                    ExistFrameData.update({Cam:PointsDictList[Cam][point]})
                else:
                    continue
            if len(ExistFrameData) <2:
                ##Only 1 cam saw point
                continue
            
            #At least 2 views detected point:
            AllCamIndexes = list(ExistFrameData.keys())
            CamIndexes = (AllCamIndexes[0],AllCamIndexes[1]) #Just choose first 2 cameras
            #Initial 3D point estimate:
            PointDicts = (ExistFrameData[CamIndexes[0]],ExistFrameData[CamIndexes[1]] )
            # PointDicts = (PointsDictList[Cameras[CamIndexes[0]]][frame], PointsDictList[Cameras[CamIndexes[1]]][frame])
            Initial3D = self.TriangulatePoints(CamIndexes, PointDicts) #OverlapID is the checkboard object ids that were used in triangulation
            PointNameList.append(point)
            
            Points3DList.append(Initial3D.tolist()[0])
            
            for CamName in ExistFrameData.keys():
                Points2DList.append(ExistFrameData[CamName]) #Save 2D point
                CamIndexList.append(self.CamNames.index(CamName))
                PointIndexList.append(Point3DIndexCounter)
                
            Point3DIndexCounter += 1
        self.Points3DArr = np.array(Points3DList)            
        self.PointIndexArr = np.array(PointIndexList)
        self.CamIndexArr = np.array(CamIndexList)
        self.Points2DArr = np.array(Points2DList)
        self.PointsNameList = PointNameList
        
        if Point3DIndexCounter == 0: ###no valid points
            return False
        else:
            return True
        

    def TriangulatePoints(self,CamIndexes, PointDicts):
        """Triangulate Points using 2 camera views"""
        CamNames1 = CamIndexes[0]
        CamNames2 = CamIndexes[1]
        
        Cam1ParamDict = self.CamParamDict[CamNames1]
        Cam2ParamDict = self.CamParamDict[CamNames2]
        
        # import ipdb;ipdb.set_trace()
        #Projection matrix:
        projectionMatrixCam1 = self.ComputeProjectionMatrix(Cam1ParamDict["R"],Cam1ParamDict["T"],Cam1ParamDict["cameraMatrix"])
        projectionMatrixCam2 = self.ComputeProjectionMatrix(Cam2ParamDict["R"],Cam2ParamDict["T"],Cam2ParamDict["cameraMatrix"])
        
        #Prepare 2D points
        Cam1PointsArr = np.array(PointDicts[0], dtype = np.float32).T
        Cam2PointsArr = np.array(PointDicts[1], dtype = np.float32).T
        
        triangulatedPointsHomogenous = cv2.triangulatePoints(projectionMatrixCam1,projectionMatrixCam2,Cam1PointsArr,Cam2PointsArr)
        triangulatedPointsArray = cv2.convertPointsFromHomogeneous(triangulatedPointsHomogenous.T)
        triangulatedPointsMatrix = np.matrix(triangulatedPointsArray)
        
        return triangulatedPointsMatrix
            
        
    def ComputeProjectionMatrix(self,rotationMatrix,translationMatrix,intrinsicMatrix):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x1 matrix
        :param translationMatrix: 3x1 Matrix
        :param intrinsicMatrix: 3x3 matrix
        :return: 3x4 projection matrix
        """
        rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]
        RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
        projectionMatrix = np.dot(intrinsicMatrix, RT)
        return projectionMatrix
    
    def GetResiduals(self,Params):
        """Given points and parameters, get residuals using reproject"""
        #Back transform to get back parameters and 3d points
        
        Points3D = Params.reshape((self.n_points, 3))
        points_proj = self.Reproject(Points3D[self.PointIndexArr])
        PointDiffs = points_proj - self.Points2DArr
        EucError = np.sqrt(PointDiffs[:,0]**2 + PointDiffs[:,1]**2)
        
        # import ipdb;ipdb.set_trace()
        return (points_proj - self.Points2DArr).ravel()
        # return EucError
        
    def Reproject(self, Points3DAll):
        """Given 3D points, do reprojection and get new 2d points"""
        AllPoint2D = []
        for i in range(len(Points3DAll)):
            rvec = self.CamRvecList[self.CamIndexArr[i]]
            tvec = self.CamTvecList[self.CamIndexArr[i]]
            FundMat = self.CamMatList[self.CamIndexArr[i]]
            Dist = self.DistList[self.CamIndexArr[i]]

            Point2D = cv2.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
            AllPoint2D.append(Point2D[0][0][0])

        AllPoint2DArr = np.array(AllPoint2D)
        return AllPoint2DArr
    
    def BundleAdjustmentSparsity(self, n,m):
        # m = self.CamIndexArr.size * 2
        # n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.CamIndexArr.size)
        # for s in range(6):
        #     A[2 * i, self.CamIndexArr * 6 + s] = 1
        #     A[2 * i + 1, self.CamIndexArr * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.PointIndexArr * 3 + s] = 1
            A[2 * i + 1, self.PointIndexArr * 3 + s] = 1

        return A
    
    def GetFinalParam(self,results):
        """Back transform arrays to get final param"""
        Points3D = results.reshape((self.n_points,3))
        
        FinalDict = {}
        for i in range(len(Points3D)):
            FinalDict.update({self.PointsNameList[i]:Points3D[i]})

        return FinalDict
    
    
    def run(self):
        """Run Bundle Adjustment"""
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 100,verbose=0, x_scale='jac', ftol=1e-8, method='trf')
                    # args=(self.n_cameras, self.n_points,self.CamIndexArr, self.PointIndexArr, self.Points2DArr,self.IntMat,self.Distortion))
        # plt.plot(res.fun)
        # plt.show()
        # import ipdb;ipdb.set_trace()

        FinalParamDict = self.GetFinalParam(res.x)

        return FinalParamDict
    

class BundleAdjustmentTool_Triangulation_Filter:
    """
    Bundle adjustment Tool, for triangulation
    New implementation: if high reprojection error, filter cam
    
    """

    def __init__(self, CamNames, CamParams):
        """
        CamNames: list of camera names
        CamParams: Dictionary of camera parameters (intrinsic and extrinsics)
        Points2D: Detections in 2D space
        
        """
        self.CamNames = CamNames
        self.CamParamDict = CamParams
        self.PrepareCamParams(CamParams)
        
    def PrepareCamParams(self,CamParams):
        """Reads in Dictionary of camera prameters then put it in correct format"""
        
        CamRvecList = []
        CamTvecList = []
        CamMatList = []
        DistList = []
        for i in range(len(self.CamNames)):
            Cam = self.CamNames[i]
            #Extrsinics
            rvec = CamParams[Cam]["R"]
            tvec = CamParams[Cam]["T"]
            CamRvecList.append(rvec)
            CamTvecList.append(tvec)
            
            ##Intrinsics:
            CamMatList.append(CamParams[Cam]["cameraMatrix"])
            DistList.append(CamParams[Cam]["distCoeffs"])

        self.CamMatList = CamMatList
        self.DistList = DistList
        self.CamRvecList = CamRvecList
        self.CamTvecList = CamTvecList
        

    def PrepareInputData(self,PointsDictList):
        """
        input:
        PointsDictList: Dictionary named by camera names, with inner dictionary of points 
        {PointName : [x,y]}
        
        output:
        Points3D (N_point,3): 3D coordinates of unique 3D points
        PointIndex (N_Observation,): index of which 3d points a given 2d point belongs
        CamIndex (N_Observation,) : index of which camera a given 2d point belongs
        Points2D (N_Observation,2) : 2D coordinates of given feature
        """
        ##Output Lists
        Points3DList = []
        PointIndexList = []
        CamIndexList = []
        Points2DList = []        
        
        #Get all unique frames
        Cameras = list(PointsDictList.keys())
        Point2DNameList = [list(val.keys()) for val in PointsDictList.values()]
        
        #get all unique 2Dpoints:
        AllUnqPoints = sorted(list(set().union(*Point2DNameList)))
        
        Point3DIndexCounter = 0 ##rolling count for 3d point index
        PointNameList = [] ##Kep tracks of point names
        # Initial3DPoints = []
        
        for point in AllUnqPoints:
            ##Exists 2D measurements in at least 2 views:
            ExistFrameData = {}
            for x in range(len(Cameras)):
                Cam = Cameras[x]
                if point in list(PointsDictList[Cam].keys()):
                    # PointIndex = PointsDictList[Cam]["ImgIDs"].index(point)
                    ExistFrameData.update({Cam:PointsDictList[Cam][point]})
                else:
                    continue
            if len(ExistFrameData) <2:
                ##Only 1 cam saw point
                continue
            
            #At least 2 views detected point:
            AllCamIndexes = list(ExistFrameData.keys())
            CamIndexes = (AllCamIndexes[0],AllCamIndexes[1]) #Just choose first 2 cameras
            #Initial 3D point estimate:
            PointDicts = (ExistFrameData[CamIndexes[0]],ExistFrameData[CamIndexes[1]] )
            # PointDicts = (PointsDictList[Cameras[CamIndexes[0]]][frame], PointsDictList[Cameras[CamIndexes[1]]][frame])
            Initial3D = self.TriangulatePoints(CamIndexes, PointDicts) #OverlapID is the checkboard object ids that were used in triangulation
            PointNameList.append(point)
            
            Points3DList.append(Initial3D.tolist()[0])
            
            for CamName in ExistFrameData.keys():
                Points2DList.append(ExistFrameData[CamName]) #Save 2D point
                CamIndexList.append(self.CamNames.index(CamName))
                PointIndexList.append(Point3DIndexCounter)
                
            Point3DIndexCounter += 1
        self.Points3DArr = np.array(Points3DList)            
        self.PointIndexArr = np.array(PointIndexList)
        self.CamIndexArr = np.array(CamIndexList)
        self.Points2DArr = np.array(Points2DList)
        self.PointsNameList = PointNameList
        
        if Point3DIndexCounter == 0: ###no valid points
            return False
        else:
            return True
        

    def TriangulatePoints(self,CamIndexes, PointDicts):
        """Triangulate Points using 2 camera views"""
        CamNames1 = CamIndexes[0]
        CamNames2 = CamIndexes[1]
        
        Cam1ParamDict = self.CamParamDict[CamNames1]
        Cam2ParamDict = self.CamParamDict[CamNames2]
        
        # import ipdb;ipdb.set_trace()
        #Projection matrix:
        projectionMatrixCam1 = self.ComputeProjectionMatrix(Cam1ParamDict["R"],Cam1ParamDict["T"],Cam1ParamDict["cameraMatrix"])
        projectionMatrixCam2 = self.ComputeProjectionMatrix(Cam2ParamDict["R"],Cam2ParamDict["T"],Cam2ParamDict["cameraMatrix"])
        
        #Prepare 2D points
        Cam1PointsArr = np.array(PointDicts[0], dtype = np.float32).T
        Cam2PointsArr = np.array(PointDicts[1], dtype = np.float32).T
        
        triangulatedPointsHomogenous = cv2.triangulatePoints(projectionMatrixCam1,projectionMatrixCam2,Cam1PointsArr,Cam2PointsArr)
        triangulatedPointsArray = cv2.convertPointsFromHomogeneous(triangulatedPointsHomogenous.T)
        triangulatedPointsMatrix = np.matrix(triangulatedPointsArray)
        
        return triangulatedPointsMatrix
            
        
    def ComputeProjectionMatrix(self,rotationMatrix,translationMatrix,intrinsicMatrix):
        """
        Computes projection matrix from given rotation and translation matrices
        :param rotationMatrix: 3x1 matrix
        :param translationMatrix: 3x1 Matrix
        :param intrinsicMatrix: 3x3 matrix
        :return: 3x4 projection matrix
        """
        rotationMatrix = cv2.Rodrigues(rotationMatrix)[0]
        RT = np.concatenate((rotationMatrix, translationMatrix), axis=1)
        projectionMatrix = np.dot(intrinsicMatrix, RT)
        return projectionMatrix
    
    def GetResiduals(self,Params):
        """Given points and parameters, get residuals using reproject"""
        #Back transform to get back parameters and 3d points
        
        Points3D = Params.reshape((self.n_points, 3))
        points_proj = self.Reproject(Points3D[self.PointIndexArr])
        PointDiffs = points_proj - self.Points2DArr
        EucError = np.sqrt(PointDiffs[:,0]**2 + PointDiffs[:,1]**2)
        
        # import ipdb;ipdb.set_trace()
        return (points_proj - self.Points2DArr).ravel()
        # return EucError
        
    def Reproject(self, Points3DAll):
        """Given 3D points, do reprojection and get new 2d points"""
        AllPoint2D = []
        for i in range(len(Points3DAll)):
            rvec = self.CamRvecList[self.CamIndexArr[i]]
            tvec = self.CamTvecList[self.CamIndexArr[i]]
            FundMat = self.CamMatList[self.CamIndexArr[i]]
            Dist = self.DistList[self.CamIndexArr[i]]

            Point2D = cv2.projectPoints(Points3DAll[i], rvec,tvec, FundMat,Dist)
            AllPoint2D.append(Point2D[0][0][0])

        AllPoint2DArr = np.array(AllPoint2D)
        return AllPoint2DArr
    
    def BundleAdjustmentSparsity(self, n,m):
        # m = self.CamIndexArr.size * 2
        # n = self.n_cameras * 6 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.CamIndexArr.size)
        # for s in range(6):
        #     A[2 * i, self.CamIndexArr * 6 + s] = 1
        #     A[2 * i + 1, self.CamIndexArr * 6 + s] = 1

        for s in range(3):
            A[2 * i, self.PointIndexArr * 3 + s] = 1
            A[2 * i + 1, self.PointIndexArr * 3 + s] = 1

        return A
    
    def GetFinalParam(self,results):
        """Back transform arrays to get final param"""
        Points3D = results.reshape((self.n_points,3))
        
        FinalDict = {}
        for i in range(len(Points3D)):
            FinalDict.update({self.PointsNameList[i]:Points3D[i]})

        return FinalDict
    
    def OptimizeAndFilter(self, results):
        """check reprojection error then filter out points that have high ones, then retriangulate"""
        Points3D = results.reshape((self.n_points,3))
        All3DPoints = Points3D[self.PointIndexArr]

        # import ipdb;ipdb.set_trace()
        Reproject2D = self.Reproject(All3DPoints)

        ErrorList = [self.GetEucDist(self.Points2DArr[x],Reproject2D[x]) for x in range(self.Points2DArr.shape[0])]
        ErrorArray = np.array(ErrorList)
        UpperBound  = ErrorArray.mean() + (2*np.std(ErrorArray)) #Upper bound: 2 sds
        OutlierIndex = np.where(ErrorArray>UpperBound)[0]

        if len(OutlierIndex) == 0:
            FinalParamDict = self.GetFinalParam(results)

            return FinalParamDict

        ###Delete indicies from all the arrays
        # import ipdb;ipdb.set_trace()
        self.PointIndexArr = np.delete(self.PointIndexArr,OutlierIndex)
        self.CamIndexArr =  np.delete(self.CamIndexArr,OutlierIndex)
        self.Points2DArr = np.delete(self.Points2DArr,OutlierIndex,0)

        # yo = self.Points2DArr.copy()

        ####Rerun bundle adjustment:

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 100,verbose=0, x_scale='jac', ftol=1e-8, method='trf')


        FinalParamDict = self.GetFinalParam(res.x)
        
        return FinalParamDict



    def GetEucDist(self,Point1,Point2):
        """Get euclidian error, both 2D and 3D"""
        
        if len(Point1) ==3 & len(Point2) ==3:
            EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
        elif len(Point1) ==2 & len(Point2) ==2:
            EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
        else:
            import ipdb;ipdb.set_trace()
            Exception("point input size error")
        
        return EucDist

    def run(self):
        """Run Bundle Adjustment"""
        self.n_cameras = len(list(set(self.CamIndexArr.tolist())))
        self.n_points = self.Points3DArr.shape[0]

        n = 3*self.n_points#total number of parameters to optimize
        m = 2 * self.Points2DArr.shape[0] #total number of residuals

        ##Parameters to optimize:
        Params = self.Points3DArr.ravel()
        f0 = self.GetResiduals(Params)        
        # Err = [abs(x) for x in f0]
        # sum(Err)/len(Err)
        # plt.plot(f0)
        # plt.show()        
        # import ipdb;ipdb.set_trace()

        A = self.BundleAdjustmentSparsity(n,m)
        res = least_squares(self.GetResiduals, Params, jac_sparsity=A,max_nfev = 100,verbose=0, x_scale='jac', ftol=1e-8, method='trf')
                    # args=(self.n_cameras, self.n_points,self.CamIndexArr, self.PointIndexArr, self.Points2DArr,self.IntMat,self.Distortion))
        # plt.plot(res.fun)
        # plt.show()
        # import ipdb;ipdb.set_trace()

        # FinalParamDict = self.GetFinalParam(res.x)

        FinalParamDict = self.OptimizeAndFilter(res.x)
        

        return FinalParamDict