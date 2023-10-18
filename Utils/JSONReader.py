# !/usr/bin/env python3
"""JSON reader class to read images from images sampled from 3DPOP"""

import json
import cv2
import os
import pickle
import numpy as np

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




class JSONReader:
    
    def __init__(self, JSONPath,DatasetPath, Type = "3D"):
        """
        Initialize JSON reader object
        JSONPath: path to json file
        DatasetPath: Path to dataset root directory to read images
        Type: 2D or 3D, based om which type was read     
        """
        
        with open(JSONPath) as f:
            self.data = json.load(f)
        
        self.DatasetPath = DatasetPath
        self.Type = Type
        self.Info = self.data["info"]
        self.PrintInfo()
        self.Annotations = self.data["Annotations"]
        
        
    def PrintInfo(self):
        print("Loading JSON...")
        print(self.Info["Description"])
        print("Collated by: %s on: %s" %(self.Info["Collated by"],self.Info["Date"]))
        print("Total Images: %s"%(self.Info["TotalImages"]))
        
    def Extract3D(self,index):
        """Extract 3D data"""
        return self.Annotations[index]["Keypoint3D"]
        
    def Extract2D(self,index):
        """Extract 2D data"""
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "Keypoint2D"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["Keypoint2D"]
            
        return Out

    def ExtractBBox(self,index):
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "BBox"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["BBox"]

        return Out
    
    def GetImagePath(self,index):
        if self.Type == "3D":
            CameraInfoDict = self.Annotations[index]["CameraData"]
            Out = [[val for key,val in SubDict.items() if key == "Path"][0] for SubDict in CameraInfoDict]
        else:
            Out = self.Annotations[index]["Path"]
        return Out
    
    def GetSequenceCode(self,FileName):
        return FileName.split("-")[0]
    
    def GetIntrinsics(self, index):
        
        #get sequence from file path        
        CamInfo = self.data["Annotations"][index]["CameraData"]
        
        SequenceCode = self.GetSequenceCode(os.path.basename(CamInfo[0]["Path"]))
        
        camMatList = []
        distCoefList= []
        
        for x in range(4): ##For 3D pop, always 4 cam
            Cam = x+1
            Intpath = os.path.join(self.DatasetPath,"Calibration","%s-Cam%s-Intrinsics.p"%(SequenceCode,Cam))
            camMat, distCoef = pickle.load(open(Intpath,"rb"))
            camMatList.append(camMat)
            distCoefList.append(distCoef)
            
        return camMatList, distCoefList

    def GetExtrinsics(self, index):
        #get sequence from file path        
        CamInfo = self.data["Annotations"][index]["CameraData"]
        
        SequenceCode = self.GetSequenceCode(os.path.basename(CamInfo[0]["Path"]))
        
        rvecList = []
        tvecList = []
        
        for x in range(4): ##For 3D pop, always 4 cam
            Cam = x+1
            Extpath = os.path.join(self.DatasetPath,"Calibration","%s-Cam%s-Extrinsics.p"%(SequenceCode,Cam))
            rvec, tvec = pickle.load(open(Extpath,"rb"))
            rvecList.append(rvec)
            tvecList.append(tvec)
            
        return rvecList, tvecList
    
    def GetGTArray(self,Indexes):
        """
        Get array of all annotation ground truth
        shape: (N,9,3)
        Indexes is list of index of image
        """
        GTList = []
        for i in Indexes:
           GT = np.array(list(self.Extract3D(i).values()))
           GTList.append(GT)
        
        GTArray = np.stack(GTList)
        # import ipdb;ipdb.set_trace()

        return GTArray
    
    def CheckAnnotations(self, index, show=True):
        if self.Type == "3D":
            ImgPath = self.GetImagePath(index)[0]
            Key2D = self.Extract2D(index)[0]
            BBox = self.ExtractBBox(index)[0]
        else:
            ImgPath = self.GetImagePath(index)
            Key2D = self.Extract2D(index)
            BBox = self.ExtractBBox(index)
        # import ipdb;ipdb.set_trace()
        
        RealImgPath = os.path.join(self.DatasetPath,ImgPath)

        img = cv2.imread(RealImgPath)
        
        ##Draw keypoints:
        for BirdID,Key2DDict in Key2D.items():
            for key, pts in Key2DDict.items():
                point = (round(pts[0]),round(pts[1]))
                colour = getColor(key)
                cv2.circle(img,point,3,colour, -1)
            BBoxData = BBox[BirdID]
            cv2.rectangle(img,(round(BBoxData[0]),round(BBoxData[1])),(round(BBoxData[2]),round(BBoxData[3])),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, BirdID,(round(BBoxData[0]),round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img



##Function to read JSON, rewrite new format with unique index for each individual
def ExtractPerIndividualJSON(JSONPath,DatasetPath,Type):
    """Function to read JSON, rewrite new format with unique index for each individual"""
    Dataset = JSONReader(JSONPath,DatasetPath, Type = "3D")

    DictList3D = []
    MasterIndexCounter = 0
    
    for i in range(len(Dataset.Annotations)):
        frameData = Dataset.Annotations[i]
        
        for bird in frameData["BirdID"]:
            cameraDataList = []
            for x in range(4):
                cam = "Cam%s"%(x+1)
                cameraDataList.append({
                    "CamName": cam,
                    "Path" : frameData["CameraData"][x]["Path"],
                    "BBox": frameData["CameraData"][x]["BBox"][bird],
                    "Keypoint2D" : frameData["CameraData"][x]["Keypoint2D"][bird]
                })
            
            DictList3D.append({
                "ImageID" : MasterIndexCounter,
                "BirdID" : bird,
                "Keypoint3D" : frameData["Keypoint3D"][bird],
                "CameraData": cameraDataList
            })
            
            MasterIndexCounter +=1
            
    OutputDict3D = {
        "info" : {
        "Description":"Re-structured 3D keypoint data from 3DPOP, per individual",
        "Collated by": "Alex Chan",
        "Date":"22/02/2023",
        "Keypoints": Dataset.Info["Keypoints"],
        "TotalImages": MasterIndexCounter-1
    },
      "Annotations":DictList3D}
    # import ipdb;ipdb.set_trace()
    with open(os.path.join(DatasetPath,'Annotation',"%s-3D-Individual.json"%Type), "w") as outfile:
        json.dump(OutputDict3D, outfile, indent=4)


##Function to read JSON, rewrite new format with unique index for each individual seperated by individual number, for inference speed estimation
def ExtractPerIndividualJSONSplitbyGroupSize(JSONPath,DatasetPath,Type, IndNumber = 1):
    """Function to read JSON, rewrite new format with unique index for each individual"""
    Dataset = JSONReader(JSONPath,DatasetPath, Type = "3D")

    DictList3D = []
    MasterIndexCounter = 0
    
    for i in range(len(Dataset.Annotations)):
        frameData = Dataset.Annotations[i]
        if len(frameData["BirdID"]) == IndNumber:
            for bird in frameData["BirdID"]:
                cameraDataList = []
                for x in range(4):
                    cam = "Cam%s"%(x+1)
                    cameraDataList.append({
                        "CamName": cam,
                        "Path" : frameData["CameraData"][x]["Path"],
                        "BBox": frameData["CameraData"][x]["BBox"][bird],
                        "Keypoint2D" : frameData["CameraData"][x]["Keypoint2D"][bird]
                    })
                
                DictList3D.append({
                    "ImageID" : MasterIndexCounter,
                    "BirdID" : bird,
                    "Keypoint3D" : frameData["Keypoint3D"][bird],
                    "CameraData": cameraDataList
                })
                
                MasterIndexCounter +=1
            else:
                continue

            
    OutputDict3D = {
        "info" : {
        "Description":"Re-structured 3D keypoint data from 3DPOP, per individual, sperated by group size",
        "Collated by": "Alex Chan",
        "Date":"24/04/2023",
        "Keypoints": Dataset.Info["Keypoints"],
        "TotalImages": MasterIndexCounter-1
    },
      "Annotations":DictList3D}
    # import ipdb;ipdb.set_trace()
    IndTypeName = '%02d'%IndNumber
    with open(os.path.join(DatasetPath,'Annotation',"%s-3D-Individual_Ind%s.json"%(Type,IndTypeName)), "w") as outfile:
        json.dump(OutputDict3D, outfile, indent=4)



        
if __name__ == "__main__":
    JSONPath = "/media/alexchan/Extreme SSD/SampleDatasets/ImageTrainingData/N6000/Annotation/Test-3D.json"
    DatasetPath = "/media/alexchan/Extreme SSD/SampleDatasets/ImageTrainingData/N6000"
    Dataset = JSONReader(JSONPath,DatasetPath, Type = "3D")
    len(Dataset.Annotations)



    ExtractPerIndividualJSONSplitbyGroupSize(JSONPath,DatasetPath,Type="Test", IndNumber = 1)
    ExtractPerIndividualJSONSplitbyGroupSize(JSONPath,DatasetPath,Type="Test", IndNumber = 2)
    ExtractPerIndividualJSONSplitbyGroupSize(JSONPath,DatasetPath,Type="Test", IndNumber = 5)
    ExtractPerIndividualJSONSplitbyGroupSize(JSONPath,DatasetPath,Type="Test", IndNumber = 10)
    


    # ExtractPerIndividualJSON(JSONPath,DatasetPath, Type = "Train")
    # JSONPath = "/home/alexchan/Documents/SampleDatasets/ImageTrainingData/N100/Annotation/Test-3D.json"
    # ExtractPerIndividualJSON(JSONPath,DatasetPath, Type = "Test")
    # JSONPath = "/home/alexchan/Documents/SampleDatasets/ImageTrainingData/N100/Annotation/Train-3D.json"
    # ExtractPerIndividualJSON(JSONPath,DatasetPath, Type = "Train")

    # Dataset = JSONReader(JSONPath,DatasetPath, Type = "3D")
    # Dataset.CheckAnnotations(510)
    # import ipdb;ipdb.set_trace()
    # Dataset.Extract2D(1)
    
    
    
    # camMatList, distCoefList = Dataset.GetIntrinsics(0)
    # rvecList, tvecList = Dataset.GetExtrinsics(0)
    # # Dataset.CheckAnnotations(3249)
    # Dataset.CheckAnnotations(3250)
    # Dataset.CheckAnnotations(3251)
    # Dataset.CheckAnnotations(3305)
    # Dataset.CheckAnnotations(3306)
    # Dataset.CheckAnnotations(6325)
    # Dataset.CheckAnnotations(6326)
    # Dataset.CheckAnnotations(6327)
    
    
    # Dataset.CheckAnnotations(9391)
    # Dataset.CheckAnnotations(9562)
    # Dataset.CheckAnnotations(9904)
    # Dataset.CheckAnnotations(10417)
    # Dataset.CheckAnnotations(10418)
    # Dataset.CheckAnnotations(10930)

    # import ipdb;ipdb.set_trace()
    # Dataset.GetImagePath(3249)
    
    ##Temp: test when y coord for bbox is both 0
    # from tqdm import tqdm
    # for i in tqdm(range(len(Dataset.Annotations))):
    #     # print(i)
    #     BBox = Dataset.ExtractBBox(i)
    #     # import ipdb;ipdb.set_trace()
    #     XCoords = [abs(box[2] - box[0]) for box in BBox.values() ]
    #     YCoords = [abs(box[3] - box[1]) for box in BBox.values()]

    #     if any(x == 0 or x < 0 for x in XCoords):
    #         print(i)
    #         # import ipdb;ipdb.set_trace()
    #     elif any(y==0 or y<0 for y in YCoords):
    #         # import ipdb;ipdb.set_trace()
    #         print(i)
    #     else:
    #         continue
        
        