"""Compute BBox Overlap"""
import numpy as np
import sys
sys.path.append("./Utils")
from HungarianAlgorithm import hungarian_algorithm


def GetBBoxOverlap(BBox1,BBox2):
    """
    Calculate bounding box overlap between 2 boxes, inputted as list, as [x1,x2,y1,y2]
    
    Outputs percentage overlap
    """
    dx = min(BBox1[2], BBox2[2]) - max(BBox1[0], BBox2[0])
    dy = min(BBox1[3], BBox2[3]) - max(BBox1[1], BBox2[1])
    if (dx>=0) and (dy>=0):
        #Area of ind 1:
        Area1 = (BBox1[2]-BBox1[0])*(BBox1[3]-BBox1[1])
        if Area1 == 0:
            return 0
        return (dx*dy)/Area1
    else:
        return 0
    


def MatchID_BBox_Hungarian(GT_BBox,result,confidence_threshold=0.5,i=None):
    """Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap and hungarian algorithm"""

    
    Pred_BBoxList = result["boxes"].to('cpu').numpy().tolist()
    
    IndexBelowThreshold = np.where(result["scores"].to("cpu").numpy() < confidence_threshold)[0].tolist()
    

    Filtered_PredBBox = [box for j, box in enumerate(Pred_BBoxList) if j not in IndexBelowThreshold ]
    # GT_BBoxList = [val for val in GT_BBox.values()]

    
    OutDict = {}
    AssingedIndex = []
    NameList = []
    PercentOverlapMatrixList = []
    
    for key,val in GT_BBox.items() :
        NameList.append(key)
        PercentOverlapList  = [GetBBoxOverlap(Pred_BBox,val) for Pred_BBox in Filtered_PredBBox]
        PercentOverlapMatrixList.append(PercentOverlapList)
    # import ipdb;ipdb.set_trace()

    if len(GT_BBox) != len(Filtered_PredBBox):
        ###If ground truth and predicted bbox doesnt have same length,
        ### Remove ground truth bbox with most mean error, probably cant match that
        for x in range((len(GT_BBox)-len(Filtered_PredBBox))):
            MaxOverlap = [np.array(row).max() for row in PercentOverlapMatrixList ]
            IndexMinMax = np.argmin(np.array(MaxOverlap)) #find index of which GT has lowest max overlap
            PercentOverlapMatrixList.pop(IndexMinMax)
            NameList.pop(IndexMinMax)

    ##Get distance Matrix
    PercentOverlapMat = 1-np.array(PercentOverlapMatrixList) #1 - overlap matrix, to for hungarian to minimize 
    MatchOut = hungarian_algorithm(PercentOverlapMat)
    OutDict = {NameList[index[0]]:index[1] for index in MatchOut }
    
    return OutDict




def MatchID_BBox_Hungarian_Basic(GT_BBox,Filtered_PredBBox,confidence_threshold=0.5,i=None):
    """
    Given ground truth bbox and model predictions, assign ID to prediction based on BBox overlap and hungarian algorithm
    
    Basic version, reads in list of prediction bboxes
    """

    
    # Pred_BBoxList = result["boxes"].to('cpu').numpy().tolist()
    
    # IndexBelowThreshold = np.where(result["scores"].to("cpu").numpy() < confidence_threshold)[0].tolist()
    

    # Filtered_PredBBox = [box for j, box in enumerate(Pred_BBoxList) if j not in IndexBelowThreshold ]
    # # GT_BBoxList = [val for val in GT_BBox.values()]

    
    OutDict = {}
    AssingedIndex = []
    NameList = []
    PercentOverlapMatrixList = []
    
    for key,val in GT_BBox.items() :
        NameList.append(key)
        PercentOverlapList  = [GetBBoxOverlap(Pred_BBox,val) for Pred_BBox in Filtered_PredBBox]
        PercentOverlapMatrixList.append(PercentOverlapList)
    # import ipdb;ipdb.set_trace()

    if len(GT_BBox) != len(Filtered_PredBBox):
        ###If ground truth and predicted bbox doesnt have same length,
        ### Remove ground truth bbox with most mean error, probably cant match that
        for x in range((len(GT_BBox)-len(Filtered_PredBBox))):
            MaxOverlap = [np.array(row).max() for row in PercentOverlapMatrixList ]
            IndexMinMax = np.argmin(np.array(MaxOverlap)) #find index of which GT has lowest max overlap
            PercentOverlapMatrixList.pop(IndexMinMax)
            NameList.pop(IndexMinMax)

    ##Get distance Matrix
    PercentOverlapMat = 1-np.array(PercentOverlapMatrixList) #1 - overlap matrix, to for hungarian to minimize 
    MatchOut = hungarian_algorithm(PercentOverlapMat)
    OutDict = {NameList[index[0]]:index[1] for index in MatchOut }
    
    return OutDict