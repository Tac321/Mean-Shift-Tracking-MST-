# Shawn Daniel
# 09/21/2018   REMAKE
# Sweet implementation: error   : memory allocation error.  Mat m;
# initialize m or do some processing
#m.release();     // Reference: https://stackoverflow.com/questions/16817263/how-to-clear-the-cvmat-contents
# Clearing data with the a.release()  turns object into an empty "Mat" variable.   #  Yes!


# For image tracker GATE growth logic
value = 5
chFlag = False

#float YSTAR
dt1 = 0.03

uCentroid=0
vCentroid=0

# Make Disappear the crosshair.
tlimtimer = 0
# Target MST Lock key timer.
timeTillLock = 0
# Detection/Lock Mode
ckeyModel = False  # Model definition
ckeyTrack = False # tracking mode
ckeyLock  = False # enter MST mode.
# Passing targeting Scope for the image.
targ_aa = [0,0]
targ_bb = [0,0]
# Viewing window for desired MST tracking target
nameWindow = 'hehe'
tracked = None

#Centroid BLind variable, used to grow the image tracker Gate.
targPrimSeen= False

import cv2
import numpy as np
import time
import argparse
import imtools
from scipy.ndimage import filters
from scipy.ndimage.filters import convolve as corr
from tracking import *
from draw import *

cap = cv2.VideoCapture(0)   # 0 is for web came, 1 is for other attached cams.value = 0
monoInc=True
monoDec=False

while True:
    # Capture frame-by-frame
    time_kp1 = time.time()
    _, frame = cap.read()
    capWidth = frame.shape[1] 
    capHeight= frame.shape[0]  
    
    srcWidth = frame.shape[1]
    srcHeight= frame.shape[0]
    width = srcWidth
    height = srcHeight
    srcArea = srcWidth*srcHeight
    # ( (searchX,searchY), searchWidth, searchHeight )  # Paints the image tracker.\
    centerX = srcWidth//2
    centerY = srcHeight//2
    
    # load the image, clone it for output, and then convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if not ckeyLock:  # Detect Logic.
        '''
        Make the gradient magnitude binary image for increasing of contour sensitivity.
        #'''
        kerSz = 5
        blurImg = cv2.GaussianBlur(frame,(kerSz,kerSz),5) # C++ mtd: GaussianBlur(src, dst, size(3,3),1.0,1.0)
        cannyImg = cv2.Canny(blurImg,100,200)             # C++ mtd: Canny(src,dst,thrL,thrH,kSizeScalor)
    
        contours, hierarchy = cv2.findContours(cannyImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # I_binNominal , cannyImg
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)  # -1 means draw all contours
        '''
        Import Bbox#'''
        # Make the image tracker Gate dynamically changing. For filter.
        if(monoInc and targPrimSeen ==False):
            value +=1
            if value >= 70:
                monoInc = False
                monoDec = True
        if(monoDec):
            value-= 10  # This is a function of dt!
            if value <= 5:
                value = 5
                monoInc = True
                monoDec = False
        #value = 5
        search = (srcArea)**(0.5)*float(value)/100.0
        searchX = centerX - search
        searchY = centerY - search
        searchWidth = (centerX + search) - searchX
        searchHeight = (centerY + search) - searchY
        
        '''
        Now with all the Bbox Drawn ALRERADY!
        #'''
        bboxFilt = []
        '''contoursFiltNP #'''
        # Draw BBox around all targets  FILTER SECTION.
        for i in range(len(contours)):
            
            tempRect = cv2.boundingRect(contours[i])   # x,y,w,h = cv2.boundingRect(cnt)
            '''
            Bbox filter, with SIZE, EDGE location wrt image pane, and image tracker GATE size.
            #'''
            # Get C.G of each Bbox
            objectCentX = tempRect[0]+tempRect[2]//2  # Coln space!
            objectCentY = tempRect[1]+tempRect[3]//2
            bboxWidth  = tempRect[2]
            bboxHeight   = tempRect[3]
            
            # Object in image tracker gate
            if objectCentX > searchX and objectCentY > searchY and objectCentX < searchWidth+searchX and objectCentY < searchHeight+searchY:
                # Bbox Size filter. w.r.t. image tracker Gate.
                if bboxHeight < searchHeight *0.75 and bboxHeight >searchHeight//50 and bboxWidth < searchWidth*0.75 and bboxWidth >searchWidth//50:
                
                        # Image Plane edge filter.  In case image tracker is too big
                    if tempRect[0] > 2 and tempRect[1] > 2 and bboxWidth+tempRect[0] < srcWidth-2 and tempRect[1] + bboxHeight*1 < srcHeight:
                        bboxFilt.append(tempRect)

        bboxFiltNP = np.array(bboxFilt)       
        # contoursFiltNP.shape   # Amount of captured Bbox
        bbCentroid = []    # List of sorted centroids, from closest to farthest from image origin.
        '''
        Bbox Sort
        '''#
        if len(bboxFilt)>0:  
            for j in range(len(bboxFilt)):
                for i in range(len(bboxFilt)-1):   
                    rect1 = bboxFilt[i]    # coln, row, width, height
                    rect2 = bboxFilt[i+1]
                    
                    # Get Rectangle 1 distance from center.
                    rectCentX = rect1[0] + rect1[2]//2 # centroidX  
                    rectCentY = rect1[1] + rect1[3]//2 # centroidY
                    distX = (centerX - rectCentX)/float(srcWidth)
                    distY = (centerY - rectCentY)/float(srcHeight)
                    rect1Dist = (distX**2 + distY**2)**0.5
                    
                    # Get Rectangle 2 distance from center.
                    rectCentX = rect2[0] + rect2[2]//2
                    rectCentY = rect2[1] + rect2[3]//2
                    distX = (centerX - rectCentX)/float(srcWidth)
                    distY = (centerY - rectCentY)/float(srcHeight)
                    rect2Dist = (distX**2 + distY**2)**0.5
            
                    if (rect2Dist < rect1Dist):
                        rectTMP = bboxFilt[i]
                        bboxFilt[i] = bboxFilt[i+1]
                        bboxFilt[i+1] = rectTMP 
            
            # Draw all targets captured withiin the image tracker.
            bbCentroid = []
            for i in range(len(bboxFilt)):
            	tempRect = bboxFilt[i]  # Sequence of 4 numbers
            	cv2.rectangle(frame, (tempRect[0],tempRect[1] ), ((tempRect[0] + tempRect[2]), (tempRect[1] + tempRect[3])), (0, 255,0), 2) # Plot bBox
            	cv2.circle(frame,((tempRect[0] + tempRect[2]//2),(tempRect[1] + tempRect[3]//2)), 3, [0,0,255], -1)  # plot centroid
            	bbCentroid.append((tempRect[0] + tempRect[2]//2, tempRect[1] + tempRect[3]//2)) #'''
    
            # For each sorted bBox.
            # Draw User choose closest target.
            userVal = 0   # User selected index value of the target.
            tempRect = bboxFilt[userVal]
            centroidX = tempRect[0] + tempRect[2]//2
            centroidY = tempRect[1] + tempRect[3]//2
            deltaTargX = centroidX - centerX
            deltaTargY = centroidY - centerY
            
            # Lock Mode Switching Logic.
            if ((deltaTargX**2+ deltaTargY**2)**0.5<20):
                timeTillLock +=dt1
            else:
                timeTillLock = 0    
            
            if (timeTillLock>1.0):
                ckeyLock = True
                ckeyModel = True
                # Saved model scoping data below.
                targ_aa = [tempRect[1] ,tempRect[0]] # [coln, row]
                targ_bb = [tempRect[1] + tempRect[3],tempRect[0] + tempRect[2] ]
        
            # Below we plot the sorted closest detected bBox wrt imagge center!
            cv2.rectangle(frame, (tempRect[0],tempRect[1] ), ((tempRect[0] + tempRect[2]), (tempRect[1] + tempRect[3])), (255, 255,0), 2)
            cv2.circle(frame,((tempRect[0] + tempRect[2]//2),tempRect[1] + tempRect[3]//2), 3, [0,0,255], -1)
            bbCentroid.append((tempRect[0] + tempRect[2]//2, tempRect[1] + tempRect[3]//2)) #'''
           
        if (len(bboxFilt)>0) :   
            targPrimSeen = True
            chFlag = True
        else:
            targPrimSeen = False
    
        # Draw closest rectangle.
        cv2.rectangle(frame, (int(searchX),int(searchY) ), (int(searchX+searchWidth), int(searchY+searchHeight)), (0, 0,255), 3)
        # Draw image center circle.
        cv2.circle(frame,(centerX,centerY), 5, [0,255,255], 2)
        
        cv2.imshow(nameWindow,frame)
        cv2.waitKey(1)
        time_kp2 = time.time()
    
        dt1 = (time_kp2 - time_kp1)
        #print ('deltaTime', dt1)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    if ckeyLock:
        # Enter MST logic here.
        # Model Definition, enter ONCE for modle definition!
        if (ckeyModel): 
            rectUserTopLeft = np.array(targ_aa)  # [Row, column]
            rectUserBottomRight = np.array(targ_bb)
            center  = (rectUserTopLeft) #+ rectUserBottomRight*0.5)
            X_gray = extractFromAABB(frame, rectUserTopLeft, rectUserBottomRight, gray=True) 
            tracked = ResultTracking(rectUserTopLeft, rectUserBottomRight, center)
            modelDensity = hat_Qu(X_gray, indexesHistogram)
            ckeyModel = False
            ckeyTrack = True
            
        # Target LOCK and TRACKING, assuming model already defined. Enter in only after model definition.
        if(ckeyTrack):
            tracked = track(frame, tracked, modelDensity, captureWidth=capWidth, captureHeight=capHeight) # Input : newVideoStream, previousTargetbBoxLocation, modelHistogram, cameraImagePaneBoundaries
            drawTracking(frame, tracked)
            cv.imshow(nameWindow, frame)  # PLAY DIFFERENT NAME.
            
            if tracked.BC < 0.40:  # Then MST has lasst target trackk w.r.t. model definition.
                # Reset all boolean keys to prepare for next time we enter MST logic
                ckeyModel = False
                ckeyTrack = False
                ckeyLock  = False 
                
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # 'Esc' key
                break
   
cv2.destroyAllWindows()
cap.release()










