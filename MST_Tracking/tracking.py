



import numpy as np

from plot import *

import scipy.ndimage

from draw import *




# RGB to luminance

indexesHistogram = np.arange(0, 5)  # I like 10! Higher numbers track the velocity of target too slow!

class ResultTracking:
    def __init__(self, aa=None, bb=None, center=None, BC=0):
        self.aa = aa if np.all(aa) is not None else np.array([0, 0])
        self.bb = bb if np.all(bb) is not None else np.array([0, 0])
        self.center = center if np.all(center) is not None else np.array([0, 0])
        self.BC = BC;

def deltaKron(a):
    a = np.equal(a, 0).astype(int)
    return a

def k(x):
    # x  is the meshplot norm^2 !!!, Used to weight neighbours by closeness.
    # Epanechnikov kernel
    d = 2  # dimension 2, b/c we look at intensity target feature space?
    mask = np.less(x, 1).astype(int)  # Boolean argument, in 'int' form. is Image "x" < 1? If yes the 1, else 0..
    #a = mask  #  Uniform
    #a = 0.5 * (3.14) * (d + 2) * (1-x) * mask # Epanechnikov
    # a = 3.0/4.0 * (1-x) * mask   # Much Darker Kernel. Epanechnikov
    a = ((2*3.14)**-0.5) * np.exp(-0.5*x)   # Gaussian Kernel. I Like this onee!
    # plotImageWithColorBar(a)
    # pause()
    return a

def getNormXY(X, Y):
    """
    :param X: Indexes in X
    :param Y: Indexes in Y
    :return: Euclidean norm of X and Y
    """
    return np.sqrt(X**2 + Y**2)

def g(x):
    gradY, gradX = gradient(k(x))
    # plotImageWithColorBar(-gradY, title='gradX')
    # plotImageWithColorBar(-gradX, title='gradY')
    # pause()
    return -gradY, -gradX

def getCenteredCoord(X, normalize=True):
    """
    :return: Y and X (coln,row) indexes centered at 0
    """
    shapeY_half = int(X.shape[0] // 2)  # Fair approximation.
    shapeX_half = int(X.shape[1] // 2)
    startY = -shapeY_half if (X.shape[0] % 2 == 0) else -shapeY_half - 1
    startX = -shapeX_half if (X.shape[1] % 2 == 0) else -shapeX_half - 1

                                       # -50   # 50
    matX, matY = np.meshgrid(np.arange(startX, startX + X.shape[1]), np.arange(startY, startY + X.shape[0]))
    normFactY = float(shapeY_half) if normalize else 1.0  # float form of half of MST viewing rectangle.
    normFactX = float(shapeX_half) if normalize else 1.0
    return np.array([matY/normFactY, matX/normFactX])

def hat_Qu(X_gray, u):  # Input is the gray image patch.  u == indexesHistogram = arange[0,5]
    XiCentered = getCenteredCoord(X_gray)  # Vector of normalized meshplots
    #print"qHat Xgray: ", X_gray.shape
    #print"qHat XcenteredCord 0,1: ", XiCentered[0].shape, " , " ,XiCentered[1].shape
    normXi = getNormXY(XiCentered[1], XiCentered[0]) # Euclidean norm of meshplots. # Como 1.0 - Gaussian weight.
    normWeight = k(normXi**2)   # Uniform, Gaussian, Epan weight created for  our MST targeting window.
    C = 1.0/np.sum(normWeight)   # C = 1/totalMass

    density = []
    for i in u:  # From index 0 to 99, within length 100 histogram bin inventory.
        dKron = deltaKron(b(X_gray)-i)  # Binary image of values, 1.0 for values fitting in histogram bin  number "i"
        density.append(C * np.sum(normWeight * dKron))  
        ''' # Weighted sum of all pixels within local viewing window w.r.t. 
        the target! NOTE: Sum Normalized by weight matrix massTotal. 
        Thus, this is like C.G.
        '''
    return np.array(density)  # Histogram as np.array of weighted quantized feature space definition.


def hat_Pu(X_gray, Ycenter, u, hx, hy):         # Similar to hat_Qu except normalized meshplot is divided by half window again!
    #hx = 1
    #hy=1
    XiCentered = getCenteredCoord(X_gray)  # Normalized mesh grid.
    normXi = getNormXY(XiCentered[1]/hx , XiCentered[0]/hy)
    normWeight = k(normXi**2) #Kernel wehgiht creation.
    C = 1.0/np.sum(normWeight)

    density = []
    for i in u:
        dKron = deltaKron(b(X_gray)-i)
        density.append(C * np.sum(normWeight * dKron))
    return np.array(density)

def binHistoLuminance(X_gray):
    """
    :return: bin histogram of rgb luminance
    """
    gray_image_histo = X_gray.ravel()
    hist, bins = np.histogram(gray_image_histo, bins=255, range=(0, 255))
    return hist.astype(int)

def b(X_gray):  # Normlize target patch with respects to Custom histogram size.
    # plotHistoCurve((X_gray/255.0*indexesHistogram.shape[0]))
    # plotHistoCurve((X_gray/255.0*1))
    # pause()
    return np.floor((X_gray/255.0*indexesHistogram.shape[0]))


def extractFromAABB(Img, aa, bb, gray=False):
    data = Img[int(aa[0]):int(bb[0]), int(aa[1]):int(bb[1]), :] # RGB image patch of todos in MST rectangle.
    return cv.cvtColor(data, cv.COLOR_BGR2GRAY) if gray else data  # Return gray scale target.
    # Note: Lumanence heavily hinders intensity feature space, I think HSV (Hue) would be a stronger measure to track.

def weight(X_gray, hat_qu, hat_pu, u):
    res = np.zeros(X_gray.shape[0:2])  # Zero matrix, size of  X-gray =! Target image patch.
    for i in u:  # For each bin number.
         # res, is sum of images.
        res += 0 if hat_pu[i] == 0 else deltaKron(b(X_gray) - i) * np.sqrt(hat_qu[i]/(hat_pu[i]+0.00000000001))  # Hopefully prevent sim crash with divide by zero.
        print " res:    ",res

    return res

def colorToCoord(X):
    return np.array([np.arange(0, X.shape[0]), np.arange(0, X.shape[1])])

def gradient(X):
    im = X.astype(float)
    sobelFilter = np.array([-0.5, 0, 0.5]) * -1 # times -1 for low to high gradient. shape=> (3L,), 1D vector.
    gradient_y = scipy.ndimage.convolve(im, sobelFilter[np.newaxis])  # sobelFilter[np.newaxis] 2D vector ofo shape: (1L,3L)  
    gradient_x = scipy.ndimage.convolve(im, sobelFilter[np.newaxis].T)
    return gradient_y, gradient_x


def track(frame, previousTracking, modelDensity, captureWidth, captureHeight):
    ''' Note: frame has been updated, so modelDensity != Y0_minus_X_coords 
        Where Y0_minus_X_coords is in same position as the true model in the last iteration.'''
    """
    Process tracking in intensity feature space.
    """
    hx = np.abs(previousTracking.bb[1] - previousTracking.aa[1]) / 2  # float division
    hy = np.abs(previousTracking.bb[0] - previousTracking.aa[0]) / 2
    epsilon = 1000 #7 # soeed limit of the MST.
    #frame = np.asarray(frame)
    frameCpy = frame.copy()

    # Given: previous targeting scope of target,...
    X_gray = extractFromAABB(frame,   previousTracking.aa, previousTracking.bb, gray=True)  # Extract target patch. is same location as the model was taken the last "frame" video feed!
    hat_qu = modelDensity  # This is the model descriptor in feature space.

    # pause()
    # Used for g(||(y -x)/h||**2)
    Y0_minus_X_coords = getCenteredCoord(X_gray, False)  # non-Normalized meshgrid.
    norm_Y0_minus_X = getNormXY(Y0_minus_X_coords[1]/hx, Y0_minus_X_coords[0]/hy) # Now the meshgrid has been normalized.
    '''
    gradKernelY, gradKernelX = g(norm_Y0_minus_X**2)
    # '''
    # Note: The below functions according to my common sense.
    gradKernelY = k(norm_Y0_minus_X**2)
    gradKernelX = gradKernelY
    # cv.imshow('gradKernelY ',gradKernelY)   # The center is white like Epanichnekov kernal but it looks like a vertical fading half moon.
    # cv.imshow('gradKernelX ',gradKernelX)   # Horizontal fading moon.
    # cv.waitKey(1)

    # Target model, of past image frame.
    Y0 = previousTracking.center.copy()  # Picture frame off current defined copy!
    Y0_AA = previousTracking.aa.copy()
    Y0_BB = previousTracking.bb.copy()

    # Target candidate   # IC.  # Assuming the target is still in the next camera frame, INITIALLY.
    Y1 = Y0.copy()
    Y1_AA = previousTracking.aa.copy()
    Y1_BB = previousTracking.bb.copy()
    nbIterBatta = 0

    while True:
        Y0_color_gray = extractFromAABB(frame, Y0_AA, Y0_BB, gray=True) # CurrentMean Tracking rectangle. Time consuming?
        hatPU_Y0 = hat_Pu(Y0_color_gray, Y0, indexesHistogram, hx, hy)  # Histogram of targetScope location assuming target is still.
        pY0 = np.sum(np.sqrt(hatPU_Y0 * hat_qu)) # This is Bhattacharya coefficient.  hatPU_Y0*hat_qu
        
        if pY0<0.20:
            print "lost, Bhattach7775: ", pY0

        weightX = weight(X_gray, hat_qu, hatPU_Y0, indexesHistogram)
        wi_g_X = gradKernelX*weightX
        wi_g_Y = gradKernelY*weightX   # This is : wi (y0)= sum{delta[S(xi) - u]*sqrt(Qu/Pu(y0) )}
        # cv.imshow('wi_g_X ',wi_g_X)   # With fading half moon as a "mask/binoculors" view real time weights of histogram cound of pixels similar to the model hat_Qu
        # cv.imshow('wi_g_Y ',wi_g_Y)   #weight
        # cv.waitKey(1)

        Y1_y = np.sum(Y0_minus_X_coords[0]*wi_g_Y) / np.sum(wi_g_Y)  # row space. of the Non-normalized meshgrid.
        Y1_x = np.sum(Y0_minus_X_coords[1]*wi_g_X) / np.sum(wi_g_X)  # coln space.
        # Below is the weighted sum vector!
        Y1_x = 0 if np.isnan(Y1_x) else Y1_x  # NOTE: inf * 0 = nan.
        Y1_y = 0 if np.isnan(Y1_y) else Y1_y

        # print("Mean shift y1.x : %f" % Y1_x)
        # print("Mean shift y1.y : %f" % Y1_y)
        '''
        # Here is the maximum unrestrained MST track Vector!
        #'''
        Y1_AA[1] = Y0_AA[1] + Y1_x   # top left corner update with MS in coln space
        Y1_AA[0] = Y0_AA[0] + Y1_y   # top left corner update with MS ni row space
        Y1_BB[1] = Y0_BB[1] + Y1_x   # Bottom right corner update with MS in coln space
        Y1_BB[0] = Y0_BB[0] + Y1_y
        Y1[1] = Y0[1] + Y1_x         # Center update with mean shifft.
        Y1[0] = Y0[0] + Y1_y

        # Keep the MS inside the video resolution
        if Y1_AA[1] < 0 or Y1_BB[1] > captureWidth or Y1_AA[0] < 0 or Y1_BB[0] > captureHeight:
            print("Warning: mean shift outside the the capture dimension")
            return ResultTracking(aa=Y0_AA, bb=Y0_BB, center=Y0, BC=pY0)  
        ''' # return, for further printing, last known MST position  ignoring MST vector. And exit this permanent while loop.
        # If any updated MST rectangle, updated with 'mean shift vector'  drifts outside video, then ignore last 
                                        # mean shift vector update! '''

        # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(255, 255, 255))
        # drawRectangle(frameCpy, previousTracking.aa, previousTracking.bb, color=(0, 0, 0))
        # cv.imshow("res2", frameCpy)

        '''
        # After calculating the mean shifted position
        Before making final decision, see if the new location is closer to model than the previous location, iterate untile your sure!
        #''' 
        Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)  # New crop of the next video frame.
        hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, hx, hy)  # New density histogram descriptor. Of the mean shifted position.
        pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu))  # This is the Bhattacharya

        # print("pY1 %f " % pY1)
        # print("pY0 %f " % pY0)
        # print("batta diff: %f" % (pY0 - pY1))
        # and np.linalg.norm(Y1 - Y0) > 0.00000001
        '''
        # if new position hasn't stayed still, and new position less like the model than the 'still' estimate,
            then iteratively guess closer to 'still' position. THIS PREVENTS ESTIMATE OVERSHOOT.
           
        # NOTE: If new position == old position, then there was no MST vector, Y1 == Y0, target is still. 
            Assume this was the case. And rightfully so.
        '''
         
        '''
        while pY1 < pY0 and Y1[0] != Y0[0] and Y1[1] != Y0[1]: # Make sure MST shifted Bhattacharya is closer to one than pY0
            Y1 = (Y0 + Y1) * 0.5   # Take the average of the two centroid positions . This means the candidate position is'nt as SURE as we thought
            Y1_AA = (Y0_AA + Y1_AA) * 0.5
            Y1_BB = (Y0_BB + Y1_BB) * 0.5

            Y1_color_gray = extractFromAABB(frame, Y1_AA, Y1_BB, gray=True)  # Extract new  avged mean shift patch location.
            hatPU_Y1 = hat_Pu(Y1_color_gray, Y1, indexesHistogram, hx, hy) # New histogram  descriptor
            pY1 = np.sum(np.sqrt(hatPU_Y1 * hat_qu)) # Recalculate  pY1, praying it's bigger than pY0, meaning more certain than the last/pY0 position
            # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 0, msColorRect))
            # cv.imshow("res2", frameCpy)
            nbIterBatta += 1
            if nbIterBatta> 5:
                nbIterBatta = 0;
                break   # exit while loop // '''
            
        # NOTE: pY1 > pY0 means new location is more sure than the old one, meaning interest patch moved.
        # If result found i.e distance between Y1 and Y0 less than a threshold, return tracking result
        if np.linalg.norm(Y1 - Y0) < epsilon:  
            '''If the difference between Y1 and Y0 is BOUNDED, then return new MST position.'''
            #print("nbIterBatta %d" % nbIterBatta)  # Tells you how far off pY1 was.
            # print("FOUND")
            # drawRectangle(frameCpy, Y1_AA, Y1_BB, color=(0, 255, 0))
            # cv.imshow("res2", frameCpy)
            # cv.waitKey(0)
            return ResultTracking(aa=Y1_AA, bb=Y1_BB, center=Y1, BC=pY0)

        # Else candidate become the model by mean shift
        Y0 = Y1.copy()
        Y0_AA = Y1_AA.copy()
        Y0_BB = Y1_BB.copy()        
        
        
        
        
def trackzzz(frame, previousTracking, modelDensity, captureWidth, captureHeight):
    ''' Note: frame has been updated, so modelDensity != Y0_minus_X_coords 
        Where Y0_minus_X_coords is in same position as the true model in the last iteration.'''
    """
    Process tracking in intensity feature space.
    """
    hx = np.abs(previousTracking.bb[1] - previousTracking.aa[1]) / 2  # float division
    hy = np.abs(previousTracking.bb[0] - previousTracking.aa[0]) / 2

    hat_qu = modelDensity  # This is the model descriptor in feature space.

    # Target model, of past image frame. 
    Y0_AA = previousTracking.aa.copy()
    Y0_BB = previousTracking.bb.copy()

    #  Note: second value is a junk value.
    Y0_color_gray = extractFromAABB(frame, Y0_AA, Y0_BB, gray=True) # CurrentMean Tracking rectangle. Time consuming?
    hatPU_Y0 = hat_Pu(Y0_color_gray,Y0_AA, indexesHistogram, hx, hy)  # Histogram of targetScope location assuming target is still.
    pY0 = np.sum(np.sqrt(hatPU_Y0 * hat_qu)) # This is Bhattacharya coefficient.  hatPU_Y0*hat_qu
    
    return ResultTracking(aa=previousTracking.aa, bb=previousTracking.bb, center=previousTracking.center, BC=pY0)
      
        
        
        
        