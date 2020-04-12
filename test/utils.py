import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import os
import time
import glob
import argparse
import math

def rect_to_bb(rect):
    x,y = rect.left(),rect.top()
    w,h = rect.right()-x , rect.bottom()-y
    return (x,y,w,h)

def shape_to_np(shape,dtype = "int"):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x,shape.part(i).y)
    return coords

def bb2circle(rect):
    (x1,y1,x2,y2) = rect
    x = (x2+x1)//2
    y = (y2+y1)//2
    r = int(math.sqrt((x2-x1)**2+(y2-y1)**2))
    return (x,y,r)

class preprocess:
    def __init__(self,img):
        self.img = img.copy()
        self.landmarks = None
        self.landmarksbb = None
        self.haarbb = None
        self.skinmask = None

    def faceDetect(self,img=None,xmlDIR=None):
        if xmlDIR is None:
            xmlDIR = 'haarcascade_frontalface_default.xml'
        if img is None:
            copyimg = self.img.copy()
        face_cascade = cv2.CascadeClassifier(xmlDIR)
        gray = cv2.cvtColor(copyimg,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.2,4)
        self.haarbb = faces

        for (x,y,w,h) in faces:
            cv2.rectangle(copyimg,(x,y),(x+w,y+h),(255,0,0),2)

        return faces

    def faceLandmark(self,img=None,datDIR=None,
            earlyreturn=False,draw=True):
        if datDIR is None:
            datDIR = 'shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(datDIR)
        if img is None:
            copyimg = self.img.copy()
        else:
            copyimg = img.copy()
        gray = cv2.cvtColor(copyimg,cv2.COLOR_BGR2GRAY)
        rects = detector(gray,1)
        if img is None:
            self.landmarksbb = rects

        landmarks = []
        for (i,rect) in enumerate(rects):
            shape = predictor(gray,rect)
            shape = shape_to_np(shape)
            landmarks.append(shape)
            
            (x,y,w,h) = rect_to_bb(rect)
            if draw:
                for (x,y) in shape:
                    cv2.circle(copyimg,(x,y),1,(0,0,255),-1)
        if img is None:
            self.landmarks = landmarks
        if earlyreturn:
            return landmarks,copyimg

        return copyimg

    def skinMask(self,img=None,mode='hsv'):
        if img is None:
            copyimg = self.img.copy()
        else:
            copyimg = img.copy()
        if mode.lower() == 'hsv':
            imgnewspace = cv2.cvtColor(copyimg,cv2.COLOR_BGR2HSV)
            skinRegion = cv2.inRange(imgnewspace,(0,58,30),(33,255,255))
        elif mode.lower() == 'ycrcb':
            imgnewspace = cv2.cvtColor(copyimg,cv2.COLOR_BGR2YCR_CB)
            skinRegion = cv2.inRange(imgnewspace,(0,133,77),(235,173,127))
        else: 
            print('Incorrect 2nd argument.')
            raise TypeError
        skinbit = cv2.bitwise_and(copyimg,copyimg,mask=skinRegion)
        if img is None:
            self.skinmask = skinbit
        return skinbit
    
    def non_max_suppression(self,overlapThresh=.3):
        if (landmarksbb is None) and (haarbb is None): return []
        pass

    def refineCrop(self,haarbb):
        refineimg = self.img.copy()
        (x,y,w,h) = haarbb
        cropsmall = refineimg[y:y+h,x:x+w,:].copy()
        smallmask = self.skinMask(cropsmall,mode='ycrcb')
        pixgreater0 = np.prod((smallmask)>0,axis=2)
        ret = (np.sum(pixgreater0)/np.prod(pixgreater0.shape)>.4)
        if not ret: return ret, []

        h_=refineimg.shape[0]-y if int(h*1.2)>refineimg.shape[0] else int(h*1.2)
        w_=refineimg.shape[1]-x if int(w*1.2)>refineimg.shape[1] else int(w*1.2)
        croplarge = refineimg[y:y+h_,x:x+w_,:].copy()
        
        landmarks,markimg = self.faceLandmark(croplarge,
                earlyreturn=True,draw=True)
        landmarks = np.array(landmarks[0])
        hmax,wmax = max(landmarks[:,1]),max(landmarks[:,0])
        hmin,wmin = min(landmarks[:,1]),min(landmarks[:,0])
        
        refineimg = markimg[hmin:hmax,wmin:wmax,:].copy()
        x1,x2 = x+wmin,x+wmax
        y1,y2 = y+hmin,y+hmax

        return ret,refineimg,(x1,y1,x2,y2)

    def run(self):
        '''
        1. Detect faces via haar features
        2. Filter out non-faces via skin colors
        3. Refine Cropping via face landmarks
        '''
        facedetect = self.faceDetect()
        print('Number of bounding boxes via haar: ',len(self.haarbb))

        plt.figure(1,figsize=(10,5))
        faces = []
        for i in range(len(self.haarbb)):
            ret,indfaceland,xy12 = self.refineCrop(self.haarbb[i])
            if ret:
                plt.subplot(2,4,i+1)
                plt.imshow(indfaceland[:,:,::-1])
                plt.axis('off')
                faces.append(xy12)
        
        plt.figure(2,figsize=(10,5))
        showimg = self.img.copy()
        for (x1,y1,x2,y2) in faces:
            cv2.rectangle(showimg,(x1,y1),(x2,y2),(255,0,0),2)
        plt.imshow(showimg[:,:,::-1])

        plt.figure(3,figsize=(10,5))
        showimg = self.img.copy()
        for (x1,y1,x2,y2) in faces:
            (x,y,r) = bb2circle((x1,y1,x2,y2))
            cv2.circle(showimg,(x,y),r,(0,255,0),thickness=5)
        plt.imshow(showimg[:,:,::-1])
        plt.show()
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract faces')
    parser.add_argument('--data-dir',required=True,type=str,default=None,
            help='location of image data')
    args = parser.parse_args()
    assert args.data_dir is not None

    DIR = args.data_dir
    img = cv2.imread(DIR)
    
    pp = preprocess(img)
    skin = pp.skinMask(mode='ycrcb')
    facedetect = pp.faceDetect()
    faceland = pp.faceLandmark()
    _ = pp.run()
    
