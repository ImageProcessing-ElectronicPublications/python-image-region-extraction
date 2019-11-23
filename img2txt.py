#!/usr/bin/env python
import sys
import cv2
import numpy as np
#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#from PIL import Image
#import image

def imagetoregion(imagename, imagedest, segmentview):
    image = cv2.imread(imagename)
    cv2.imshow('orig',image)
    cv2.waitKey(0)

    #Initial Processing of the image starts...!!!!!!!!!!!

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    cv2.waitKey(0)

    #binary
    ret,thresh = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('Binarized',thresh)
    cv2.waitKey(0)

    #detecting edges>>>>

    edges=cv2.Canny(thresh,140,200)
    cv2.imshow("edges",edges)
    cv2.waitKey(0)

    #dilation
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    cv2.imshow('dilated',img_dilation)
    cv2.waitKey(0)

    #Localization Horizontal projection
    (x,y)=img_dilation.shape
    z=[sum(y) for y in img_dilation]
    Tx=(((sum(z))/(len(z))/20))
    #print Tx

    #Vertical Projection
    ndilation=zip(*img_dilation)
    X=[sum(row) for row in ndilation]
    mean=(sum(X))/(len(X))
    maximofX=(max(X)/10)
    Ty=mean+maximofX
    #print Ty

    #Adaptive threashold for horizontal  projection
    th1 = cv2.adaptiveThreshold(img_dilation,Tx,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) #block size and constant.

    #Adaptive threashold for vertical projection
    th2 = cv2.adaptiveThreshold(img_dilation,Ty,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    fthreashold=cv2.add(th1,th2)
    cv2.imshow("threashold1",fthreashold)
    cv2.waitKey(0)

    #Initial Processing of the image finishes....!!!!!

    #find contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height
        x, y, w, h = cv2.boundingRect(ctr)

        #Removing the false area that are not textes.
        if w<35 and h<35:
            continue

        if segmentview:
            # Getting ROI
            roi = image[y:y+h, x:x+w]

            #result = pytesseract.image_to_string(Image.open(roi))

            # show ROI
            #cv2.imshow(roi)
            cv2.imshow('segment no:'+str(i),roi)
            cv2.waitKey(0)

        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),1)

        #print result

    cv2.namedWindow('marked areas', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('marked areas', 800, 600)
    cv2.imshow('marked areas',image)
    cv2.waitKey(0)
    if not(segmentview):
        cv2.imwrite(imagedest, image)

    '''
    result = pytesseract.image_to_string(Image.open('final.jpg'))
    with open('fiel12.txt',mode='w') as file:
        file.write(result)
        print("Done")
    '''

if __name__ == '__main__':
    segmentview = True
    filename = 'test.jpg'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    filedest = 'final.jpg'
    if len(sys.argv) > 2:
        filedest = sys.argv[2]
        segmentview = False
    imagetoregion(filename, filedest, segmentview)
