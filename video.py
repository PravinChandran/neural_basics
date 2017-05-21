

import cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
print "OpenCV Version : %s " % cv2.__version__



cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    canny        = 1
    thresholding = 0

    if canny:
        thresh_img = cv2.Canny(imgray,100,200)
        #fig=plt.figure()
        #plt.imshow(thresh_img, cmap='gray')

    if thresholding:
        kernel = np.ones((5,5),np.uint8)
        imgray_dialate = cv2.dilate(imgray,kernel,iterations = 2)
        imgray_erode = cv2.erode(imgray_dialate,kernel,iterations = 2)

        imgray_processed = imgray_erode
        ret, thresh_img = cv2.threshold(imgray_processed,100,255,cv2.THRESH_BINARY)
    
    img_out = imgray.copy()
    #img_empty = np.zeros_like(thresh_img)
    #print img_empty.shape

    contours, hierarchy = cv2.findContours(thresh_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    font_size=0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    #fig = plt.figure()

    #----------------------------------#
    #Processing Contours: Pick top few #
    #----------------------------------#
    #areaArray=[]
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    areaArray.append(area)
    #sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    #top_contours = [sorteddata[0][1], sorteddata[1][1], sorteddata[2][1]]
     
        
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w>10 and h>10:
            #print x,y,w,h
            rect = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
        
            rect_dim = ",".join([str(int(x)) for x in rect[1]])
            rect_loc = (int(rect[0][0]), int(rect[0][1]))
            rect_loc = (box[3][0], box[3][1])
            cv2.drawContours(img_out, [box], 0, (255,0,0), 3)
            cv2.putText(img_out,rect_dim, rect_loc, font, font_size,(255,255,255),2)   
    
        
            #plt.subplot(211);
            #plt.imshow(img_out)
            count+=1
    
    #print count  
    
    
    # Display the resulting frame
    #cv2.imshow('frame',thresh_img)
    cv2.imshow('frame',img_out)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
        
        

    
    
    
    
    
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

