import cv2
import numpy as np


cap = cv2.VideoCapture("video.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # Redimensionner les 2 images avant de les mettre cote à  cote
    imstack1 = cv2.resize(bgr,(600,600))
    imstack2 = cv2.resize(frame2,(600,600))
    imstack_final = np.hstack((imstack2,imstack1))
    cv2.imshow('frame2',imstack_final)

    k = cv2.waitKey(1) & 0xff
    # q pour quitter
    if k == ord('q'):
        break
    elif k == ord('a'):
        cv2.imwrite('opticalfb.png',imstack_final)
        #cv2.imwrite('opticalhsv.png',bgr)
    prvs = next
cv2.destroyAllWindows()