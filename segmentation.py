import cv2
import numpy as np

def segmentation_objet_mvmnt(seuil, operation_morphologique = False):
    '''
    :param seuil:
    :param operation_morphologique:
    :return: objects segmentés
    convertir la canal luminosité (V) en 255.
    '''
    cap = cv2.VideoCapture("video.mp4")

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255 #Saturation à 255
    hsv[...,2] = 255 #Luminosité à 255
    #Définition de la valeur de seuille pour la segmentation
    g = 0
    while(1):
        g +=1
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        #hsv[...,0] = ang*180/np.pi/2
        hsv[...,0] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        _,hsv[...,0] = cv2.threshold(hsv[...,0], seuil, 255, cv2.THRESH_BINARY)
        #Application de l'érosion et de la dilatation
        if operation_morphologique:
            kernel = np.ones((10,10),np.uint8)
            hsv[...,0] = cv2.erode(hsv[...,0],kernel, iterations = 1)
            hsv[...,0] = cv2.dilate(hsv[...,0],kernel, iterations = 1)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # Redimensionner les 2 images avant de les mettre cote à  cote
        imstack1 = cv2.resize(bgr,(500,500))
        imstack2 = cv2.resize(frame2,(500,500))
        imstack_final = np.hstack((imstack2,imstack1))
        cv2.imshow('frame2',imstack_final)
    
        k = cv2.waitKey(1) & 0xff
        # q pour quitter 
        if k == ord('q'):
            break
        elif g == 200 :
            cv2.imwrite('opticalfb.png',imstack_final)
            #cv2.imwrite('opticalhsv.png',bgr)
        prvs = next
    cv2.destroyAllWindows()

# Tester avec différentes valeurs: 200, 150, 100 50 25 10
#segmentation_objet_mvmnt(10)

# Ameliorer resultat avec des transformations morphologiques
segmentation_objet_mvmnt(15, True)




