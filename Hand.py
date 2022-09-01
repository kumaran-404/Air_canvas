

import cv2
import mediapipe as mp
import numpy as np 
from collections import deque 
from PIL import Image as im
from matplotlib import pyplot as plt
from Predict import predict 


class HandDetection :

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hand = self.mp_hands.Hands(max_num_hands=1)
        self.video  = cv2.VideoCapture(0)
        self.width = self.video.get(3)
        self.height = self.video.get(4)
        self.results = []
        self.image = None 
        self.isWriteMode = False 
        self.bpoints = [deque(maxlen=1024)]
        print(self.width,self.height)

    # starts webcam
    def Capture(self):
        text ="No-Write Mode"
        prev = None 
        paintWindow = np.zeros((int(self.height),int(self.width),3)) 
        cv2.rectangle(paintWindow,(200,100),(400,400),(255,0,0),2)
        while(True):
            success,image = self.video.read()

            if(not success):
                print("Error opening !")
                break 
            #Flip Image 
            image = cv2.flip(image,1)
            #Converting to rgb 
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            self.results = self.hand.process(image)

            #store image 
            self.image = image 

            if(self.results.multi_hand_landmarks):
                for i in self.results.multi_hand_landmarks :
                    self.mp_drawing.draw_landmarks(image,i,self.mp_hands.HAND_CONNECTIONS)

        
            coordinates = self.getFingerTip()
            # print(coordinates)
            if(coordinates):
                if(self.IsThumbBehindIndex(coordinates)):
                    self.isWriteMode = True 
                    text = "Write mode"
                    if(prev==None):
                        cv2.circle(paintWindow, coordinates[0], 6, (255, 255, 255),-1)
                        prev = coordinates[0]
                    else :
                        cv2.line(paintWindow,prev,coordinates[0],(255, 255, 255), 8)
                        cv2.circle(paintWindow, coordinates[0], 6, (255, 255, 255),-1)
                        prev = coordinates[0]                    
                    
                else :
                    self.isWriteMode = False 
                    prev = None 
                    text = "No write mode"
                    
                    

            txt = cv2.putText(image,text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2,cv2.LINE_AA)
            
            # cv2.imshow("paint",paintWindow)
            # cv2.imshow("ArtBoard",image)
            # cv2.imshow("ArtBoard",txt)
            
            bld = np.uint8(0.6*(paintWindow)+0.4*(image))
            cv2.imshow("Blend Mode",bld)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        paintWindow = paintWindow[100:400]
        a = []

#	print(paintWindow.shape)
        for i in range(len(paintWindow )):
            a.append( paintWindow[i][100:500])
            
        a= cv2.resize(np.array(paintWindow), (28, 28))
        predict(a)
        plt.imshow(a, interpolation='nearest')
        plt.show()
        self.video.release()

   

    def getFingerTip(self):
            coordinates = None 
            if(self.results.multi_hand_landmarks):
                hand= self.results.multi_hand_landmarks[0]
                index = hand.landmark[8]
                thumb = hand.landmark[4]
                index = np.multiply(np.array((index.x,index.y)) , np.array((self.width,self.height))).astype(int)
                thumb = np.multiply(np.array((thumb.x,thumb.y)) , np.array((self.width,self.height))).astype(int)
                coordinates = tuple((list(index),list(thumb,)))

            return coordinates
    
    def IsThumbBehindIndex(self,coordinates):
        return coordinates[0][0]<coordinates[1][0]  

cv2.destroyAllWindows()



hands = HandDetection()
hands.Capture()
