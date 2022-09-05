

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


    def setWindow(self):
        paintWindow = np.zeros((int(self.height),int(self.width),3)) 
        cv2.rectangle(paintWindow,(200,100),(400,400),(255,0,0),2)
        #clear rectangle 
        cv2.rectangle(paintWindow , (25,25) , (150,80),(255,0,0),2)
        cv2.putText(paintWindow,"clear", (55,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2,cv2.LINE_AA)
        #predict rectangle 
        cv2.rectangle(paintWindow ,(500,25) ,(625,80), (255,0,0),2  )
        cv2.putText(paintWindow,"predict", (515,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2,cv2.LINE_AA)
        return paintWindow

    # starts webcam
    def Capture(self):
        text ="No-Write Mode"
        prev = None 
        paintWindow = self.setWindow()
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


            if(coordinates):

                #clear the window 
                if(self.clearBoard(coordinates)):
                    paintWindow = self.setWindow()
                
                #predict output 
                self.predict(coordinates,paintWindow)


                if(self.IsThumbBehindIndex(coordinates) and self.InsideBoundary(coordinates)):
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
      
#         paintWindow = paintWindow[102:398]
#         a = []
# #         plt.imshow(paintWindow, interpolation='nearest')
# # #	print(paintWindow.shape)
#         for i in range(len(paintWindow )):
#             a.append( paintWindow[i][202:398])
        
    
# #         a= cv2.resize(np.array(a), (28, 28))
# #         #plt.imshow(a, interpolation='nearest')
# #         #predict(a)
#         a = np.array(a)
#         a = a[:,:,0]
#         k = im.fromarray(np.array(a)) 
#         k = k.convert('RGB')
#         k.save("test.png")
#         plt.imshow(np.array(a),  interpolation='nearest')
#         plt.show()
        self.video.release()

    def predict(self,coordinates,PaintWindow):
        Predict = coordinates[0][0]>=500 and coordinates[0][1]>=25 and coordinates[0][0]<=625  and coordinates[0][1]<=80 

        if(Predict):
            result = predict(PaintWindow)
            cv2.putText(PaintWindow,"Prediction is "+str(result), (200,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2,cv2.LINE_AA)

            


    def clearBoard(self,coordinates):
        print(coordinates)
        if(coordinates[0][0]>=25 and coordinates[0][1]>=25 and coordinates[0][0]<=150 and coordinates[0][1]<=80 ):
            return True 
        return False 

    def InsideBoundary(self,coordinates):
        return coordinates[0][0]>=200 and coordinates[0][1]>=100 and coordinates[0][0]<=400 and coordinates[0][1]<=400 

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
