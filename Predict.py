import joblib
model = joblib.load("./model.pkl")
# from keras.models import load_model
import numpy as np 
from PIL import Image



def predict(image):
	
    #converting image to gray scale
    grayScale = np.array([])
    for i in image:
        for j in i :
            factor = 0.2989 * j[0] + 0.5870 * j[1] + 0.1140 * j[2]
            grayScale= np.append(grayScale,factor) 
    # image  = grayScale.reshape((28,28))
    # image = cv2.convert("RGB")

    input = grayScale.reshape(-1,grayScale.shape[0])
    print(input.shape)
    print("prediction:",np.argmax(model.predict(input)[0]))


  
