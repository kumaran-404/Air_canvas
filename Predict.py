from keras.models import load_model
model = load_model("./models")
import numpy as np 



def predict(paintWindow):

    #converting image to gray scale
    paintWindow = paintWindow[102:398]
    a = []

    for i in range(len(paintWindow )):
        a.append( paintWindow[i][202:398])

    
#         a= cv2.resize(np.array(a), (28, 28))
#         #plt.imshow(a, interpolation='nearest')
#         #predict(a)
    a = np.array(a)
    a = a[:,:,0] 

    input = a.reshape(-1,a.shape[0])
    print(input.shape)
    return np.argmax(model.predict(input)[0])



