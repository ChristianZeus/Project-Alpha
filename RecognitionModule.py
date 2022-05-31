
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
import cv2 as cv
import mediapipe as mp
import numpy as np
import os

class CameraProess():
    def __init__(self):
        self.model = load_model('Model/LFW_Version1.h5')
        self.threshold = 0.9655
        self.labels = os.listdir('Data/train')
        self.L = 20
        self.t = 3
        detec = mp.solutions.face_detection
        self.face_detection = detec.FaceDetection(min_detection_confidence = 0.75) #or FaceDetection(0.75) default = 0.5

    # Take an input image and ang return face and location in image
    def detection(self,image_rgb): 
        try:
            height, width, channels = image_rgb.shape
            FACES = self.face_detection.process(image_rgb)
            if FACES.detections is not None:
                for id, value in enumerate(FACES.detections):
                    bounding_boxC = value.location_data.relative_bounding_box
                    bounding_box = int(bounding_boxC.xmin*width), int(bounding_boxC.ymin*height), \
                                int(bounding_boxC.width*width), int(bounding_boxC.height*height)
                    x,y,w,h = bounding_box
                    x1,y1 = x+w, y+h
                    face = image_rgb[y:y+h, x:x+w]
                    #Top left X Y
                    cv.line(image_rgb, (x,y), (x+self.L,y),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                    cv.line(image_rgb, (x,y), (x,y+self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                    #Top right X1 Y
                    cv.line(image_rgb, (x1,y), (x1-self.L,y),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                    cv.line(image_rgb, (x1,y), (x1,y+self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                    #Bottom, left X Y1
                    cv.line(image_rgb, (x,y1), (x+self.L,y1),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                    cv.line(image_rgb, (x,y1), (x,y1-self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                    #Bottom right X1 Y1
                    cv.line(image_rgb, (x1,y1), (x1-self.L,y1),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                    cv.line(image_rgb, (x1,y1), (x1,y1-self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                    #Predict Image
                    return face
            else:
                return None
        except Exception as e:
            return None

    # Predict image
    def predict_image(self, face_image, threshold = 0.9655):
        try:
            predict_img = cv.resize(face_image, dsize = (224,224))
            predict_img = predict_img.reshape(1,224,224,3)
            input_img = preprocess_input(predict_img)
            prediction = self.model.predict(x = input_img, steps = 1, verbose= 0)
            index = int(np.argmax(prediction, axis=1))
            predict_value = float(np.max(prediction, axis=1))
            if predict_value >= threshold:
                name = self.labels[index]
            else:
                name = 'Indefinite'
        except Exception as e:
            name = 'Indefinite'
        return name
        

    # Take an image and return recognition image
    def process_image(self, img):
        try: 
            bounding_box = self.detection(img)
            if bounding_box is not False:
                x,y,w,h = bounding_box 
                x1,y1 = x+w, y+h
                #Top left X Y
                cv.line(img, (x,y), (x+self.L,y),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                cv.line(img, (x,y), (x,y+self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                #Top right X1 Y
                cv.line(img, (x1,y), (x1-self.L,y),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                cv.line(img, (x1,y), (x1,y+self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                #Bottom, left X Y1
                cv.line(img, (x,y1), (x+self.L,y1),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                cv.line(img, (x,y1), (x,y1-self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)

                #Bottom right X1 Y1
                cv.line(img, (x1,y1), (x1-self.L,y1),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                cv.line(img, (x1,y1), (x1,y1-self.L),color=(0,255,0), thickness=self.t, lineType=cv.LINE_AA)
                #Predict Image
                face_image = img[y:y+h, x:x+w]
                recognition = self.predict_image(face_image, self.threshold)
                return img, recognition
            else:
                return img, None
        except Exception as e:
            return img, None
            

if __name__ == "__main__":
    camera = CameraProess()
    img = cv.imread('jisoo.jpg')
    img = camera.detection(img)
    value = camera.predict_image(img)
    print(value)
    cv.imshow("MediaNets", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

