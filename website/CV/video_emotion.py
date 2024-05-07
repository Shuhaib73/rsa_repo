
import cv2 as cv
import numpy as np 
from tensorflow.keras.models import load_model


class EmotionDetectionReal:
    def __init__(self, model_path, cascade_path):
        self.model = load_model(model_path)
        self.face_cascade = cv.CascadeClassifier(cascade_path)
        self.labels_dic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def detect_real_emotion(self):

        video = cv.VideoCapture(0)

        while True:
            ret, frame = video.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

            for x, y, w, h in faces:
                sub_face_img = gray[y:y+h, x:x+w]
                
                resized = cv.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))  #len(num_of_img), img_h, img_w, img_color
                
                result = self.model.predict(reshaped)

                label=np.argmax(result, axis=1)[0]
                print(label)

                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                cv.rectangle(frame, (x, y), (x+w, y+h), (50, 255, 50), 2)
                cv.rectangle(frame, (x, y-40), (x+w, y), (50, 255, 50), -1)

                cv.putText(frame, self.labels_dic[label], (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("Emotion Detection", frame)
            k = cv.waitKey(1)

            if k==ord('q'):
                break
        video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    Model = 'Emotion_Detection_Model.h5'
    FaceModel = 'haarcascade_frontalface_default.xml'

    detector = EmotionDetectionReal(Model, FaceModel)

    detector.detect_real_emotion()