import cv2
import numpy as np

capture = cv2.VideoCapture(0)
dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data = []

while True:
    ret, img = capture.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 5)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (50,50))
            data.append(face)
            
        if len(data) >= 100:
            break
        
        cv2.imshow('detecting', img)
        
    if cv2.waitKey(1) &  0xff == 27:
        break

data = np.asarray(data)
np.save('anmol.npy', data)
capture.release()
cv2.destroyAllWindows()
