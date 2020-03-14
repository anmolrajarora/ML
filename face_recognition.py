import cv2
import numpy as np

data_1 = np.load('anmol.npy').reshape(101,50*50*3)
data_1 = data_1[:-1]
data_2 = np.load('ranveer.npy').reshape(100,50*50*3)

users = {
    0 : "Anmol",
    1 : "Ranveer"
}

data = np.concatenate([data_1, data_2])

labels = np.zeros(len(data))
labels[100:] = 1

def calculateDistance(x, stored_image):
    return np.sqrt( np.sum( (x - stored_image) ** 2 ) )

def knn(x, data, k=5):
    
    n = data.shape[0]
    distances = []
    
    for i in range(n):
        distance = calculateDistance(x, data[i])
        distances.append(distance)

    sortedIndex = np.argsort(distances)
    req_labels = labels[sortedIndex][:k]

    counts = np.unique(req_labels, return_counts = True)
    argmax = np.argmax(counts[1])
    return counts[0][argmax]

capture = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    ret, image = capture.read()
    if ret:
        faces = cascade.detectMultiScale(image)
        for x,y,w,h in faces:
            cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0), 5)

            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (50,50))

            label = knn(face.flatten(), data)

            name = users[int(label)]
            
            #image   text   coordinates fontFamily  fontScale color fontWeight
            cv2.putText(image, name, (x,y), font, 1, (0,255,0), 2)
        cv2.imshow('image.jpg',image)
        if cv2.waitKey(1) & 0xff == 27:
            break
capture.release()
cv2.destroyAllWindows()
        
    


'''3 x 3

123,124,123,123,124,123,123,124,123,123,124,123,123,124,123,123,124,123,123,124,123,123,124,123,123,124,123

(123,124,10) (123,124,10) (123,124,10)
(123,124,123) (123,124,123) (123,124,123) 
(123,124,123) (123,124,123) (123,124,123)'''

