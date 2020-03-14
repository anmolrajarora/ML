import cv2

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("img_42.jpg")

faces = dataset.detectMultiScale(img, 1.25)

#print(faces)

for x,y,w,h in faces:
    #cv2.rectangle(image, start_point (x,y), end_point (width, height), color (bgr), thickness)
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5)
    #cv2.putText(image, text, (x,y), font, fontScale, color(bgr)[, thickness
    cv2.putText(img, 'Person', (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

cv2.imshow("result", img)



