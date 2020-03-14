import cv2

#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture("video_1.mp4")
dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, img = capture.read()
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    if ret:
        faces = dataset.detectMultiScale(img, 1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            #cv2.putText(image, text, (x,y), font, fontScale, color(bgr)[, thickness
            cv2.putText(img, 'Person', (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow("result", img)
    if cv2.waitKey(1) & 0xff == 27:
        break
#cv2.imwrite("result.jpg", img)

capture.release()
cv2.destroyAllWindows()
