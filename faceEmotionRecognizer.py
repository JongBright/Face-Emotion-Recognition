import cv2
from deepface import DeepFace



image = cv2.imread('Test files/4.jpeg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imageInfo = DeepFace.analyze(img, actions=['emotion'])
faceEmotion = imageInfo['dominant_emotion'].capitalize()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
faceDimensions = []
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
    faceDimensions.append(x)
    faceDimensions.append(y)

cv2.putText(image, str(faceEmotion), (faceDimensions[0], faceDimensions[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)



cv2.imshow('Image', image)
cv2.waitKey(0)


