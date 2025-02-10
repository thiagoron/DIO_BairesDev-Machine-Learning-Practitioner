import numpy as np
import cv2 as cv
import os

# Carregar o classificador de faces
face_cascade = cv.CascadeClassifier()
if not face_cascade.load(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    print("Erro ao carregar o classificador de faces")
    exit()

# Criar um modelo de classificação para os rostos
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Carregar as imagens de treinamento
thiago_images = []
milena_images = []
thiago_labels = []
milena_labels = []

# Carregar as imagens de treinamento para Thiago
for file in os.listdir("thiago"):
    img = cv.imread(os.path.join("thiago", file))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thiago_images.append(gray)
    thiago_labels.append(0)

# Carregar as imagens de treinamento para Milena
for file in os.listdir("milena"):
    img = cv.imread(os.path.join("milena", file))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    milena_images.append(gray)
    milena_labels.append(1)

# Treinar o modelo de classificação
faces = thiago_images + milena_images
labels = thiago_labels + milena_labels
face_recognizer.train(faces, np.array(labels))

# Abrir a câmera
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi)
        if label == 0:
            name = "Thiago"
        elif label == 1:
            name = "Milena"
        else:
            name = "Desconhecido"

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()