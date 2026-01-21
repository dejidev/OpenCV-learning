import os
import cv2 as cv
import numpy as np


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'../Resources/Faces/train/'

# p = []

# for i in os.listdir(r'../Resources/Faces/train/'):
#     p.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)


            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)
            # If no faces are detected, skip adding to features and labels

create_train()


print('Training done ---------------')


print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer.create()


#Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

np.save('features.npy', features)
np.save('labels.npy', labels)



