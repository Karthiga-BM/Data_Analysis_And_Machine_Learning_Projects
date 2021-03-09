import os
import cv2
import numpy as np
from face_detector import Face
from imutils import paths
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

out = cv2.VideoWriter('Mask_No_Mask_Detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (500, 500))


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(img, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def testmask(Data):
    for data in Data:
        path = data[0]
        label = data[1]
        print(path)
        bounding_box = face.face_detect(path)
        if len(bounding_box) > 0:
            for index in range(len(bounding_box)):
                img = preprocess(bounding_box[index]['img'])
                predictions = model.predict(img)  # (mask,nomask)
                # determine the class label and color we'll use to draw
                # the bounding box and text
                for pred in predictions:
                    p_label = "Mask" if pred[0] > pred[1] else "No Mask"
                    #print("Prediction: ", p_label, label)
                bounding_box[index]['pred'] = p_label
                bounding_box[index]['actual'] = label
                bounding_box[index]['path'] = path

            show_data(bounding_box)


def show_data(boundings):
    for index in range(len(boundings)):
        img = cv2.imread(boundings[index]['path'])
        img = np.asarray(img)
        col = (0, 0, 255) if boundings[index]['pred'] == 'No Mask' else (14, 150, 14)
        cv2.rectangle(img, (boundings[index]['x1'], boundings[index]['y1']),
                      (boundings[index]['x2'], boundings[index]['y2']), col, 2)

        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (15, 5), (200, 50), (224, 206, 206), cv2.FILLED)

        if boundings[index]['actual'] == boundings[index]['pred']:
            if boundings[index]['pred'] == 'No Mask':
                cv2.putText(img, "Access Denied", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            else:
                cv2.putText(img, "Access Granted", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        else:
            print()
        cv2.imshow("Mask/No Mask Detection", img)
        cv2.waitKey(1000)
        out.write(img)


face = Face()
imagePaths = list(paths.list_images("Dataset/test"))
labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]

labels = ['Mask' if x == 'with_mask' else 'No Mask' for x in labels]
labels = np.array(labels)
imagePaths = np.array(imagePaths)

print("[INFO] creating the testing dataset...")
labels = labels.reshape(labels.shape[0], 1)
ImagePaths = imagePaths.reshape(imagePaths.shape[0], 1)
Dataset = np.append(ImagePaths, labels, axis=1)
# Shuffling the dataset by rows
np.random.shuffle(Dataset)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model('detector_model')
testmask(Dataset)
out.release()
