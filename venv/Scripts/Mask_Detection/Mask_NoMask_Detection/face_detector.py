# face detection with mtcnn on a photograph
import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN


class Face:

    def draw_faces(self, filename, result_list):
        # load the image
        data = pyplot.imread(filename)
        bounding_box = []
        # plot each face as a subplot
        for i in range(len(result_list)):
            # get coordinates
            d = dict()
            x1, y1, width, height = result_list[i]['box']
            x2, y2 = x1 + width, y1 + height
            img = data[y1:y2, x1:x2]
            d['x1'] = x1
            d['y1'] = y1
            d['x2'] = x2
            d['y2'] = y2
            d['img'] = img
            bounding_box.append(d)

        return bounding_box

    def face_detect(self, filename):
        # load image from file
        pixels = cv2.imread(filename)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        for face in faces:
            print(face)

        boundings = self.draw_faces(filename, faces)
        return boundings
