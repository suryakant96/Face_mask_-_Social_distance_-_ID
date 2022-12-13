


import tensorflow as tf
import numpy as np




mask=tf.keras.models.load_model('my_model.h5')
mask.summary()




import cv2
from time import time
import matplotlib.pyplot as plt

opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="face-detection-caffee/models/deploy.prototxt",caffeModel="face-detection-caffee/models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


def cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.5, display=True):
    image_height, image_width, _ = image.shape

    output_image = image.copy()

    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)

    start = time()

    results = opencv_dnn_model.forward()

    end = time()

    cropbox = []

    for face in results[0][0]:

        face_confidence = face[2]

        if face_confidence > min_confidence:
            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)
            cropbox.append([[x1, y1], [x2, y2]])

    return results, cropbox
    
cam=cv2.VideoCapture("Test.avi")
i=0
flag=0

while (cam.isOpened()):
    ret,img = cam.read()

    rslt, cbox =cvDnnDetectFaces(img, opencv_dnn_model, display=True)
    if len(cbox) > 1:
        cv2.putText(img, text="Maintain_Social_Distance", org=(100, 50), fontScale=1,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=2)
    elif len(cbox) == 1:
        flag = 1

        x1 = cbox[0][0][1] - 50
        if x1 < 0:
            x1 = 0

        x2 = cbox[0][1][1] + 50
        y1 = cbox[0][0][0] - 50
        if y1 < 0:
            y1 = 0
        y2 = cbox[0][1][0] + 50
    
        imgc=img.copy()
        cropped_image = imgc[x1:x2, y1:y2]
        cv2.imwrite("image.jpg",cropped_image)
        image = tf.keras.utils.load_img(r"image.jpg",target_size=(100,100,3))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = mask.predict(input_arr)
        print(predictions)
        
        # cv2.rectangle(img, pt1=(cbox[0][0][0], cbox[0][0][1]), pt2=(cbox[0][1][0], cbox[0][1][1]), color=(0, 255, 0),
        #               thickness=5)
        if predictions>=0.5:
            cv2.putText(img, text="No_Mask", org=(cbox[0][0][0]-50, cbox[0][0][1]), fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=2)
            cv2.rectangle(img, pt1=(cbox[0][0][0], cbox[0][0][1]), pt2=(cbox[0][1][0], cbox[0][1][1]), color=(0, 0, 255),thickness=5)
            cv2.putText(img, text="Person" + str(i), org=(cbox[0][1][0] - 50, cbox[0][0][1]), fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=2)
        elif predictions<0.5:
            cv2.putText(img, text="Mask", org=(cbox[0][0][0]-50, cbox[0][0][1]), fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=2)
            cv2.rectangle(img, pt1=(cbox[0][0][0], cbox[0][0][1]), pt2=(cbox[0][1][0], cbox[0][1][1]), color=(0, 255, 0),thickness=5)
            cv2.putText(img, text="Person" + str(i), org=(cbox[0][1][0] - 50, cbox[0][0][1]), fontScale=1,fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255), thickness=2)


    elif len(cbox) == 0:

        if flag == 1:
            i = i + 1

            flag = 0

    cv2.imshow("Output",img)

    if cv2.waitKey(1) & 0Xff==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()





