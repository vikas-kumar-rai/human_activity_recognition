from flask import Flask
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)


def predict():
    print('predict')
    model = load_model('human_activity_recognitation.h5')
    vid = cv2.VideoCapture(0)
    count = 0
    lst = []
    text = ''
    label = {0: 'YOGA  - tree pose', 1: 'YOGA - goddess pose', 2: 'walking'}

    while (True):

        ret, frame = vid.read()
        count += 1
        test_image = cv2.resize(frame, (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        img_class = model.predict_classes(test_image)
        lst.append(img_class[0])

        if count % 3 == 0:
            text = label[int(np.max(lst))]
            count = 0
            lst = []

        cv2.putText(frame, f"{text}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

        cv2.imshow('frame', cv2.resize(frame, (1080, 720)))

        # the 'q' button is set as the
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    predict()
