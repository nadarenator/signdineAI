import sys
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model 
from mp_utils import Pipeline

def main(args):
    actions = np.array(['pizza', 'burger', 'salad', 'soup', 'sphagetti', 'chicken', 'fish', 'vegetables', 'fruits', 'turkey', 
    'pork', 'hotdog', 'cheese', 'macaroni', 'pepperoni', 'lasagna', 'sharing', 'platter', 'honey', 'garlic', 'sticks', 'fusion', 'hot', 
    'fiery', 'ham', 'salt', 'egg', 'seafood', 'clam', 'mushroom', 'soft', 'crab', 'thai', 'red', 'wings', 'crispy', 'waffle'])
    #var to hold prediction:
    result = []
    #30 frames for a prediction:
    sequence = []
    #sentence to render last n predicted words:
    sentence = []
    #corpus to store all predicted words:
    corpus = []
    #threshold to validate predictions:
    threshold = 0.9
    #mediapipe pipeline:
    p = Pipeline()
    #LSTM model:
    model = load_model("models/LSTM_model.h5")
    cap = cv2.VideoCapture(0)
    #mediapipe model
    with p.mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            #read feed
            ret, frame = cap.read()
            #make detection
            image, results = p.mediapipe_detection(frame, holistic)
            print(results)
            #draw landmarks
            p.draw_styled_landmarks(image, results)
            #prediction logic:
            keypoints = p.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                result = model.predict(np.expand_dims(sequence, axis = 0))[0]
                print(actions[np.argmax(result)])
            #visualization logic:
            if len(result) != 0 and result[np.argmax(result)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(result)] != sentence[-1]:
                        sentence.append(actions[np.argmax(result)])
                        corpus.append(actions[np.argmax(result)])
                else:
                    sentence.append(actions[np.argmax(result)])
                    corpus.append(actions[np.argmax(result)])
            if len(sentence) > 5:
                sentence = sentence[-5:]
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #show to screen
            cv2.imshow('OpenCV Feed', image)
            #break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)