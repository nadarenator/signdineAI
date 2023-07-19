import sys
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model 
from mp_utils import Pipeline

def main(args):
    actions = np.array(['kebab', 'chicken', 'beef', 'impossible', 'burrito', 'veggie', 'quesadilla', 'hummus', 'guacamole', 'cheese', 'bowl', 'salmon', 'tacos'])
    #30 frames for a prediction:
    sequence = []
    #buffer to store intermediate predictions:
    buffer = []
    #sentence to render last n predicted words:
    sentence = []
    #corpus to store all predicted words:
    corpus = []
    #threshold to validate predictions:
    threshold = 0.95
    frequency =5
    #mediapipe pipeline:
    p = Pipeline()
    #LSTM model:
    model = load_model("models/lstm/LSTM_model.h5")
    cap = cv2.VideoCapture(0)
    #mediapipe model
    with p.mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            #read feed
            ret, frame = cap.read()
            #make detection
            image, results = p.mediapipe_detection(frame, holistic)
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
                if result[np.argmax(result)] > threshold:
                    buffer.append(actions[np.argmax(result)])
                    if len(buffer) > 1:
                        if buffer[-1] != buffer[-2]:
                            buffer = buffer[-1:]
                if len(buffer) == frequency:
                    if len(sentence) > 0:
                        if buffer[-1] != sentence[-1]:
                            sentence.append(buffer[-1])
                            corpus.append(buffer[-1])
                    else:
                        sentence.append(buffer[-1])
                        corpus.append(buffer[-1])
                    buffer = []
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