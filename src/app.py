import cv2 
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm  
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
import streamlit as st

HEIGHT, WIDTH = 512, 640
input_image_size = (HEIGHT, WIDTH)
min_score_thresh = 0.40
export_dir = 'models\conv2d\exported_model'

category_index_dict = {
  0:{
      "id": 0,
      "name": "beef"
    },
  1:{
      "id": 1,
      "name": "bowl"
    },
  2:{
      "id": 2,
      "name": "burrito"
    },
  3:{
      "id": 3,
      "name": "cheese"
    },
  4:{
      "id": 4,
      "name": "chicken"
    },
  5:{
      "id": 5,
      "name": "guacamole"
    },
  6:{
      "id": 6,
      "name": "hummus"
    },
  7:{
      "id": 7,
      "name": "impossible"
    },
  8:{
      "id": 8,
      "name": "kebab"
    },
  9:{
      "id": 9,
      "name": "quesadilla"
    },
  10:{
      "id": 10,
      "name": "salmon"
    },
  11:{
      "id": 11,
      "name": "tacos"
    },
  12:{
      "id": 12,
      "name": "veggie"
    }
}

#import saved model:
imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image

# Function to perform object detection on the frame
def perform_object_detection(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = build_inputs_for_object_detection(frame, input_image_size)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype = tf.uint8)
    image_np = image[0].numpy()
    result = model_fn(image)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=category_index_dict,
        use_normalized_coordinates=False,
        max_boxes_to_draw=1,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4
        )
    return image_np

# Main Streamlit code
def main():
    # Set the page title
    st.title("SignDine")

    # Create a placeholder for the captured frame
    frame_placeholder = st.empty()

    # Create buttons for capture reset and output reset
    reset_output_button = st.button("Reset Output")
    close_capture_button = st.button("Close Capture")

    # Initialize the video capture object
    video_capture = cv2.VideoCapture(0)

    # Start capturing frames from the webcam
    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()

        # Perform object detection on the frame
        processed_frame = perform_object_detection(frame)

        # Display the captured frame
        frame_placeholder.image(processed_frame)
        
        if close_capture_button:
          video_capture.release()
          cv2.destroyAllWindows()

        # Wait for a key press or exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
# Run the main function
if __name__ == '__main__':
    main()
