import tensorflow as tf
import numpy as np
import pickle
from numpy.linalg import norm
import mediapipe as mp
import cv2
import time

class FaceRecognizer:
    def __init__(self, model_path="models/face_recognition_model.tflite", encodings_path="known_faces_encodings.pkl"):
       
        # Load the known faces encodings from a pickle file
        with open(encodings_path, "rb") as pickle_file:
            self.known_face_embeddings = pickle.load(pickle_file)
        print(f"Loaded {len(self.known_face_embeddings)} known face embeddings")

        # Load the TensorFlow Lite model for face recognition
        self.interpreter = tf.lite.Interpreter(model_path=model_path)

        # Allocate tensors for the model (this prepares the model to run inference)
        self.interpreter.allocate_tensors()

        # Get the input details of the model (e.g., shape, type)
        self.interpreter_face_recognition_input_details = self.interpreter.get_input_details()

        # Get the output details of the model (e.g., shape, type)
        self.interpreter_face_recognition_output_details = self.interpreter.get_output_details()

        # Initialize Mediapipe face detector
        mp_face_detection = mp.solutions.face_detection
        self.face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

    def cosine_similarity( self, frame_face_embeddings, known_face_embeddings):

        # measures how similar the two vectors are in terms of direction (ranges from -1 to 1).
        return np.dot(frame_face_embeddings, known_face_embeddings) / (norm(frame_face_embeddings) * norm(known_face_embeddings))

    def get_embedding(self, frame):
        
        # Resize the input frame to 160x160 pixels using linear interpolation
        frame = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)

        # Normalize the frame: scale pixel values to the range [-1, 1] by subtracting 127.5 and dividing by 128.0
        frame = (frame - 127.5) / 128.0

        # Convert the frame to float32 and add an extra dimension (batch size of 1) to match model input requirements
        face_recognition_input_data = np.expand_dims(frame.astype(np.float32), axis=0)

        # Set the preprocessed frame tensor as the model's input tensor
        self.interpreter.set_tensor(self.interpreter_face_recognition_input_details[0]['index'], face_recognition_input_data)

        # Run the model inference
        self.interpreter.invoke()

        # Retrieve the model's output tensor (first element in the output tensor) and return it
        return self.interpreter.get_tensor(self.interpreter_face_recognition_output_details[0]['index'])[0]

    def recognize_faces(self, frame):
        # Extract the height and width of the input frame (frame).
        height, width , _ = frame.shape

        # Process the input frame to detect faces using the face_detector.
        detection_results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        recognized_faces = []

        # Start timing the recognition process.
        face_recognition_process_start_time = time.time()

        # If any faces are detected in the frame, process each face.
        if detection_results.detections:
            for face_detection in detection_results.detections:
                # Extract the detection score and continue if the score is below threshold.
                detection_score = face_detection.score[0]
                if detection_score < 0.7:  # Ignore faces with low detection score.
                    continue

                # Get the relative bounding box coordinates for the detected face.
                bounding_box = face_detection.location_data.relative_bounding_box
                x1 = int(bounding_box.xmin * width)  # Left coordinate of bounding box.
                y1 = int(bounding_box.ymin * height)  # Top coordinate of bounding box.
                x2 = x1 + int(bounding_box.width * width)  # Right coordinate of bounding box.
                y2 = y1 + int(bounding_box.height * height)  # Bottom coordinate of bounding box.

                # Crop the detected face from the frame using the bounding box.
                face_crop = frame[y1:y2, x1:x2]
                
                # If the cropped face is empty, skip this face.
                if face_crop.size == 0:
                    continue

                # Generate the face embedding for the cropped face.
                frame_face_embeddings = self.get_embedding(face_crop)

                # Initialize the best match as "Unknown" with a score of 0.
                best_face_match = "Unknown"
                face_recognition_best_score = 0

                # Compare the face embedding with known embeddings.
                for known_face_name, known_face_embeddings in self.known_face_embeddings.items():
                    # Compute cosine similarity between the embeddings.
                    similarity = self.cosine_similarity(frame_face_embeddings, known_face_embeddings)

                    # Update the best match if the similarity score is better than the previous one.
                    if similarity > face_recognition_best_score:
                        face_recognition_best_score = similarity
                        # If similarity is above a threshold, set the best match.
                        best_face_match = known_face_name if similarity > 0.4 else "Unknown"

                # Append the results: bounding box, match, and score to the faces list.
                recognized_faces.append((x1, y1, x2, y2, best_face_match, face_recognition_best_score))

        # Measure the inference time for the face recognition process.
        face_recognition_process_inference_time = time.time() - face_recognition_process_start_time

        # Return the list of recognized faces and the elapsed time for the recognition process.
        return recognized_faces, face_recognition_process_inference_time
