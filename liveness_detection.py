import tensorflow as tf
import numpy as np
from PIL import Image
import time

class LivenessDetection:
    def __init__(self, model_path="models/liveness_detection_model.tflite"):
        
        # Load the TensorFlow Lite model for liveness detection
        self.interpreter = tf.lite.Interpreter(model_path=model_path)

        # Allocate tensors for the model (prepares the model to run inference)
        self.interpreter.allocate_tensors()

        # Get the input details of the model (e.g., input shape, data type)
        self.liveness_detection_interpreter_input_details = self.interpreter.get_input_details()

        # Get the output details of the model (e.g., output shape, data type)
        self.liveness_detection_interpreter_output_details = self.interpreter.get_output_details()

    def preprocess_frame(self, frame):
    
        # Resize the frame to 224x224 pixels to match the model input size
        frame = Image.fromarray(frame).resize((224, 224))

        # Normalize the frame by scaling pixel values to [0, 1]
        frame = np.array(frame, dtype=np.float32) / 255.0
        
        # Normalize using ImageNet mean and std values
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        standard_deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame = (frame - mean) / standard_deviation  # Normalize the frame

        # Convert the frame to CHW (Channel, Height, Width) format
        frame = np.transpose(frame, (2, 0, 1))

        # Add batch dimension and return the preprocessed frame
        return np.expand_dims(frame, axis=0).astype(np.float32)

    def detect_liveness(self, frame):
   
        # Start timing the liveness detection process
        liveness_detection_process_start_time = time.time()

        # Preprocess the frame for input to the model
        liveness_detection_input_data = self.preprocess_frame(frame)

        # Set the preprocessed input data as the model's input tensor
        self.interpreter.set_tensor(self.liveness_detection_interpreter_input_details[0]['index'], liveness_detection_input_data)

        # Run the model inference
        self.interpreter.invoke()

        # Retrieve the output of the model
        liveness_detection_output_data = self.interpreter.get_tensor(self.liveness_detection_interpreter_output_details[0]['index'])[0]

        # Check if the model output indicates binary classification (live or spoof)
        if liveness_detection_output_data.shape[-1] == 1: 
            
            # Apply sigmoid function to get the probability of being live
            liveness_detection_probability = 1 / (1 + np.exp(-liveness_detection_output_data.item()))

            # Classify the result based on the probability (below 0.5 is 'live', above is 'spoof')
            liveness_detection_prediction = "live" if liveness_detection_probability < 0.5 else "spoof"
        
        # Measure the total inference time for liveness detection
        liveness_detection_process_inference_time = time.time() - liveness_detection_process_start_time

        # Return the predicted liveness status, probability, and inference time
        return liveness_detection_prediction, liveness_detection_probability, liveness_detection_process_inference_time
