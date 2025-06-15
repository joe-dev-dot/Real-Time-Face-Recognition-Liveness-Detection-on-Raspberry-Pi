import cv2
import time
from multiprocessing import Process, Queue
from face_recognition import FaceRecognizer
from liveness_detection import LivenessDetection
from picamera2 import Picamera2

# Function to process face recognition in a separate process
def face_recognition_process(face_recognition_frame_queue, face_recognition_result_queue):
    # Initialize the face recognition model
    face_recognizer = FaceRecognizer()  
    
    while True:
        # Check if the frame queue is not empty
        if not face_recognition_frame_queue.empty():
            # Get the frame from the queue (non-blocking)
            face_recognition_frame = face_recognition_frame_queue.get_nowait()  
            
            # Perform face recognition on the frame
            recognized_faces, face_recognition_process_inference_time = face_recognizer.recognize_faces(face_recognition_frame)
            
            # Send the recognized faces and the time taken for processing back to the result queue
            face_recognition_result_queue.put((recognized_faces, face_recognition_process_inference_time))
        else:
            time.sleep(0.01)
            
# Function to process liveness detection in a separate process
def liveness_detection_process(liveness_detection_frame_queue, liveness_detection_result_queue):
    # Initialize the liveness detection model
    liveness_detection = LivenessDetection()  
    
    while True:
        # Check if the frame queue is not empty
        if not liveness_detection_frame_queue.empty():
            # Get the frame from the queue (non-blocking)
            liveness_detection_frame = liveness_detection_frame_queue.get_nowait()  
            
            # Perform liveness detection on the frame
            liveness_detection_prediction, liveness_detection_probability, liveness_detection_process_inference_time = liveness_detection.detect_liveness(liveness_detection_frame)
            
            # Send the liveness detection result (prediction, probability, and processing time) back to the result queue
            liveness_detection_result_queue.put((liveness_detection_prediction, liveness_detection_probability, liveness_detection_process_inference_time))
        else:
            time.sleep(0.01)

# Main block where the processes are managed and video capture takes place
if __name__ == '__main__':

    # Create queues for face recognition and liveness detection
    face_recognition_frame_queue = Queue(maxsize=1)
    face_recognition_result_queue = Queue(maxsize=1)

    liveness_detection_frame_queue = Queue(maxsize=1)
    liveness_detection_result_queue = Queue(maxsize=1)

    # Create and start the process for face recognition
    face_recognition_process_instance = Process(target=face_recognition_process, args=(face_recognition_frame_queue, face_recognition_result_queue))
    # Create and start the process for liveness detection
    liveness_detection_process_instance = Process(target=liveness_detection_process, args=(liveness_detection_frame_queue, liveness_detection_result_queue))

    # Start the processes
    face_recognition_process_instance.start()
    liveness_detection_process_instance.start()

    # Initialize the camera (PiCamera in this case)
    picam2 = Picamera2()  # Initialize PiCamera object
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})  # Camera configuration (RGB888 format, 640x480 resolution)
    picam2.configure(config)  # Configure the camera with the given settings
    picam2.start()  # Start capturing from the camera
    time.sleep(1)  # Give it a second to initialize the camera

    # Variable to track the last time the frames were processed
    last_face_recognition_and_liveness_detection_processed_time = time.time()

    # Initialize result holders for proper display
    recognized_faces = []
    liveness_detection_prediction = "Unknown"
    liveness_detection_probability = 0.0
    face_recognition_process_inference_time = 0.0
    liveness_detection_process_inference_time = 0.0

    try:
        while True:
            # Capture the current frame from the camera
            camera_frame = picam2.capture_array()  
            current_time = time.time()  # Get the current timestamp

            # Send frames at regular intervals (every 0.2 seconds)
            if current_time - last_face_recognition_and_liveness_detection_processed_time >= 0.2:
                # Make sure both queues are empty before putting new frames
                if face_recognition_frame_queue.empty() and liveness_detection_frame_queue.empty():
                    try:
                        # Send the frame to both face recognition and liveness detection queues
                        face_recognition_frame_queue.put_nowait(camera_frame.copy())  # Put the frame into the face recognition queue
                        liveness_detection_frame_queue.put_nowait(camera_frame.copy())  # Put the frame into the liveness detection queue
                    except:
                        pass  # Ignore any exceptions (queues might be full)
                last_face_recognition_and_liveness_detection_processed_time = current_time  # Update the last processed time

            # Retrieve the result from face recognition process if available
            if not face_recognition_result_queue.empty():
                recognized_faces, face_recognition_process_inference_time = face_recognition_result_queue.get()

            # Retrieve the result from liveness detection process if available
            if not liveness_detection_result_queue.empty():
                liveness_detection_prediction, liveness_detection_probability, liveness_detection_process_inference_time = liveness_detection_result_queue.get()

            # Display the results: draw bounding boxes for faces and show liveness status
            for (x1, y1, x2, y2, best_face_match, face_recognition_best_score) in recognized_faces:
                cv2.rectangle(camera_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle around detected face
                cv2.putText(camera_frame, f'{best_face_match} ({face_recognition_best_score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Display face recognition results

            # Show the liveness status on the frame (Green if live, Red if not)
            liveness_detection_result_color = (0, 255, 0) if liveness_detection_probability < 0.8 else (0, 0, 255)  # Green if live, Red if not
            cv2.putText(camera_frame, f'Liveness: {liveness_detection_prediction} ({liveness_detection_probability:.2f})', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_detection_result_color, 2)  # Display liveness result

            # Show the frame with the results
            cv2.imshow("Face Recognition + Liveness detection", camera_frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        face_recognition_process_instance.terminate()  # Terminate the face recognition process
        liveness_detection_process_instance.terminate()  # Terminate the liveness detection process
        picam2.stop()  # Stop the PiCamera
        cv2.destroyAllWindows()  # Close any OpenCV windows
