import os
import pickle
import cv2
from face_recognition import FaceRecognizer  # Ensure your class allows direct access to face embeddings

def generate_embeddings(dataset_path="dataset", output_path="known_faces_encodings.pkl"):
    face_recognizer = FaceRecognizer()
    embeddings_dict = {}

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            # Get the face embeddings only (should not compute similarity here)
            faces = face_recognizer.detect_faces(image)
            if not faces:
                print(f"No face detected in: {image_path}")
                continue

            for (x1, y1, x2, y2) in faces:
                face_img = image[y1:y2, x1:x2]
                embedding = face_recognizer.get_embedding(face_img)
                if embedding is not None:
                    person_embeddings.append(embedding.squeeze().tolist())

        if person_embeddings:
            embeddings_dict[person_name] = person_embeddings
        else:
            print(f"No valid embeddings for {person_name}")

    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    generate_embeddings()
