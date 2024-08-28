import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize MTCNN detector
detector = MTCNN()
# Initialize FaceNet
facenet = FaceNet()

def detect_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect faces
    print(1)
    faces = detector.detect_faces(image_rgb)
    print(2)
    results = []
    for face in faces:
        bounding_box = face['box']
        keypoints = face['keypoints']

        x, y, w, h = bounding_box
        face_image = image_rgb[y:y+h, x:x+w]

        # Preprocess the face image for FaceNet
        face_image = cv2.resize(face_image, (160, 160))
        face_image = np.expand_dims(face_image, axis=0)
        face_image = (face_image - 127.5) / 128.0

        # Get face embedding
        embedding = facenet.embeddings(face_image)

        results.append({
            'bounding_box': bounding_box,
            'keypoints': keypoints,
            'embedding': embedding[0]
        })

    return image, results

def draw_faces(image, results):
    for face in results:
        bounding_box = face['bounding_box']
        keypoints = face['keypoints']

        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for point in keypoints.values():
            cv2.circle(image, point, 2, (0, 0, 255), 2)

    return image

# Example usage
image_path = 'cutest_baby.jpg'
image, results = detect_faces(image_path)

if results:
    image_with_faces = draw_faces(image, results)
    cv2.imwrite("cutest_baby_with_bbox.jpg", image_with_faces)

    for i, face in enumerate(results):
        print(f"Face {i+1}:")
        print(f"  Bounding Box: {face['bounding_box']}")
        print(f"  Keypoints: {face['keypoints']}")
        print(f"  Embedding shape: {face['embedding'].shape}")
else:
    print("No faces detected in the image.")