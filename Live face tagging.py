import cv2
import dlib
import numpy as np
import os

# Load the face detector, shape predictor, and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'D:\\Downloads\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1(
    'D:\\Downloads\\dlib_face_recognition_resnet_model_v1.dat\\dlib_face_recognition_resnet_model_v1.dat')

# Load the stored facial embeddings
embeddings_dir = 'face_embeddings'
stored_embeddings = {}
for file_name in os.listdir(embeddings_dir):
    user_name = file_name.split('.')[0]
    stored_embeddings[user_name] = np.load(os.path.join(embeddings_dir, file_name))


# Function to compare two face embeddings
def compare_embeddings(embedding1, embedding2, threshold=0.6):
    return np.linalg.norm(embedding1 - embedding2) < threshold


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)
        face_embedding = np.array(face_rec_model.compute_face_descriptor(frame, face_landmarks))

        # comparing with stored embeddings
        name = "Unknown"
        for user_name, stored_embedding in stored_embeddings.items():
            if compare_embeddings(face_embedding, stored_embedding):
                name = user_name
                break

        # Drawing a rectangle around the face and display the recognized user's name
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

