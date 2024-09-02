import cv2
import dlib
import numpy as np
import os

# load the face detector and the shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\\Downloads\\shape_predictor_68_face_landmarks.dat\\'
                                 'shape_predictor_68_face_landmarks.dat')  # places the 68 dots on the face inside
# the bounding box which then calculates the coordinates. 1-17 Jaw, 18-27 Eyebrows, 28-36 Nose, 37-48 Eyes, 49-68 Mouth


face_rec_model = dlib.face_recognition_model_v1('D:\\Downloads\\dlib_face_recognition_resnet_model_v1.dat\\'
                                                'dlib_face_recognition_resnet_model_v1.dat')  # encodes the embeddings into a 128 size vector

# directory to store facial embeddings
embeddings_dir = 'face_embeddings'
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# initialize the webcam
cap = cv2.VideoCapture(0)

user_name = input("Enter your name: ")  # register the user

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # RGB to gray scale

    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)
        face_embedding = np.array(face_rec_model.compute_face_descriptor(frame, face_landmarks))

        # Save the embedding as a .npy file
        np.save(os.path.join(embeddings_dir, f'{user_name}.npy'), face_embedding)

        # Drawing a rectangle around the face and display the user's name
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, user_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("User Registration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
