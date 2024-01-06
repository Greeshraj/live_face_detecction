import cv2
import face_recognition
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('face_database.db')
cursor = conn.cursor()

# Create a table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        encoding TEXT NOT NULL
    )
''')
conn.commit()

# Load a sample image for face recognition
known_image = face_recognition.load_image_file("path_to_known_image.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Access the Webcam
video_capture = cv2.VideoCapture(0)

# Identifying Faces in the Video Stream
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face_image = vid[y:y + h, x:x + w]

        # Encode the face using face_recognition library
        face_encoding = face_recognition.face_encodings(face_image)

        # Compare the face encoding with the known face encoding
        if len(face_encoding) > 0 and face_recognition.compare_faces([known_encoding], face_encoding[0])[0]:
            # If a match is found, draw a rectangle and label the person
            cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Retrieve and display the name associated with the recognized face
            cursor.execute('SELECT name FROM faces WHERE encoding = ?', (str(face_encoding[0]),))
            result = cursor.fetchone()
            if result:
                name = result[0]
                cv2.putText(vid, f'Name: {name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return faces

# Creating a Loop for Real-Time Face Detection
while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

    faces = detect_bounding_box(video_frame)

    cv2.imshow("My Face Detection Project", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the database connection
video_capture.release()
cv2.destroyAllWindows()
conn.close()
