import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image

# Inisialisasi Firebase menggunakan credential
cred = credentials.Certificate("credential.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://test-firebase-f28f7-default-rtdb.firebaseio.com/',
})

# Load the Haar cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('DataSet/training.xml')  # Load training data

# Dapatkan referensi database
ref = db.reference('captured_photo')
faces_ref = db.reference('detected_faces')
faces_id_ref = db.reference('detected_faces/id')
employees_ref = db.reference('employee')

# Ambil data employee dari Firebase
employees_data = employees_ref.get()

# Proses data employees untuk mendapatkan nama berdasarkan ID
employees = {str(k): v['nama'] for k, v in employees_data.items()}


# Variabel untuk melacak apakah ini perubahan pertama atau tidak
first_change = True

# Fungsi callback untuk memproses perubahan pada database
def callback(event):
    global first_change
    
    if first_change:
        # Mengabaikan perubahan pertama
        first_change = False
        return
    
    print("================================================================================")
    
    data = event.data
    path = event.path
    print(path)
    
    iterator = iter(data.items())

    # Mengambil node pertama
    key1, value1 = next(iterator)
    print("Node 1:", key1)
    print("Isi 1:", value1)

    # Mengambil node kedua
    key2, value2 = next(iterator)
    print("Node 2:", key2)
    print("Isi 2:", value2)
    
    # Google Drive image link
    google_drive_link = "https://drive.google.com/uc?id=" + value1
    
    # Function to fetch image from Google Drive link
    def fetch_image_from_google_drive(link):
        print(google_drive_link)
        response = requests.get(link)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Request image from Google Drive
    frame = fetch_image_from_google_drive(google_drive_link)

    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around the faces and recognize
    for (x, y, w, h) in faces:
        # Recognize the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 80:
            employee_name = employees[str(id)]
            print(employee_name)
            print(confidence)
            data = {
                'id': employee_name,
                'confidence': confidence,
                'photo_id': value1,
                'photo_name': value2
            }
            faces_ref.push(data)
            faces_id_ref.set(employee_name)
        else:
            print("UNKNOWN")
            print(confidence)

# Memasang callback untuk mendeteksi perubahan
ref.listen(callback)

# Untuk menjalankan program agar tetap berjalan dan mendeteksi perubahan secara real-time
while True:
    pass


# In[ ]:




