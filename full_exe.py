import cv2
import numpy as np
import os
import requests
import time
import random
import shutil

dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"
cascade_path = "haarcascade_frontalface_default.xml"

# Take camera IP from user
camera_url = "http://" + input("Enter the IP for the camera: ") + ":8080/shot.jpg"

# Ensure dataset and trainer folders exist
os.makedirs(dataset_path, exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Load the Haar Cascade
if not os.path.exists(cascade_path):
    raise FileNotFoundError("[ERROR] Haarcascade file not found!")
faceCascade = cv2.CascadeClassifier(cascade_path)

# Load the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def augment_image(image):
    augmented_images = []
    angles = [-15, 15]
    for angle in angles:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)
    augmented_images.append(cv2.flip(image, 1))
    augmented_images.append(np.clip(image * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8))
    augmented_images.append(cv2.GaussianBlur(image, (5, 5), 0))
    return augmented_images

def train_model():
    print("\n[INFO] Training the face recognition model...")

    # Validate dataset path
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path '{dataset_path}' does not exist.")
        return

    users = [user for user in sorted(os.listdir(dataset_path)) 
             if os.path.isdir(os.path.join(dataset_path, user))]
    if not users:
        print("[ERROR] No user directories found in the dataset.")
        return

    user_ids = {user: idx for idx, user in enumerate(users)}
    faceSamples, ids = [], []

    for user in users:
        person_path = os.path.join(dataset_path, user)
        images_found = False

        for image_file in os.listdir(person_path):
            imagePath = os.path.join(person_path, image_file)
            
            if not os.path.isfile(imagePath):
                continue  # Skip if not a file
            
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERROR] Couldn't read image: {imagePath}")
                continue

            faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                print(f"[WARNING] No face detected in image: {imagePath}")
                continue

            images_found = True
            for (x, y, w, h) in faces:
                face = cv2.resize(img[y:y + h, x:x + w], (200, 200))
                faceSamples.append(face)
                ids.append(user_ids[user])

        if not images_found:
            print(f"[WARNING] No valid images found for user: {user}")

    if not faceSamples:
        print("[ERROR] No faces found in the dataset for training.")
        return

    try:
        recognizer.train(faceSamples, np.array(ids))
        recognizer.write(trainer_path)
        print(f"[INFO] Model trained with {len(set(ids))} unique users.")
    except cv2.error as e:
        print(f"[ERROR] OpenCV error during training: {e}")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")

def list_users():
    users = os.listdir(dataset_path)
    print("\n[INFO] Registered Users:")
    for user in users:
        print(f"- {user}")

def delete_user():
    list_users()
    name = input("Enter the name of the user to delete: ").strip()
    person_path = os.path.join(dataset_path, name)
    if os.path.exists(person_path):
        shutil.rmtree(person_path)
        print(f"[INFO] User '{name}' deleted successfully.")
    else:
        print(f"[ERROR] User '{name}' not found.")

def recognize_users():
    if not os.path.exists(trainer_path):
        print("[ERROR] No trained model found. Please train the model first.")
        return

    recognizer.read(trainer_path)
    users = sorted(os.listdir(dataset_path))
    user_ids = {idx: user for idx, user in enumerate(users)}

    while True:
        try:
            response = requests.get(camera_url, timeout=5)
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                    id_, confidence = recognizer.predict(face)
                    name = user_ids.get(id_, "Unknown")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Recognizing Faces", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to {camera_url}: {e}")
            break
    cv2.destroyAllWindows()

def capture_from_camera(url, person_path, allow_add_user):
    print(f"[INFO] Connecting to camera: {url}")
    count = 0
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_array = np.array(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if allow_add_user:
                    count += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    save_path = os.path.join(person_path, f"{count}.jpg")
                    cv2.imwrite(save_path, gray)
                    for i, aug_img in enumerate(augment_image(gray)):
                        aug_path = os.path.join(person_path, f"{count}_aug{i}.jpg")
                        cv2.imwrite(aug_path, aug_img)
                cv2.imshow(f"Capturing Faces from {url}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to {url}: {e}")
            break
    print(f"[INFO] Image capture complete from {url}.")
    cv2.destroyAllWindows()

def add_user():
    name = input("Enter the person's name: ").strip()
    person_path = os.path.join(dataset_path, name)
    os.makedirs(person_path, exist_ok=True)
    print("[INFO] Capturing images for 30 seconds from the camera...")
    capture_from_camera(camera_url, person_path, allow_add_user=True)
    print(f"[INFO] Image capture complete. Images saved for {name}.")

def main():
    while True:
        print("\n=== FACE RECOGNITION SYSTEM ===")
        print("1. Add a new user")
        print("2. Delete a user")
        print("3. List users")
        print("4. Recognize users")
        print("5. Train model")
        print("6. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            add_user()
        elif choice == "2":
            delete_user()
        elif choice == "3":
            list_users()
        elif choice == "4":
            recognize_users()
        elif choice == "5":
            train_model()
        elif choice == "6":
            print("[INFO] Exiting the program.")
            break
        else:
            print("[ERROR] Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
