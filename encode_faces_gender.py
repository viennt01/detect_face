# USAGE
# python encode_faces.py --dataset dataset --encodings encodings_gender.pickle

from imutils import paths
import argparse
import pickle
import cv2
import os
import face_recognition

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detector to use: cnn or hog")
args = vars(ap.parse_args())

# Lấy paths của images trong dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Khởi tạo danh sách known encodings và known names
knownEncodings = []
knownGenders = []  # Thêm danh sách để lưu thông tin giới tính

# Duyệt qua các image paths
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    gender = imagePath.split(os.path.sep)[-2]  # Giới tính sẽ nằm ở thư mục cha

    # Load image và chuyển từ BGR sang RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect khuôn mặt
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Lưu encoding, gender
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownGenders.append(gender)  # Lưu giới tính cùng với encodings

# Lưu encodings, names và genders vào file pickle
print("[INFO] serializing encodings and genders...")
data = {"encodings": knownEncodings, "genders": knownGenders}

with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))
