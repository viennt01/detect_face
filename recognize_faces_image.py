import face_recognition
import argparse
import pickle
import cv2

# Function to detect faces using OpenCV DNN
def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

# Argument parser to pass inputs like encodings, image, etc.
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-s", "--encodings_gender", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to the test image")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Load face recognition data (encodings and names)
print("[INFO] loading encodings...")
with open(args["encodings"], "rb") as f:
    data = pickle.load(f)
    
with open(args["encodings_gender"], "rb") as f:
    data_gender = pickle.load(f)


# Load the image and convert from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using face_recognition library
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# Initialize list to store names
names = []
genders = []

# Gender and age detection model setup
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.42633377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(37-43)', '(48-53)', '(60-100)']
genderList = ['Man', 'Woman']

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Process each face detected
for encoding, box in zip(encodings, boxes):
    matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
    name = "Unknown"
    
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    names.append(name)
    
for encoding, box in zip(encodings, boxes):
    matches = face_recognition.compare_faces(data_gender["encodings"], encoding, 0.4)  # Sửa ở đây
    gender = "Unknown"
    
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        genderCounts = {}
        for i in matchedIdxs:
            gender = data_gender["genders"][i]
            genderCounts[gender] = genderCounts.get(gender, 0) + 1
        gender = max(genderCounts, key=genderCounts.get)
    genders.append(gender)

# Age and Gender detection
padding = 20
for ((top, right, bottom, left), name, gender) in zip(boxes, names, genders):
    face = image[max(0, top - padding):min(bottom + padding, image.shape[0] - 1), max(0, left - padding):min(right + padding, image.shape[1] - 1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()] if gender == "Unknown" else gender

    # Predict Age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    # Draw bounding box, name, age, and gender
    # label = "{} - {}, Age: {}".format(name, gender, age)
    label = "{}".format(gender)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Display the final image
cv2.imshow("Recognized Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
