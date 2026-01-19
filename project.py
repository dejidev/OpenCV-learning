import os
import cv2
import numpy as np
import mediapipe as mp
import winsound
from datetime import datetime
import pandas as pd

# ===================== INITIALIZATION =====================
MAX_WIDTH, MAX_HEIGHT = 1280, 720

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(False, max_num_faces=5)

file_path = os.path.realpath(__file__)
src_dir = os.path.dirname(file_path)
output_dir = src_dir.replace("src", "Output")
yolo_dir = src_dir.replace("src", "YOLOv4-tiny")

# ===================== YOLO LOAD =====================
with open(os.path.join(yolo_dir, "coco.names")) as f:
    class_names = [line.strip() for line in f]

net = cv2.dnn.readNet(
    os.path.join(yolo_dir, "yolov4-tiny.weights"),
    os.path.join(yolo_dir, "yolov4-tiny.cfg")
)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ===================== CAMERA DETECTION =====================
cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cameras.append(cap)
    else:
        cap.release()

if not cameras:
    raise Exception("No cameras found")

os.makedirs(output_dir, exist_ok=True)

video_writers, anomaly_dirs, anomaly_logs = [], [], []

# ===================== SETUP OUTPUT =====================
for idx, cam in enumerate(cameras):
    ret, frame = cam.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
    w, h = int(w * scale), int(h * scale)

    cam_dir = os.path.join(output_dir, f"Camera_{idx}")
    os.makedirs(cam_dir, exist_ok=True)

    types = ["Left", "Right", "Talking", "Phone"]
    subdirs = {t: os.path.join(cam_dir, "Anomalies", t) for t in types}
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    anomaly_dirs.append(subdirs)
    anomaly_logs.append([])

    writer = cv2.VideoWriter(
        os.path.join(cam_dir, "processed_footage.mp4"),
        cv2.VideoWriter_fourcc(*"MP4V"),
        10, (w, h)
    )
    video_writers.append(writer)

# ===================== MAIN LOOP =====================
while all(cam.isOpened() for cam in cameras):
    for idx, cam in enumerate(cameras):
        ret, frame = cam.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        phone = {"x": [], "y": [], "w": [], "h": []}

        for out in outputs:
            for d in out:
                scores = d[5:]
                cid = np.argmax(scores)
                if class_names[cid] == "cell phone" and scores[cid] > 0.5:
                    cx, cy = int(d[0]*w), int(d[1]*h)
                    pw, ph = int(d[2]*w), int(d[3]*h)
                    phone["x"].append(cx - pw//2)
                    phone["y"].append(cy - ph//2)
                    phone["w"].append(pw)
                    phone["h"].append(ph)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_mesh.process(rgb)

        anomaly_detected, labels = False, []

        if faces.multi_face_landmarks:
            for face in faces.multi_face_landmarks:
                lms = face.landmark
                x1, x2 = int(min(l.x for l in lms)*w), int(max(l.x for l in lms)*w)
                y1, y2 = int(min(l.y for l in lms)*h), int(max(l.y for l in lms)*h)

                pose_res = pose.process(rgb[y1:y2, x1:x2])
                color, label = (0,255,0), "Forward"

                if pose_res.pose_landmarks:
                    lm = pose_res.pose_landmarks.landmark
                    nose = lm[mp_pose.PoseLandmark.NOSE]
                    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    orient = (nose.x - (ls.x+rs.x)/2) / nose.z
                    if orient > 0.03:
                        label = "Right"
                    elif orient < -0.03:
                        label = "Left"

                lips = (lms[15].y - lms[13].y)
                if lips > 0.017:
                    labels.append("Talking")

                if label != "Forward":
                    labels.append(label)

                if labels:
                    color = (0,0,255)
                    anomaly_detected = True

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,", ".join(set(labels)),(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        for x,y,w2,h2 in zip(phone["x"], phone["y"], phone["w"], phone["h"]):
            cv2.rectangle(frame,(x,y),(x+w2,y+h2),(0,0,255),2)
            labels.append("Phone")
            anomaly_detected = True

        video_writers[idx].write(frame)

        if anomaly_detected:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            winsound.Beep(1000,300)
            print(f"[ALERT] Camera {idx}: {set(labels)} @ {ts}")

        cv2.imshow(f"Camera {idx}", cv2.flip(frame,1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ===================== CLEANUP =====================
for idx, cam in enumerate(cameras):
    cam.release()
    if anomaly_logs[idx]:
        pd.DataFrame(anomaly_logs[idx]).to_excel(
            os.path.join(output_dir,f"Camera_{idx}/anomaly_log.xlsx"),
            index=False
        )

for w in video_writers:
    w.release()

cv2.destroyAllWindows()
