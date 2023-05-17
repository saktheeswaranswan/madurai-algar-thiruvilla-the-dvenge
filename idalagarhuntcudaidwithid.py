import cv2
import numpy as np
import time
import csv
import os

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

output_folder = "cropped_images"
os.makedirs(output_folder, exist_ok=True)

object_ids = {}
crop_counter = {}

cap = cv2.VideoCapture(0)

csv_file = open("object_detections.csv", "w")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Frame Index", "Object ID", "Class"])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # CUDA inference
    with cv2.cuda_GpuMat() as gpu_blob:
        gpu_blob.upload(blob)
        outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                width = frame.shape[1]
                height = frame.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        class_label = classes[class_id]
        confidence = confidences[i]

        if class_label in ["cat", "person"]:
            if class_label not in object_ids:
                object_ids[class_label] = 0
                crop_counter[class_label] = 0
            object_id = f"{class_label}_{object_ids[class_label]}"
            object_ids[class_label] += 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
            csv_writer.writerow([timestamp, frame_index, object_id, class_label])

            if crop_counter[class_label] < 100:
                output_path = os.path.join(output_folder, f"{object_id}_{timestamp.replace(':', '-')}")
                os.makedirs(output_path, exist_ok=True)
                crop_img = frame[y:y+h, x:x+w]
                image_filename = f"{object_id}_{crop_counter[class_label]}.jpg"
                cv2.imwrite(os.path.join(output_path, image_filename), crop_img)
                crop_counter[class_label] += 1
                cv2.putText(frame, f"Crops: {crop_counter[class_label]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"{class_label} {object_id} {confidence:.2f}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

