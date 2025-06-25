from ultralytics import YOLO
import cv2

# Model 1: Deteksi sajam dan senpi (custom trained)
model_weapon = YOLO('runs/detect/train/weights/best.pt')
weapon_classes = ['knife', 'pistol']

# Model 2: Deteksi person dari YOLOv8 COCO pretrained
model_person = YOLO('yolov8n.pt')
person_class_index = 0  # di COCO, person = class 0

# Ambil dari stream webcam / IP cam
stream_url = 'http://192.168.18.209:5050/video_feed'
# cap = cv2.VideoCapture(stream_url)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Gagal membuka stream dari {stream_url}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Prediksi dari kedua model
    result_weapon = model_weapon(frame)[0]
    result_person = model_person(frame, conf=0.5)[0]

    # Hitung deteksi
    count_senjata = [0, 0]  # sajam, senpi
    count_person = 0

    # ---- Tampilkan deteksi weapon (custom model) ----
    for box in result_weapon.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls_id < len(weapon_classes):
            label = weapon_classes[cls_id]
            count_senjata[cls_id] += 1
        else:
            label = f"ID:{cls_id}"

        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ---- Tampilkan deteksi person (COCO pretrained model) ----
    for box in result_person.boxes:
        cls_id = int(box.cls)
        if cls_id == person_class_index:
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            count_person += 1
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tampilkan hitungan total
    y_offset = 30
    cv2.putText(frame, f'Person: {count_person}', (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_offset += 30
    for i, count in enumerate(count_senjata):
        cv2.putText(frame, f'{weapon_classes[i]}: {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30

    cv2.imshow('Deteksi Senjata dan Person - YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
