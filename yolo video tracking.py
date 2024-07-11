import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    with open('coco_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers, target_classes=['person', 'clock']):
    height, width = img.shape[:2]
    scalefactor = 1.0 / 255.0
    size = (416, 416)
    mean = (0, 0, 0)
    swapRB = True
    crop = False
    test_img = cv.dnn.blobFromImage(img, scalefactor, size, mean, swapRB, crop)

    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] in target_classes:  # 신뢰도가 50% 이상인 경우만 취함
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔

# 비디오 파일 경로 설정
video_path = 'KakaoTalk_20240710_152518779.mp4'

# 비디오 파일 열기
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit('비디오 파일을 열 수 없습니다.')

# 첫 프레임 읽기
ret, frame = cap.read()
if not ret:
    sys.exit('첫 프레임을 읽을 수 없습니다.')

# 객체 검출
objects = yolo_detect(frame, model, out_layers, target_classes=['person', 'clock'])
if len(objects) == 0:
    sys.exit('첫 프레임에서 검출된 객체가 없습니다.')

# 첫 번째 객체 선택 (사람 또는 시계만 필터링하여 선택)
x, y, x2, y2, confidence, class_id = objects[0]
bbox = (x, y, x2 - x, y2 - y)

# 객체 트래커 초기화
tracker = cv.legacy.TrackerKCF_create()  # cv2 -> cv.legacy로 변경
tracker.init(frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 트래킹 업데이트
    ret, bbox = tracker.update(frame)
    if ret:
        # 트래킹된 객체에 사각형 그리기
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        # 클래스 이름과 신뢰도 표시
        label = f'{class_names[class_id]}: {confidence:.2f}'
        cv.putText(frame, label, (p1[0], p1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv.putText(frame, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 결과 프레임 보여주기
    cv.imshow('Tracking', frame)

    # ESC 키를 누르면 종료
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
