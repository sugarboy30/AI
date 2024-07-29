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

















# Deepsort code start




# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow.compat.v1 as tf

#tf.compat.v1.disable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



class Dummy:
    def __init__(self, video:str, output:str="./io_data/output/output.avi", coco_names_path:str ="./io_data/input/classes/coco.names", output_format:str='XVID', 
    iou:float=0.45, score:bool=0.5, dont_show:bool=False, count:bool=False):
        '''
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            iou: IOU threshold
            score: Matching score threshold
            dont_show: dont show video output
            count: count objects being tracked on screen
            coco_file_path: File wich contains the path to coco naames
        '''
        self.video = video
        self.output = output
        self.output_format = output_format
        self.count = count
        self.iou = iou
        self.dont_show = dont_show
        self.score = score
        self.coco_names_path = coco_names_path



def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


def read_class_names():
    '''
    Raad COCO classes names 
    '''
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
    
    return dict(zip(range(len(classes)), classes))


if __name__ == "__main__":
    main()
















