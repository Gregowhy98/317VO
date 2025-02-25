import math
from ultralytics import YOLO
import numpy as np
import cv2
import onnxruntime
import yaml
import time

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(32, 3))

class YOLOv8:
    def __init__(self, model_path, device="cuda", visualize=False):
        self.model_path = model_path
        self.device = device
        self.visualize = visualize
        self.model = YOLO(self.model_path)

    def predict(self, src):
        if src is None:
            raise ValueError("Can NOT predict None")
        src = self._preprocess(src)
        ret = self.model(src, device=self.device, visualize=self.visualize)
        ret = self._postprocess(ret)
        return ret

    def _preprocess(self, src):
        return src

    def _postprocess(self, ret):
        return ret

class YOLOv8ONNX:
    def __init__(self, model_cofig_path, device="cuda", debug=False):
        self.model_config_path = model_cofig_path
        self.config = self._read_config()
        self.onnx_model_path = self.config['model_path']
        self.providers = []
        self.debug_mode = debug
        if device == "cuda" or device == "CUDA":
            if self.debug_mode:
                print("[YOLOv8_ONNX] Using CUDA")
            self.providers = ["CUDAExecutionProvider"]
        if device == "cpu" or device == "CPU":
            if self.debug_mode:
                print("[YOLOv8_ONNX] Using CPU")
            self.providers = ["CPUExecutionProvider"]
        if device == "dml" or device == "directml" or device == "DirectML" or device == "DML":
            if self.debug_mode:
                print("[YOLOv8_ONNX] Using DirectML")
            self.providers = ["DmlExecutionProvider"]
        if device == "TensorRT" or device == "tensorrt" or device == "TRT" or device == "trt":
            if self.debug_mode:
                print("[YOLOv8_ONNX] Using TensorRT")
            self.providers = ["TensorrtExecutionProvider"]
        self.session = onnxruntime.InferenceSession(self.onnx_model_path, providers=self.providers)
        self.model_inputs = self.session.get_inputs()
        self.input_shape = self.model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        self.classes_map_str = self.session.get_modelmeta().custom_metadata_map["names"]
        self.classes_map = eval(self.classes_map_str)
        self.num_classes = len(self.classes_map)
        self.conf_threshold = 0.45
        self.iou_threshold = 0.5
        self.num_masks = len(self.classes_map) + 4
    def _read_config(self):
        '''
        config should look like this:
        model_path: "path/to/model.onnx"
        '''
        with open(self.model_config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config

    def debugger(self, title, event, *args, **kwargs):
        start_time = time.time()
        ret = event(*args, **kwargs)
        print(f"[YOLOv8_ONNX] {title} time: {time.time() - start_time}")
        return ret
    def _preprocessing(self, src):
        # src should be read by cv2.imread
        self.img_height, self.img_width, _ = src.shape
        proc_src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        proc_src = cv2.resize(proc_src, (self.input_width, self.input_height))
        # normalize image
        proc_src = np.array(proc_src) / 255.0
        proc_src = np.transpose(proc_src, (2, 0, 1))
        proc_src = proc_src[np.newaxis, :, :, :].astype(np.float32)
        return proc_src

    def _postprocessing(self, ret):
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(ret[0])
        self.mask_maps = self.process_mask_output(mask_pred, ret[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, self.classes_map, mask_alpha, mask_maps=self.mask_maps)

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]


    # def _postprocessing_legacy(self, src, ret):
    #     ret0 = np.squeeze(ret[0]).transpose()
    #     ret1 = np.squeeze(ret[1])
    #     boxes = ret0[:, 0:-32]
    #     masks = ret0[:, -32:]
    #     split_idx = ret0.shape[1] - 32
    #     ret1 = ret1.reshape(ret1.shape[0], -1)
    #     masks = np.matmul(masks, ret1)  # perhaps faster using numpy.matmul()
    #     detections = np.hstack([boxes, masks])
    #     objects = []
    #     image_height, image_width, _ = src.shape
    #     model_width, model_height = self.input_width, self.input_height
    #     for row in detections:
    #         prob = row[4:split_idx].max()
    #         if prob < 0.2:
    #             continue
    #         class_id = row[4:split_idx].argmax()
    #         label = self.classes_map[class_id]
    #         xc, yc, w, h = row[:4]
    #         # 把x1, y1, x2, y2的坐标恢复到原始图像坐标
    #         x1 = (xc - w / 2) / model_width * image_width
    #         y1 = (yc - h / 2) / model_height * image_height
    #         x2 = (xc + w / 2) / model_width * image_width
    #         y2 = (yc + h / 2) / model_height * image_height
    #         # 获取实例分割mask
    #         mask = get_mask(row[split_idx:split_idx+160*160], (x1, y1, x2, y2), image_width, image_height)
    #         # 从mask中提取轮廓
    #         polygon = get_polygon(mask, x1, y1)
    #         objects.append([x1, y1, x2, y2, label, prob, polygon, mask])
    #     objects.sort(key=lambda x: x[5], reverse=True)
    #     results = []
    #     while len(objects) > 0:
    #         results.append(objects[0])
    #         objects = [object for object in objects if iou(object, objects[0]) < 0.5]
    #     return results

    def run(self, src):
        if self.debug_mode:
            proc_src = self.debugger("Preprocessing", self._preprocessing, src)
            ret = self.debugger("Inference", self.session.run, None, {self.model_inputs[0].name: proc_src})
            ret = self.debugger("Postprocessing", self._postprocessing, ret)
        else:
            proc_src = self._preprocessing(src)
            ret = self.session.run(None, {self.model_inputs[0].name: proc_src})
            ret = self._postprocessing(ret)
        return ret

    # def draw_results(self, src, ret):
    #     img = src.copy()
    #     for object in ret:
    #         x1, y1, x2, y2, label, prob, polygon, mask = object
    #         cv2.putText(img, f"{label}: {prob:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #         cv2.polylines(img, [np.array(polygon, dtype=int)], True, (0, 255, 0), 2)
    #     return img

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes


#显示绑定框和标签    
def draw_detections(image, boxes, scores, class_ids, class_names, mask_alpha=0.3, mask_maps=None):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img

def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    x1 = max(x1, x3)
    y1 = max(y1, y3)
    x2 = min(x2, x4)
    y2 = min(y2, y4)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    return intersection / (area1 + area2 - intersection)

def get_polygon(mask, x1, y1):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    contour = contours[0]
    contour = contour.reshape(-1, 2)
    contour = contour + [x1, y1]
    return contour

def get_mask(row, box, img_width, img_height):
    mask = row.reshape(160, 160)
    x1, y1, x2, y2 = box
    # box坐标是相对于原始图像大小，需转换到相对于160*160的大小
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    if mask_x1 > mask_x2:
        mask_x1, mask_x2 = mask_x2, mask_x1
    if mask_y1 > mask_y2:
        mask_y1, mask_y2 = mask_y2, mask_y1
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    mask = sigmoid(mask)
    # 把mask的尺寸调整到相对于原始图像大小
    if mask.size:
        mask = cv2.resize(mask, (round(x2 - x1), round(y2 - y1)))
    else:
        mask = np.zeros((round(y2 - y1), round(x2 - x1)))
    mask = (mask > 0.5).astype("uint8") * 255
    return mask

def draw_polygon_filter_areas(img, polygons, classes_to_filter=None, color=None):
    # make a black image with the same size as the original image
    ret = np.zeros_like(img)
    if classes_to_filter is None:
        # make classes_to_filter to be all False
        classes_to_filter = np.zeros(polygons, dtype=bool)
    for idx, polygon in enumerate(polygons):
    # paint polygon area
        if classes_to_filter[idx]:
            cv2.fillConvexPoly(ret, np.array(polygon, dtype=int), color)
    #ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return ret

def get_filter_index(class_idx_array, filter_idxes):
    return np.array([i in filter_idxes for i in class_idx_array])

def draw_polygon_areas(img, polygons, classes_to_filter=None):
    # make a black image with the same size as the original image
    ret = np.zeros_like(img)
    if classes_to_filter is None:
        # make classes_to_filter to be all False
        classes_to_filter = np.zeros(len(polygons), dtype=bool)
    for idx, polygon in enumerate(polygons):
        # paint polygon area
        if classes_to_filter[idx]:
            cv2.fillConvexPoly(ret, [np.array(polygon, dtype=int)], (255, 255, 255))
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return ret


def draw_points(img, polygons, class_array = None):
    ret = img.copy()
    if class_array is None:
        class_array = np.zeros(len(polygons))
    for points in polygons:
        # find the top right point
        top_right = points[0]
        for point in points:
            if point[0] > top_right[0] and point[1] < top_right[1]:
                top_right = point
            cv2.circle(ret, (int(point[0]), int(point[1])), 2, (0, 255, 0), 1)
        # draw class text on top of the polygon
        cv2.putText(ret, str(class_array[0]), (int(top_right[0]), int(top_right[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return ret
    
def point_point_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            
def pick_line_matches(polygons, line_matches, threshold=2):
    # find the line which's endpoints are near the polygon
    all_vertices = []
    for polygon in polygons:
        vertices = np.array(polygon)
        all_vertices.append(vertices)
    line_matches_ret = []

    for line_match in line_matches:
        # find the nearest vertex
        for vertices in all_vertices:
            for vertex in vertices:
                if point_point_distance(vertex, line_match[0]) < threshold or point_point_distance(vertex, line_match[1]) < threshold:
                    line_matches_ret.append(line_match)
                    break
    
    line_matches_ret = np.array(line_matches_ret)
    return line_matches_ret
    
    
if __name__ == "__main__":
    model_path = "/home/wenhuanyao/YOLO/runs/segment/cityscapes_filtered_yolov8x_640x640_100eps_bs16/weights/best.onnx"
    yolo = YOLOv8(model_path, visualize=True)
    src = './testpic.png'
    src = cv2.imread(src, cv2.IMREAD_COLOR)

    ret = yolo.predict(src)
    
    ret = ret[0]
    ret.show()
    
    filter_idx_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 16: 'dog'}
    filter_idxes = filter_idx_dict.keys()
    polygons_xy = ret.masks.xy
    class_idx_array = np.array(ret.boxes.cls.cpu().numpy(), dtype=int)
    # convert idx to str by ret.names dict
    class_array = np.array([ret.names[i] for i in class_idx_array])

    # cv2.imshow("polygon filter areas", draw_polygon_filter_areas(src, polygons_xy, get_filter_index(class_idx_array, filter_idxes)))
    # cv2.waitKeyEx(0)
    # cv2.imshow("polygon points", draw_points(src, ret.masks.xy, class_array=class_array))
    # cv2.waitKey(0)
    # cv2.imshow("polygon areas", draw_polygon_areas(src, ret.masks.xy))
    # cv2.waitKeyEx(0)
    print(ret)
