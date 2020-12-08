import numpy as np, cv2, pickle
from camera_parameters import *
with open('id_lbl_map.pkl', 'rb') as (f):
    id_lbl_map = pickle.load(f)

def extract_box(box, im):
    box_w = (box[2] - box[0]) * SCALE_RATIO
    box_h = (box[3] - box[1]) * SCALE_RATIO
    box_top = max(0, box[1] * SCALE_RATIO - box_h // 4)
    box_bottom = min(CAP_HEIGHT, box[3] * SCALE_RATIO + 3 * box_h // 4)
    box_l = max(0, box[0] * SCALE_RATIO - box_w // 2)
    box_r = min(CAP_WIDTH, box[2] * SCALE_RATIO + box_w // 2)
    return im[int(box_top):int(box_bottom), int(box_l):int(box_r)]


def crop_img(img):
    w = img.shape[1]
    h = img.shape[0]
    return img[int(0.1 * h):int(0.65 * h), int(0.25 * w):int(0.95 * w)]


def handle_prediction(outputs):
    """
    extracts boxes, classes and scores from model output
    """
    out_cpu = outputs['instances'].to('cpu')
    boxes = out_cpu.pred_boxes.tensor.numpy()
    classes = out_cpu.pred_classes.numpy()
    classes = np.array([id_lbl_map[str(clss)] for clss in classes])
    scores = out_cpu.scores.numpy()
    return (boxes, classes, scores)


def nms(boxes, classes, scores, overlapThresh):
    if len(boxes) == 0:
        return []
    else:
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = w * h / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
             np.where(overlap > overlapThresh)[0])))

        return (
         boxes[pick].astype('int'), classes[pick], scores[pick])
