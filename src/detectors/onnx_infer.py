# src/detectors/onnx_infer.py
import onnxruntime as ort
import numpy as np
import cv2, time
from typing import List, Dict, Tuple

class ONNXDetector:
    def __init__(self, onnx_path: str, input_size:int=640, conf_thres:float=0.25, iou_thres:float=0.45, providers=None):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        if providers:
            self.sess = ort.InferenceSession(onnx_path, so, providers=providers)
        else:
            self.sess = ort.InferenceSession(onnx_path, so)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def letterbox(self, img):
        h0, w0 = img.shape[:2]
        target = self.input_size
        r = min(target / w0, target / h0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = target - new_unpad[0], target - new_unpad[1]
        dw /= 2; dh /= 2
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        img_rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        img_trans = np.transpose(img_rgb, (2,0,1))[None, ...].astype(np.float32)
        return img_trans, r, (left, top)

    @staticmethod
    def _iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
        x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        a = (box[2]-box[0]) * (box[3]-box[1])
        b = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        return inter / (a + b - inter + 1e-9)

    def nms(self, dets, iou_thres=0.45):
        if not dets: return []
        boxes = np.array([d['box'] for d in dets]).astype(np.float32)
        scores = np.array([d['conf'] for d in dets])
        classes = np.array([d['class'] for d in dets])
        keep = []
        for c in np.unique(classes):
            idxs = np.where(classes==c)[0]
            order = scores[idxs].argsort()[::-1]
            while order.size>0:
                i = order[0]
                keep.append(idxs[i])
                if order.size==1: break
                ious = self._iou(boxes[idxs[i]], boxes[idxs[order[1:]]])
                inds = np.where(ious <= iou_thres)[0]
                order = order[inds+1]
        return [dets[i] for i in keep]

    def postprocess(self, pred, orig_shape, r, pad):
        H, W = orig_shape[:2]
        dets = []
        out = pred
        if out.ndim==3 and out.shape[0]==1:
            out = out[0]
        if out.ndim==2 and out.shape[1] >= 5:
            xywh = out[:, :4]; conf = out[:,4]
            if out.shape[1] > 6:
                class_probs = out[:,5:]; class_ids = np.argmax(class_probs, axis=1)
                class_scores = class_probs[np.arange(len(class_ids)), class_ids]
                scores = conf * class_scores
            else:
                class_ids = out[:,5].astype(np.int32) if out.shape[1] > 5 else np.zeros(len(conf), dtype=np.int32)
                scores = conf
            keep = scores > self.conf_thres
            xywh = xywh[keep]; scores = scores[keep]; class_ids = class_ids[keep]
            for i in range(len(scores)):
                x_c, y_c, w, h = xywh[i]
                x_c = (x_c - pad[0]) / r; y_c = (y_c - pad[1]) / r
                w = w / r; h = h / r
                x1 = max(0, int(x_c - w/2)); y1 = max(0, int(y_c - h/2))
                x2 = min(W-1, int(x_c + w/2)); y2 = min(H-1, int(y_c + h/2))
                dets.append({'box':(x1,y1,x2,y2), 'conf':float(scores[i]), 'class':int(class_ids[i])})
        dets = self.nms(dets, self.iou_thres)
        return dets

    def infer(self, frame):
        img, r, pad = self.letterbox(frame)
        ort_inputs = {self.input_name: img}
        t0 = time.time()
        outputs = self.sess.run(self.output_names, ort_inputs)
        t1 = time.time()
        preds = outputs[0]
        dets = self.postprocess(preds, frame.shape, r, pad)
        return dets, (t1-t0)
