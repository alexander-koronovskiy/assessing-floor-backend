import cv2
import numpy as np
import torch
from torch import nn
from copy import deepcopy
from numba import njit
from torchvision.models.detection import mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .exceptions import NetworkError
from .transforms import (
    Pad,
    Resize,
    Compose,
    ToTensor,
)


@njit
def get_angle(pts: np.ndarray):
    a = np.array([pts[0][0][0], pts[0][0][1]], dtype=pts.dtype)
    b = np.array([pts[1][0][0], pts[1][0][1]], dtype=pts.dtype)
    c = np.array([pts[2][0][0], pts[2][0][1]], dtype=pts.dtype)
    # TypingError: np.linalg.norm() only supported on float and complex arrays.
    ba = (a - b).astype(np.float32)
    bc = (c - b).astype(np.float32)
    unit_vector_ba = ba / np.linalg.norm(ba)
    unit_vector_bc = bc / np.linalg.norm(bc)
    dot_product = np.dot(unit_vector_ba, unit_vector_bc)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    try:
        return int(angle_deg)
    except Exception:
        raise ValueError('NaN value detected')


@njit
def move_points(contour: np.ndarray, pts: np.ndarray, angle: int, ext: list, weight: float = 1):
    ext_left, ext_right, ext_bot, ext_top = ext
    a = np.array([pts[0][0][0], pts[0][0][1]])
    b = np.array([pts[1][0][0], pts[1][0][1]])
    c = np.array([pts[2][0][0], pts[2][0][1]])

    right_angle = False
    if 45 < angle < 135:
        right_angle = True
        diff_x_ba = abs(b[0] - a[0])
        diff_y_ba = abs(b[1] - a[1])
        diff_x_bc = abs(b[0] - c[0])
        diff_y_bc = abs(b[1] - c[1])
        rap_ba = diff_x_ba / max(diff_y_ba, 1)
        rap_bc = diff_x_bc / max(diff_y_bc, 1)

        if rap_ba < rap_bc:
            a[0] = int((a[0] * weight + b[0]) / (2 + weight - 1))
            b[0] = a[0]
            c[1] = int((c[1] + b[1]) / 2)
            b[1] = c[1]
        else:
            c[0] = int((c[0] + b[0]) / 2)
            b[0] = c[0]
            a[1] = int((a[1] * weight + b[1]) / (2 + weight - 1))
            b[1] = a[1]
    else:
        diff_x_ba = abs(b[0] - a[0])
        diff_y_ba = abs(b[1] - a[1])
        diff_x_bc = abs(b[0] - c[0])
        diff_y_bc = abs(b[1] - c[1])
        if diff_x_ba + diff_x_bc > diff_y_ba + diff_y_bc:
            a[1] = int((a[1] * weight + b[1] + c[1]) / (3 + weight - 1))
            b[1] = a[1]
            c[1] = a[1]
        else:
            a[0] = int((a[0] * weight + b[0] + c[0]) / (3 + weight - 1))
            b[0] = a[0]
            c[0] = a[0]
    return a, b, c, right_angle


@njit
def delete_rows(arr: np.ndarray, indices: [int]) -> np.ndarray:
    mask = np.full(arr.shape[0], True)
    for i in indices:
        mask[i] = False

    return arr[mask]


@njit
def delete_row(arr: np.ndarray, i: int) -> np.ndarray:
    mask = np.full(arr.shape[0], True)
    mask[i] = False

    return arr[mask]


@njit
def straighten_contours(contours: list):
    straight_contours = []
    for cnt in contours:
        idx = 0
        ext_left = cnt[cnt[:, :, 0].argmin()][0]
        ext_right = cnt[cnt[:, :, 0].argmax()][0]
        ext_top = cnt[cnt[:, :, 1].argmin()][0]
        ext_bot = cnt[cnt[:, :, 1].argmax()][0]

        while idx != int(cnt.size / 2):
            try:
                angle = get_angle(cnt[idx:idx + 10])
            except Exception:
                idx += 1
                continue
            if idx + 2 >= len(cnt):
                idx += 1
                continue
            a, b, c, right_angle = move_points(
                cnt, cnt[idx:idx + 3],
                angle,
                [ext_left, ext_right, ext_bot, ext_top])
            cnt[idx][0] = a
            cnt[idx + 1][0] = b
            cnt[idx + 2][0] = c
            idx += 1
            if not right_angle:
                idx -= 1
                cnt = delete_row(cnt, idx + 1)

            if idx == 1:
                cnt = np.append(cnt, cnt[:2], axis=0)

        cnt = delete_rows(cnt, [0, 1])
        squeezed = [coord[0] for coord in cnt]
        straight_contours.extend(squeezed)
    return straight_contours



def make_content_bounds(original_image: np.ndarray, preprocessed=False):
    mask = original_image
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if not preprocessed:
        mask[:, :] = 255 - mask[:, :]

    return _make_content_bounds(mask, np.array([], dtype=np.int32).reshape((0, 2)))


@njit
def _make_content_bounds(mask: np.ndarray, poly: np.ndarray):
    for i in range(0, mask.shape[0]):
        nonzero = np.where(mask[i, :] != 0)
        if nonzero[0].shape[0] != 0:
            poly = np.append(poly, np.array([[nonzero[0][0], i]], dtype=np.int32), axis=0)

    for i in range(mask.shape[0] - 1, -1, -1):
        nonzero = np.where(mask[i, :] != 0)
        if nonzero[0].shape[0] != 0:
            poly = np.append(poly, np.array([[nonzero[0][-1], i]], dtype=np.int32), axis=0)

    return poly


def make_mask(poly, mask_shape):
    if mask_shape[-1] == 3:
        mask = np.zeros(mask_shape[:2], dtype=np.uint8)
    else:
        mask = np.zeros(mask_shape, dtype=np.uint8)

    if len(poly) != 0:
        mask = cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1)

    return mask


def in_content_bounds(content_mask, points, img_shape):
    points_img = np.zeros(img_shape, dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    points_img = cv2.drawContours(points_img, [points], -1, (255, 255, 255), -1)
    # cuts contours from nn by content mask
    points_img = cv2.bitwise_and(points_img, points_img, mask=content_mask)
    # takes new shape of contours
    new_points = make_content_bounds(deepcopy(points_img), preprocessed=True)

    return new_points


class BaseModule(nn.Module):
    """
    Network block with basic functionality
    """

    def __init__(self, cuda: bool = False):
        super().__init__()
        self._str_device = 'cuda' if cuda else 'cpu'
        self._device = torch.device(self._str_device)

    def get_device(self):
        return self._device

    def load_dict(self, path: str):
        try:
            # note: tape for script working
            self.load_state_dict(torch.load(path, map_location=self._str_device))
        except OSError as exc:
            raise NetworkError(f'Cannot load from {path}') from exc


class MaskRCNN(mask_rcnn.MaskRCNN, BaseModule):
    _DEFAULT_IMAGE_SIZE = 1024

    _CNT_TOL = 1e-3

    def __init__(self, cuda: bool = False):
        init_net = mask_rcnn.maskrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=False,
            trainable_backbone_layers=1)
        backbone = init_net.backbone
        in_features = init_net.roi_heads.box_predictor.cls_score.in_features
        box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = init_net.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
        super().__init__(
            backbone, None,
            rpn_head=init_net.rpn.head,
            box_head=init_net.roi_heads.box_head,
            box_predictor=box_predictor,
            mask_head=init_net.roi_heads.mask_head,
            mask_predictor=mask_predictor)
        self._str_device = 'cuda' if cuda else 'cpu'
        self._device = torch.device(self._str_device)
        self.default_transform = None
        self.init_default_transform()

    def init_default_transform(self, tr=None):
        transform = [ToTensor()]
        if tr is not None:
            transform.append(tr)
        transform.append(Pad(fill=0))
        transform.append(Resize(size=self._DEFAULT_IMAGE_SIZE))
        self.default_transform = Compose(transform)

    def preprocess(self, image_bytes: bytes):
        np_buffer = np.fromstring(image_bytes, np.uint8)
        raw = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)
        content_mask = self.make_mask(raw)
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        orig_shape = image.shape[:2]  # Save original shape here to rescale results
        inputs = self.default_transform(image)
        return inputs.unsqueeze(0).double().to(self._device), orig_shape, content_mask

    @staticmethod
    def get_content_bounds(original_image, preprocessed=False):
        return make_content_bounds(deepcopy(original_image), preprocessed)

    @staticmethod
    def make_mask(original_image, preprocessed=False):
        poly = make_content_bounds(deepcopy(original_image), preprocessed)
        return make_mask(poly, original_image.shape)

    @classmethod
    def approximate_poly(cls, polygons):
        c = max(polygons, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        return cv2.approxPolyDP(c, cls._CNT_TOL * peri, True)

    @classmethod
    def postprocess(cls, outputs, orig_shape, content_mask):
        # Compute rescaling coefficients
        orig_h, orig_w = orig_shape
        if orig_w > orig_h:
            coeff = orig_w / cls._DEFAULT_IMAGE_SIZE
            pad_x = 0
            pad_y = (orig_w - orig_h) // 2
        else:
            coeff = orig_h / cls._DEFAULT_IMAGE_SIZE
            pad_x = (orig_h - orig_w) // 2
            pad_y = 0

        approx_points = []
        outputs = outputs[0]['masks'].squeeze(1).cpu().numpy()
        for raw_mask in outputs:
            thr_mask = (raw_mask > 0.2).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(thr_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue

            approx = cls.approximate_poly(cnts)
            points_rescaled = [  # Rescale to return coords mapped to original image
                [int(x * coeff) - pad_x, int(y * coeff) - pad_y]
                for x, y in approx[:, 0]]
            points = in_content_bounds(content_mask, points_rescaled, content_mask.shape)
            if len(points) == 0:
                continue

            approx = cls.approximate_poly([points])
            points = straighten_contours([approx])
            approx_points.append(points)
        return approx_points
