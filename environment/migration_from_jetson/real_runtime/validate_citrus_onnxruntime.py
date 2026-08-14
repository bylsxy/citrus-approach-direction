#!/usr/bin/env python3
"""Validate the pinned CPU ONNX stack without starting ROS motion nodes."""

import hashlib
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from cv_bridge import CvBridge

from elfin_vision.onnx_yolo_seg import OnnxYoloSeg


MODEL = os.environ.get(
    'ELFIN_ONNX_MODEL',
    '/home/catas/elfin_citrus_data/server_bundle_20260722/'
    'best_citrus_seg.onnx')
EXPECTED_SHA256 = (
    '507821ee21f312497baf14c85ee04b10fe09be6a2bbaa71a3a420a49e3988341')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    print('numpy=%s cv2=%s onnxruntime=%s' % (
        np.__version__, cv2.__version__, ort.__version__))
    print('providers=%s' % ort.get_available_providers())
    actual_sha256 = file_sha256(MODEL)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError('model SHA256 mismatch: %s' % actual_sha256)
    print('model_sha256=%s' % actual_sha256)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bridge = CvBridge()
    message = bridge.cv2_to_imgmsg(image, encoding='bgr8')
    round_trip = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
    if round_trip.shape != image.shape or round_trip.dtype != image.dtype:
        raise RuntimeError('cv_bridge NumPy round trip changed image layout')
    print('cv_bridge_round_trip=OK shape=%s dtype=%s' % (
        round_trip.shape, round_trip.dtype))

    started = time.monotonic()
    model = OnnxYoloSeg(
        MODEL, {0: 'citrus', 1: 'tree'}, confidence=0.35, iou=0.45,
        image_size=640)
    load_s = time.monotonic() - started
    started = time.monotonic()
    detections = model.infer(image)
    inference_s = time.monotonic() - started
    print('model_load_s=%.3f inference_s=%.3f detections=%d providers=%s' % (
        load_s, inference_s, len(detections), model.providers))
    print('VALIDATION=PASS')


if __name__ == '__main__':
    main()
