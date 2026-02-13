"""YuNet wrapper (mínimo) para cv2.FaceDetectorYN.

Basado en el uso oficial de OpenCV (FaceDetectorYN) y el modelo YuNet.
"""

from __future__ import annotations

import numpy as np
import cv2 as cv


class YuNet:
    def __init__(
        self,
        modelPath: str,
        inputSize=(320, 320),
        confThreshold: float = 0.9,
        nmsThreshold: float = 0.3,
        topK: int = 5000,
        backendId: int = 0,
        targetId: int = 0,
    ):
        self._model_path = modelPath
        self._input_size = tuple(inputSize)
        self._conf = float(confThreshold)
        self._nms = float(nmsThreshold)
        self._topk = int(topK)
        self._backend = int(backendId)
        self._target = int(targetId)

        self._detector = cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf,
            nms_threshold=self._nms,
            top_k=self._topk,
            backend_id=self._backend,
            target_id=self._target,
        )

    def setInputSize(self, input_size):
        self._detector.setInputSize(tuple(input_size))

    def infer(self, image):
        # OpenCV devuelve (retval, faces). faces puede ser None.
        faces = self._detector.detect(image)
        # faces: (retval, mat)
        mat = faces[1] if isinstance(faces, (tuple, list)) and len(faces) > 1 else None
        if mat is None:
            # estándar: 0 faces
            return np.empty((0, 15), dtype=np.float32)
        return mat
