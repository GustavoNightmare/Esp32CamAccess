"""SFace wrapper (mínimo) para cv2.FaceRecognizerSF."""

from __future__ import annotations

import cv2 as cv


class SFace:
    # distType: 0 = cosine similarity, 1 = normL2 distance
    DIST_COSINE = 0
    DIST_NORML2 = 1

    def __init__(
        self,
        modelPath: str,
        backendId: int = 0,
        targetId: int = 0,
    ):
        self._model_path = modelPath
        self._backend = int(backendId)
        self._target = int(targetId)

        self._rec = cv.FaceRecognizerSF.create(
            model=self._model_path,
            config="",
            backend_id=self._backend,
            target_id=self._target,
        )

    def align_crop(self, image, face_bbox_row):
        return self._rec.alignCrop(image, face_bbox_row)

    def feature(self, aligned_face):
        return self._rec.feature(aligned_face)

    def infer(self, image, face_bbox_row=None):
        aligned = image if face_bbox_row is None else self.align_crop(image, face_bbox_row)
        return self.feature(aligned)

    def match(self, feat1, feat2, dist_type: int = DIST_COSINE):
        return self._rec.match(feat1, feat2, dist_type)
