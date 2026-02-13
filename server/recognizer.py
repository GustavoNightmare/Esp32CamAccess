from __future__ import annotations

import os
import time
import glob
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from config import (
    DB_PATH,
    FACE_DET_MODEL,
    FACE_REC_MODEL,
    DET_CONF_THRESHOLD,
    DET_NMS_THRESHOLD,
    DET_TOP_K,
    COSINE_THRESHOLD,
    CONFIRM_FRAMES,
    COOLDOWN_S,
    GRANT_HOLD_S,
    RECOG_INTERVAL_MS,
)
from yunet import YuNet
from sface import SFace


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class FaceMatch:
    found_face: bool
    det_conf: float = 0.0
    name: str = "sin_cara"  # sin_cara | desconocido | <persona>
    cosine: float = 0.0
    allowed: bool = False
    event_id: int = 0
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
    landmarks: Optional[List[Tuple[int, int]]] = None  # 5 puntos


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _iter_images(person_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(person_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(person_dir, f"*{ext.upper()}")))
    paths.sort()
    return paths


def _pick_largest_face(faces: np.ndarray) -> Optional[np.ndarray]:
    if faces is None or len(faces) == 0:
        return None
    # faces: Nx15, bbox = [x,y,w,h]
    areas = faces[:, 2] * faces[:, 3]
    idx = int(np.argmax(areas))
    return faces[idx]


def _row_bbox_landmarks(face_row: np.ndarray) -> Tuple[Tuple[int, int, int, int], List[Tuple[int, int]], float]:
    # face_row: [x,y,w,h, l0x,l0y, l1x,l1y, ... l4x,l4y, score]
    bbox = face_row[0:4].astype(np.int32)
    lm = face_row[4:14].astype(np.int32).reshape((5, 2))
    score = float(face_row[-1])
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), [(int(x), int(y)) for x, y in lm], score


class FaceDB:
    """Carga embeddings (features) desde data/db/<nombre>/*.jpg."""

    def __init__(self, db_path: str, detector: YuNet, recognizer: SFace):
        self.db_path = db_path
        self.detector = detector
        self.recognizer = recognizer
        self.features: Dict[str, List[np.ndarray]] = {}
        self._lock = threading.Lock()

    def reload(self) -> Dict[str, int]:
        """Devuelve un resumen {nombre: cantidad_features}."""
        with self._lock:
            self.features = {}

            if not os.path.isdir(self.db_path):
                _safe_mkdir(self.db_path)
                return {}

            people = [d for d in os.listdir(self.db_path) if os.path.isdir(os.path.join(self.db_path, d))]
            people.sort()

            for person in people:
                pdir = os.path.join(self.db_path, person)
                imgs = _iter_images(pdir)
                feats: List[np.ndarray] = []
                for img_path in imgs:
                    img = cv.imread(img_path)
                    if img is None:
                        continue
                    feat = self._extract_feature(img)
                    if feat is not None:
                        feats.append(feat)
                if feats:
                    self.features[person] = feats

            return {k: len(v) for k, v in self.features.items()}

    def enroll(self, name: str, img_bgr: np.ndarray, save_to_disk: bool = True) -> bool:
        name = name.strip()
        if not name:
            return False

        feat = self._extract_feature(img_bgr)
        if feat is None:
            return False

        with self._lock:
            self.features.setdefault(name, []).append(feat)

        if save_to_disk:
            out_dir = os.path.join(self.db_path, name)
            _safe_mkdir(out_dir)
            ts = int(time.time() * 1000)
            out_path = os.path.join(out_dir, f"{ts}.jpg")
            cv.imwrite(out_path, img_bgr)
        return True

    def match(self, feat: np.ndarray) -> Tuple[str, float]:
        """Devuelve (nombre, cosine_sim). Si no hay match, retorna ('desconocido', score)."""
        feat = _l2_normalize(feat)

        best_name = "desconocido"
        best_score = -1.0

        with self._lock:
            for name, feats in self.features.items():
                for f in feats:
                    s = _cosine_sim(feat, f)
                    if s > best_score:
                        best_score = s
                        best_name = name

        return best_name, float(best_score)

    def _extract_feature(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize([w, h])
        faces = self.detector.infer(img_bgr)
        face_row = _pick_largest_face(faces)
        if face_row is None:
            return None

        # OJO: alignCrop espera la fila tipo (1,15) o (15,). Probamos (1,15) para evitar errores.
        row = face_row.reshape(1, -1).astype(np.float32)
        feat = self.recognizer.infer(img_bgr, row)
        if feat is None:
            return None
        return _l2_normalize(feat)


class FaceAccessController:
    """Control de acceso: detecta + reconoce + decide allowed + genera overlays."""

    def __init__(self):
        self.detector = YuNet(
            modelPath=FACE_DET_MODEL,
            inputSize=(320, 320),
            confThreshold=DET_CONF_THRESHOLD,
            nmsThreshold=DET_NMS_THRESHOLD,
            topK=DET_TOP_K,
        )
        self.recognizer = SFace(modelPath=FACE_REC_MODEL)

        self.db = FaceDB(DB_PATH, self.detector, self.recognizer)
        self.db_summary = self.db.reload()

        # estado para smoothing
        self._last_candidate: str = ""
        self._candidate_count: int = 0
        self._last_match_name: str = "desconocido"
        self._last_match_score: float = 0.0
        self._last_det_conf: float = 0.0
        self._last_face_row: Optional[np.ndarray] = None

        self._last_recog_ms: int = 0
        self._last_grant_ts: float = 0.0
        self._grant_until: float = 0.0
        self._event_id: int = 0

    @property
    def event_id(self) -> int:
        return self._event_id

    def reload_db(self) -> Dict[str, int]:
        self.db_summary = self.db.reload()
        return self.db_summary

    def enroll(self, name: str, img_bgr: np.ndarray) -> bool:
        ok = self.db.enroll(name, img_bgr, save_to_disk=True)
        if ok:
            self.db_summary = {k: len(v) for k, v in self.db.features.items()}
        return ok

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, FaceMatch, bool]:
        """Procesa un frame y devuelve:
        - frame_annotated
        - match (FaceMatch)
        - decision_changed (bool) => útil para mandar al ESP32
        """

        now = time.time()
        now_ms = int(now * 1000)

        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize([w, h])
        faces = self.detector.infer(frame_bgr)
        face_row = _pick_largest_face(faces)

        decision_changed = False

        if face_row is None:
            # Sin cara
            self._candidate_count = 0
            self._last_candidate = ""
            self._last_match_name = "sin_cara"
            self._last_match_score = 0.0
            self._last_det_conf = 0.0
            self._last_face_row = None

            match = FaceMatch(found_face=False, name="sin_cara", cosine=0.0, allowed=False, event_id=self._event_id)
            annotated = self._draw(frame_bgr, match)
            # Si veníamos de mostrar ENTRA, eso puede cambiar la UI, pero para el ESP32 no importa.
            return annotated, match, decision_changed

        # cara detectada
        bbox, lms, det_conf = _row_bbox_landmarks(face_row)
        self._last_det_conf = det_conf
        self._last_face_row = face_row

        # ¿Estamos dentro del hold de acceso?
        in_grant_hold = now < self._grant_until

        # Recalcular reconocimiento (embedding) cada cierto tiempo
        if now_ms - self._last_recog_ms >= RECOG_INTERVAL_MS:
            self._last_recog_ms = now_ms

            row = face_row.reshape(1, -1).astype(np.float32)
            feat = self.recognizer.infer(frame_bgr, row)
            if feat is not None:
                feat = _l2_normalize(feat)
                name, score = self.db.match(feat)
            else:
                name, score = "desconocido", 0.0

            self._last_match_name = name
            self._last_match_score = score

            # Smoothing: sólo cuenta si pasa el umbral
            candidate = name if score >= COSINE_THRESHOLD else "desconocido"

            if candidate != "desconocido" and candidate == self._last_candidate:
                self._candidate_count += 1
            else:
                self._last_candidate = candidate if candidate != "desconocido" else ""
                self._candidate_count = 1 if candidate != "desconocido" else 0

            # ¿Concedemos acceso?
            can_grant = (
                candidate != "desconocido"
                and self._candidate_count >= CONFIRM_FRAMES
                and (now - self._last_grant_ts) >= COOLDOWN_S
            )
            if can_grant:
                self._event_id += 1
                self._last_grant_ts = now
                self._grant_until = now + GRANT_HOLD_S
                decision_changed = True

        # Decisión final (allowed)
        # allowed si estamos en hold (ya concedido) Y el último match no es desconocido
        allowed = in_grant_hold or (now < self._grant_until)

        # Si estamos en hold, mantenemos el nombre mostrado si era conocido
        display_name = self._last_match_name
        display_score = self._last_match_score

        # Si no supera umbral, marcamos como desconocido (pero mantenemos score para debug)
        if display_score < COSINE_THRESHOLD:
            display_name = "desconocido"

        # Si no estamos en hold, allowed debe ser False
        if not (now < self._grant_until) or display_name == "desconocido":
            allowed = False

        match = FaceMatch(
            found_face=True,
            det_conf=det_conf,
            name=display_name,
            cosine=float(display_score),
            allowed=allowed,
            event_id=self._event_id,
            face_bbox=bbox,
            landmarks=lms,
        )

        annotated = self._draw(frame_bgr, match)
        return annotated, match, decision_changed

    def _draw(self, frame_bgr: np.ndarray, match: FaceMatch) -> np.ndarray:
        out = frame_bgr.copy()

        # Texto status
        if not match.found_face:
            status_text = "SIN CARA"
            rgb = (255, 0, 0)  # BGR: azul
        else:
            if match.allowed:
                status_text = "ENTRA"
                rgb = (0, 255, 0)  # verde
            else:
                if match.name == "desconocido":
                    status_text = "NO ENTRA"
                    rgb = (0, 0, 255)  # rojo
                else:
                    status_text = "VERIFICANDO"
                    rgb = (0, 255, 255)  # amarillo

        # Banner
        cv.rectangle(out, (0, 0), (out.shape[1], 40), rgb, thickness=-1)
        cv.putText(out, status_text, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv.LINE_AA)

        # Info extra
        info = ""
        if match.found_face:
            info = f"ID: {match.name}  cos: {match.cosine:.3f}  det: {match.det_conf:.2f}"
        else:
            info = "Esperando rostro..."
        cv.putText(out, info, (10, out.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv.LINE_AA)

        # Dibuja bbox y landmarks si hay cara
        if match.found_face and match.face_bbox is not None:
            x, y, w, h = match.face_bbox
            x2, y2 = x + w, y + h

            # overlay semi-transparente
            overlay = out.copy()
            cv.rectangle(overlay, (x, y), (x2, y2), rgb, thickness=-1)
            alpha = 0.25
            out = cv.addWeighted(overlay, alpha, out, 1 - alpha, 0)

            # borde
            cv.rectangle(out, (x, y), (x2, y2), rgb, 2)

            # landmarks
            if match.landmarks:
                for (lx, ly) in match.landmarks:
                    cv.circle(out, (lx, ly), 2, (255, 255, 255), 2)

            # etiqueta
            label = f"{match.name} ({match.cosine:.2f})"
            cv.putText(out, label, (x, max(20, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

        return out
