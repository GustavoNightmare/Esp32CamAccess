import os

def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None or v == "" else v

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default

# Servidor
PORT = _env_int("PORT", 8000)

# Seguridad muy simple (para LAN): el ESP32 debe enviar este token en la URL del WS:
# ws://HOST:PORT/ws/esp32?token=...
ESP32_TOKEN = _env_str("ESP32_TOKEN", "CAM_TOKEN_CAMBIAESTO")

# Paths
DB_PATH = _env_str("DB_PATH", "/app/data/db")

# Modelos (ya vienen dentro de la imagen Docker)
FACE_DET_MODEL = _env_str("FACE_DET_MODEL", "/app/models/face_detection_yunet_2023mar.onnx")
FACE_REC_MODEL = _env_str("FACE_REC_MODEL", "/app/models/face_recognition_sface_2021dec.onnx")

# Detector (YuNet)
DET_CONF_THRESHOLD = _env_float("DET_CONF_THRESHOLD", 0.9)
DET_NMS_THRESHOLD = _env_float("DET_NMS_THRESHOLD", 0.3)
DET_TOP_K = _env_int("DET_TOP_K", 5000)

# Reconocimiento (SFace)
# Cosine similarity: mayor = más similar (máx 1.0)
# Umbral de referencia en LFW ~ 0.363, pero para control de acceso suele ser mejor algo más estricto (p.ej. 0.45-0.55).
COSINE_THRESHOLD = _env_float("COSINE_THRESHOLD", 0.45)

# Anti-falsos positivos: requiere N frames seguidos del mismo ID
CONFIRM_FRAMES = _env_int("CONFIRM_FRAMES", 3)

# Evitar que esté abriendo todo el tiempo
COOLDOWN_S = _env_float("COOLDOWN_S", 3.0)

# Cuánto tiempo se queda en pantalla el estado "ENTRA" después de conceder
GRANT_HOLD_S = _env_float("GRANT_HOLD_S", 2.0)

# Cada cuánto se permite recalcular el embedding/match (ms)
RECOG_INTERVAL_MS = _env_int("RECOG_INTERVAL_MS", 150)

# Streaming MJPEG
STREAM_FPS = _env_int("STREAM_FPS", 15)
STREAM_JPEG_QUALITY = _env_int("STREAM_JPEG_QUALITY", 80)
