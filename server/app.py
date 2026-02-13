from __future__ import annotations

import json
import time
import threading
import queue
from typing import Optional

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_sock import Sock

import config
from recognizer import FaceAccessController


app = Flask(__name__)
sock = Sock(app)

# Controlador (carga modelos + DB). Se comparte entre hilo de procesamiento y endpoints.
controller = FaceAccessController()
controller_lock = threading.Lock()

print("[server] DB cargada:", controller.db_summary)

# Cola de frames desde el ESP32 (tamaño 1 para 'dropear' frames viejos y mantener tiempo real)
frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1)

state_lock = threading.Lock()
latest_jpeg: Optional[bytes] = None
latest_status: dict = {
    "ts": 0.0,
    "found_face": False,
    "name": "sin_cara",
    "cosine": 0.0,
    "det_conf": 0.0,
    "allowed": False,
    "event_id": 0,
}

decision_lock = threading.Lock()
last_decision: dict = {
    "allowed": False,
    "name": "sin_cara",
    "cosine": 0.0,
    "event_id": 0,
    "rgb": [0, 0, 255],
    "unlock_ms": 800,
}
decision_version: int = 0


def _placeholder_jpeg() -> bytes:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv.putText(img, "Esperando ESP32...", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    ok, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


PLACEHOLDER = _placeholder_jpeg()


def processor_loop():
    global latest_jpeg, latest_status, last_decision, decision_version

    last_sent_copy = None

    while True:
        jpeg_bytes = frame_queue.get()  # bloquea hasta tener frame

        # Decodifica JPEG
        npbuf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv.imdecode(npbuf, cv.IMREAD_COLOR)
        if frame is None:
            continue

        with controller_lock:
            annotated, match, _ = controller.process(frame)

        # Encode JPEG para el navegador (mjpeg)
        q = int(getattr(config, "STREAM_JPEG_QUALITY", 80))
        ok, out = cv.imencode(".jpg", annotated, [int(cv.IMWRITE_JPEG_QUALITY), q])
        if ok:
            with state_lock:
                latest_jpeg = out.tobytes()
                latest_status = {
                    "ts": time.time(),
                    "found_face": match.found_face,
                    "name": match.name,
                    "cosine": match.cosine,
                    "det_conf": match.det_conf,
                    "allowed": match.allowed,
                    "event_id": match.event_id,
                }

        # Decisión para ESP32 (RGB en formato [R,G,B])
        if not match.found_face:
            rgb = [0, 0, 255]  # azul
        else:
            if match.allowed:
                rgb = [0, 255, 0]  # verde
            elif match.name == "desconocido":
                rgb = [255, 0, 0]  # rojo
            else:
                rgb = [255, 255, 0]  # amarillo

        decision = {
            "allowed": bool(match.allowed),
            "name": match.name,
            "cosine": float(match.cosine),
            "event_id": int(match.event_id),
            "rgb": rgb,
            "unlock_ms": 800,
        }

        # Actualiza decisión sólo si cambia (para no spamear)
        with decision_lock:
            if last_sent_copy is None or decision != last_sent_copy:
                last_decision = decision
                decision_version += 1
                last_sent_copy = decision.copy()


# Lanza el hilo de procesamiento
threading.Thread(target=processor_loop, daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/mjpeg")
def mjpeg():
    boundary = "frame"
    fps = max(1, int(getattr(config, "STREAM_FPS", 15)))
    delay = 1.0 / fps

    def gen():
        nonlocal delay
        while True:
            with state_lock:
                frame = latest_jpeg or PLACEHOLDER
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
            time.sleep(delay)

    return Response(gen(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/api/status")
def api_status():
    with state_lock:
        s = dict(latest_status)
    return jsonify(s)


@app.get("/api/db")
def api_db():
    with controller_lock:
        summary = dict(controller.db_summary)
    return jsonify({"db_path": config.DB_PATH, "people": summary})


@app.post("/api/reload_db")
def api_reload_db():
    with controller_lock:
        summary = controller.reload_db()
    return jsonify({"ok": True, "people": summary})


@app.get("/enroll")
def enroll_form():
    return render_template("enroll.html")


@app.post("/enroll")
def enroll_post():
    # Subida simple de imagen + nombre
    name = (request.form.get("name") or "").strip()
    f = request.files.get("image")
    if not name:
        return jsonify({"ok": False, "error": "Falta name"}), 400
    if f is None:
        return jsonify({"ok": False, "error": "Falta image"}), 400

    data = f.read()
    npbuf = np.frombuffer(data, dtype=np.uint8)
    img = cv.imdecode(npbuf, cv.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "Imagen inválida"}), 400

    with controller_lock:
        ok = controller.enroll(name, img)
        summary = dict(controller.db_summary)

    if not ok:
        return jsonify({"ok": False, "error": "No se detectó un rostro en la imagen"}), 400

    return jsonify({"ok": True, "people": summary})


@sock.route("/ws/esp32")
def ws_esp32(ws):
    # Token en query param
    token = request.args.get("token", "")
    if token != config.ESP32_TOKEN:
        ws.send(json.dumps({"error": "bad_token"}))
        ws.close()
        return

    last_v = -1
    while True:
        msg = ws.receive()
        if msg is None:
            break

        # ignorar mensajes texto
        if isinstance(msg, str):
            continue

        # Encola frame (drop si está llena)
        try:
            frame_queue.put_nowait(msg)
        except queue.Full:
            try:
                _ = frame_queue.get_nowait()
            except Exception:
                pass
            try:
                frame_queue.put_nowait(msg)
            except Exception:
                pass

        # Enviar última decisión si cambió
        with decision_lock:
            v = decision_version
            d = dict(last_decision)

        if v != last_v:
            ws.send(json.dumps(d))
            last_v = v


if __name__ == "__main__":
    # Para debug sin gunicorn
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
