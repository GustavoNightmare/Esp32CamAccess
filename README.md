# ESP32-CAM (OV5640) → Servidor Flask (Docker) → Reconocimiento Facial → LED/Relé

Este proyecto es un **MVP** para control de acceso:

- La **ESP32-CAM** captura frames JPEG (≈20 FPS) y los envía por **WebSocket**.
- Un servidor **Flask** (en Docker) recibe los frames, hace:
  - **Detección de rostro** (YuNet / OpenCV DNN)
  - **Embedding + comparación** (SFace / OpenCV)
- El servidor devuelve una decisión al ESP32 (JSON), para encender un **LED RGB** y/o activar un **relé**.
- Una web ligera muestra el stream con “máscara” (bbox/landmarks) y el estado “ENTRA / NO ENTRA”.

> Nota: Este proyecto está pensado para **LAN**. No expongas el puerto a Internet sin TLS, autenticación fuerte, etc.

---

## 1) Servidor (Orange Pi / Linux)

### Requisitos

- Docker + Docker Compose
- Internet para descargar modelos ONNX en el build

### Ejecutar

Desde la carpeta raíz:

```bash
docker compose up --build
```

Abrir en el navegador:

- `http://IP_DE_TU_ORANGEPI:8000/`

### Base de datos de rostros

La DB vive en:

- `./data/db/<NOMBRE>/*.jpg`

También puedes usar la página:

- `http://IP_DE_TU_ORANGEPI:8000/enroll`

Para recargar sin reiniciar:

- botón “Recargar DB” en la web
- o `POST /api/reload_db`

---

## 2) ESP32-CAM (Arduino)

Abre `esp32/esp32_cam_access.ino`

Edita:

- WiFi SSID/PASS
- IP del servidor
- Token (`ESP32_TOKEN`) → debe ser el mismo que en docker-compose.yml

Dependencias Arduino:

- ESP32 by Espressif (Arduino core)
- WebSocketsClient
- ArduinoJson

La asignación de pines de cámara depende de tu placa. Ajusta `camera_pins.h` si tu módulo no es el típico AI-Thinker.

---

## Ajustes finos

En `docker-compose.yml` puedes ajustar:

- `COSINE_THRESHOLD` (más alto = más estricto)
- `CONFIRM_FRAMES` (más frames seguidos para dar acceso)
- `RECOG_INTERVAL_MS` (cada cuánto recalcula el match)
- `STREAM_FPS` (FPS a la web)
