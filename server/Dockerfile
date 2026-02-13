FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias mínimas para OpenCV (headless) + descarga de modelos
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Descarga de modelos (YuNet + SFace) desde el repositorio oficial en HuggingFace (OpenCV Zoo)
RUN mkdir -p /app/models \
    && wget -O /app/models/face_detection_yunet_2023mar.onnx \
        https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && wget -O /app/models/face_recognition_sface_2021dec.onnx \
        https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

COPY . /app

EXPOSE 8000

# 1 worker para que el estado compartido (último frame / última decisión) sea único.
CMD ["gunicorn", "-k", "gevent", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
