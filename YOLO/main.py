import requests
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np

# Caminho para o modelo YOLO treinado
# model_path = "best_nano.pt"
model_path = "best_small.pt"
model = YOLO(model_path)

# IP do ESP32-CAM na sua rede
ESP32_URL = "http://192.168.1.14/capture"  # Altere para o IP da sua ESP32-CAM

try:
    # 1. Pega a imagem da ESP32-CAM
    print("Capturando imagem da ESP32-CAM...")
    response = requests.get(ESP32_URL, timeout=10)

    if response.status_code == 200:
        # 2. Converte o conteúdo da resposta para uma imagem OpenCV
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 3. Rotaciona a imagem 90 graus no sentido anti-horário
        # Você pode alterar o argumento para outras rotações:
        # cv2.ROTATE_90_CLOCKWISE: 90 graus no sentido horário
        # cv2.ROTATE_180: 180 graus
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Imagem rotacionada 90 graus.")

        # 4. Processa a imagem rotacionada com o modelo YOLO
        print("Processando imagem com o modelo YOLO...")
        results = model(rotated_img, save=True, save_conf=True)

        # 5. Encontra o objeto mais comum
        if results and results[0].boxes:
            detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
            if detected_classes:
                class_counts = Counter(detected_classes)
                most_common_object, count = class_counts.most_common(1)[0]
                print(f"Objeto que mais aparece: '{most_common_object}' (contagem: {count})")
            else:
                print("Nenhum objeto foi detectado na imagem.")
        else:
            print("Nenhum objeto foi detectado na imagem.")

        print("A imagem processada com as detecções foi salva na pasta 'runs/detect/'.")

    else:
        print(f"Erro HTTP {response.status_code}. Certifique-se de que o ESP32-CAM está ligado e o IP está correto.")

except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}. Verifique a conexão com a ESP32-CAM e o IP.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")