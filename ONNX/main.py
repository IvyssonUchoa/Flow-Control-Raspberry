import requests
import cv2
import numpy as np
import onnxruntime as ort

# Caminho para o modelo ONNX
MODEL_PATH = "best.onnx"

# Lista de classes (substitua pelos rótulos reais do seu modelo)
CLASSES = ["classe0", "classe1", "classe2", "classe3", "classe4"]

# IP da ESP32-CAM
ESP32_URL = "http://192.168.1.14/capture"  # ajuste conforme seu ESP

# Função de pré-processamento
def preprocess_cv2(cv2_img, input_size=(640, 640)):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # (C, H, W)
    img_data = np.expand_dims(img_data, axis=0)    # (1, C, H, W)
    return img_data

try:
    print("Capturando imagem da ESP32-CAM...")
    response = requests.get(ESP32_URL, timeout=10)

    if response.status_code == 200:
        # Converte bytes para imagem OpenCV
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Rotaciona se necessário
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Imagem capturada e rotacionada.")

        # Pré-processamento
        img_input = preprocess_cv2(rotated_img)

        # Carrega modelo ONNX
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Inferência
        preds = session.run([output_name], {input_name: img_input})[0][0]  # [0][0] pega o vetor da imagem

        # Imprime todas as classes com suas probabilidades
        print("\nClasses previstas:")
        for idx, prob in enumerate(preds):
            print(f"{CLASSES[idx]}: {prob:.4f}")

    else:
        print(f"Erro HTTP {response.status_code}. Verifique o IP da ESP32-CAM.")

except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}")
except Exception as e:
    print(f"Ocorreu um erro: {e}")
