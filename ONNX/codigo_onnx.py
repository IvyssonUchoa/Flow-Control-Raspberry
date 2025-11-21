import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import time
import requests
import os 
from typing import List, Dict

# --- CONFIGURAÇÃO ---
IMAGE_SAVE_PATH = "captured_images/current_capture.jpg"
LOG_SAVE_PATH = "logs/results_log.txt"

ESP32_CAM_URL = "http://192.168.1.14/capture" # AJUSTE O IP!
MODEL_PATH = "best_nano.onnx" 

BROKER = "192.168.1.8"
PORT = 1883
TOPIC = "hidroponia/servo"
LOOP_INTERVAL_SECONDS = 3600  # 1 hora

# --------------------
# Garante que o diretório de salvamento exista
os.makedirs(os.path.dirname(IMAGE_SAVE_PATH), exist_ok=True)

# Garante que o diretório de Logs exista
os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)
# --------------------

# ----------------------------------------------------------------------
# CLASSE: ONNXYOLODetector (Retorna a lista de nomes das fases)
# ----------------------------------------------------------------------

class ONNXYOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        self.session = ort.InferenceSession(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_name = self.session.get_inputs()[0].name
        
        self.fase_to_angle = {
            "fase_1": 30,
            "fase_2": 60,
            "fase_3": 90
        }
        
    def preprocess(self, image_path: str):
        """
        Pré-processamento da imagem, incluindo ROTAÇÃO DE 90 GRAUS PARA A ESQUERDA.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Imagem não encontrada ou inválida: {image_path}")
        
        # 1. ROTAÇÃO DE 90 GRAUS PARA A ESQUERDA (Anti-horário)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("🔄 Imagem rotacionada 90° para a esquerda.")

        # 2. Redimensionar e normalizar
        input_img = cv2.resize(image, (640, 640))
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0
        
        return input_img, image.shape[:2]
    
    def postprocess(self, outputs, original_shape):
        """
        Pós-processamento das detecções.
        ***MODIFICADO: Retorna uma lista dos nomes das classes detectadas.***
        """
        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        
        valid_detections = scores > self.conf_threshold
        predictions = predictions[valid_detections]
        scores = scores[valid_detections]
        
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        keep_indices = self.non_max_suppression(predictions[:, :4], scores)
        
        # Lista para armazenar os nomes das fases (strings)
        detected_phases = []
        for idx in keep_indices:
            class_id = int(class_ids[idx])
            class_name = f"fase_{class_id + 1}"
            if class_name in self.fase_to_angle:
                detected_phases.append(class_name)
        
        # Retorna a lista de nomes das fases
        return detected_phases 
    
    def non_max_suppression(self, boxes, scores):
        """Implementação simples de NMS (inalterada)"""
        if len(boxes) == 0: return []
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        return keep
    
    def detect(self, image_path: str):
        """Executar detecção em uma imagem, recebendo o caminho do arquivo"""
        try:
            input_tensor, original_shape = self.preprocess(image_path)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            # Recebe a lista de nomes das fases
            detected_phases = self.postprocess(outputs, original_shape)
            return detected_phases
        except Exception as e:
            print(f"⚠️ Erro na detecção para o arquivo '{image_path}': {e}")
            return []

# ----------------------------------------------------------------------
# FUNÇÃO DE REGISTROS LOGS
# ----------------------------------------------------------------------

def log_results(status, data):
    """
    Abre um log para registrar os resultados de processamento e erros. 
    """

    with open(LOG_SAVE_PATH, "a") as log:
        log_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[{log_date}] - {status} - {data}\n"

        log.write(message)

    return

# ----------------------------------------------------------------------
# FUNÇÃO DE CAPTURA E SALVAMENTO
# ----------------------------------------------------------------------

def capture_and_save_image(url: str, save_path: str):
    """
    Faz uma requisição HTTP para a ESP32-CAM, decodifica e salva a imagem.
    Retorna True se for bem-sucedido, False caso contrário.
    """
    print(f"\n--- [ETAPA 1: CAPTURA] ---")
    print(f"📸 Tentando capturar imagem de: {url}")
    try:
        response = requests.get(url, timeout=5) 
        response.raise_for_status() 

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Não foi possível decodificar a imagem recebida.")

        # Salva a imagem no caminho especificado
        cv2.imwrite(save_path, image)
        
        print(f"✅ Imagem capturada e salva em: {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na requisição HTTP (ESP32-CAM): {e}")

        # Registra o log
        log_results(
            status="FALHA",
            data=f"Erro na requisição HTTP (ESP32-CAM): {e}"
        )

        return False
    except Exception as e:
        print(f"❌ Erro ao processar/salvar a imagem: {e}")

        # Registra o log
        log_results(
            status="FALHA",
            data=f"Erro ao processar/salvar a imagem: {e}"
        )

        return False

# ----------------------------------------------------------------------
# LOOP PRINCIPAL (MODIFICADO: Imprime as classes e envia um valor fixo/neutro)
# ----------------------------------------------------------------------

# Inicialização
try:
    detector = ONNXYOLODetector(MODEL_PATH)
except Exception as e:
    print(f"Fatal: Não foi possível carregar o modelo ONNX. Erro: {e}")
    exit()

client = mqtt.Client()
try:
    client.connect(BROKER, PORT, 60)
    client.loop_start() 
    print(f"🔗 Conectado ao broker MQTT: {BROKER}:{PORT}")
except Exception as e:
    print(f"⚠️ Não foi possível conectar ao broker MQTT. Erro: {e}")

print("\n--- INICIANDO LOOP DE CLASSIFICAÇÃO SEQUENCIAL ---\n")
while True:
    
    # 1. CAPTURA E SALVAMENTO
    image_saved = capture_and_save_image(ESP32_CAM_URL, IMAGE_SAVE_PATH)
    
    if image_saved:
        
        # 2. PROCESSAMENTO E CLASSIFICAÇÃO
        print(f"\n--- [ETAPA 2: PROCESSAMENTO] ---")
        # detect() agora retorna uma lista de STRINGS (nomes das fases)
        detected_phases = detector.detect(IMAGE_SAVE_PATH) 
        
        if detected_phases:
            # Imprime todas as classes encontradas
            print(f"📈 Fases detectadas: {detected_phases}")
            
            # ATENÇÃO: Se não for mais para calcular a média, o que você quer enviar via MQTT?
            # Manteremos a lógica original de envio do primeiro ângulo encontrado (ou de um valor neutro).
            
            # Exemplo de envio: A fase mais comum (ou a primeira) é usada para o ângulo
            first_phase = detected_phases[0]
            angle_to_send = detector.fase_to_angle.get(first_phase, 0) # Usa 0 se não encontrar
            
            # Registra o log
            log_results(
                status="SUCESSO",
                data=f"Fases detectadas: {detected_phases} | Angulo correspondente: {angle_to_send}"
            )

            # 3. ENVIO MQTT
            client.publish(TOPIC, str(angle_to_send))
            print(f"🎉 Ângulo correspondente enviado via MQTT ({first_phase}) → {angle_to_send}°")
            
        else:
            print("🧐 Nenhuma fase detectada no arquivo capturado.")

            # Registra o log
            log_results(
                status="FALHA",
                data=f"Nenhuma fase detectada no arquivo capturado"
            )
            
    else:
        print("\n🤷 Pulando processamento: Falha na captura de imagem.")
        
    print(f"\n========================================================")
    print(f"Aguardando {LOOP_INTERVAL_SECONDS} segundos para a próxima rodada...")
    time.sleep(LOOP_INTERVAL_SECONDS)