import cv2
import numpy as np
import onnxruntime as ort

class ONNXDetector:
    """
    Classe para processamento e detecção de objetos com ONNX
    """

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