from configs.config import *
from utils.functions import *
from classes.ONNXDetector import ONNXDetector


# Conecta ao broker MQTT
try:
    import paho.mqtt.client as mqtt

    client = mqtt.Client()
    client.connect(BROKER, PORT, 60)
    client.loop_start() 
    print(f"🔗 Conectado ao broker MQTT: {BROKER}:{PORT}")
except Exception as e:
    print(f"⚠️ Não foi possível conectar ao broker MQTT. Erro: {e}")


# Carrega o modelo ONNX
try:
    detector = ONNXDetector(MODEL_PATH)
except Exception as e:
    print(f"Fatal: Não foi possível carregar o modelo ONNX. Erro: {e}")
    exit()


# Inicia a obtenção e processamento de Imagem
try:
    # CAPTURA E SALVAMENTO
    image_saved = capture_and_save_image(ESP32_CAM_URL, IMAGE_SAVE_PATH)

    if image_saved:
        # PROCESSAMENTO E CLASSIFICAÇÃO
        detected_phases = detector.detect(IMAGE_SAVE_PATH) 
        
        if detected_phases:
            # Imprime todas as classes encontradas
            print(f"📈 Fases detectadas: {detected_phases}")
            
            # Usando primeira fase detectada para envio
            # Substituir por média das fases
            first_phase = detected_phases[0]
            angle_to_send = detector.fase_to_angle.get(first_phase, 0) # Usa 0 se não encontrar

            # ENVIO MQTT
            client.publish(TOPIC, str(angle_to_send))
            print(f"🎉 Ângulo correspondente enviado via MQTT ({first_phase}) → {angle_to_send}°")

            # REGISTRO DE LOG E BANCO
            save_to_database(detected_phases, angle_to_send)

            log_results(
                status="SUCESSO",
                data=f"Fases detectadas: {detected_phases} | Angulo correspondente: {angle_to_send}"
            )
            
        else:
            print("Nenhuma fase detectada no arquivo capturado.")

            # REGISTRO DE LOG
            log_results(
                status="FALHA",
                data=f"Nenhuma fase detectada no arquivo capturado"
            )
            
    else:
        print("\nPulando processamento: Falha na captura de imagem.")

except Exception as e:
    print(f"Falha de processamento: {e}")

    # REGISTRO DE LOG
    log_results(
        status="FALHA",
        data=f"Falha de processamento: {e}"
    )