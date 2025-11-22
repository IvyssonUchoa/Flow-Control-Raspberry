import os
import cv2
import time
import numpy as np
import requests

from configs.config import LOG_SAVE_PATH, IMAGE_SAVE_PATH
from classes.SupabaseDB import SupabaseDB


# ----------------------------------------------------------------------
# FUNÇÃO DE REGISTROS LOGS
# ----------------------------------------------------------------------
def log_results(status, data):
    """
    Abre um log para registrar os resultados de processamento e erros. 
    """
    os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)

    with open(LOG_SAVE_PATH, "a") as log:
        log_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"[{log_date}] - {status} - {data}\n"

        log.write(message)

    return


# ----------------------------------------------------------------------
# FUNÇÃO DE REGISTRO NO BD
# ----------------------------------------------------------------------
def save_to_database(fases, angulo):
    """
    Armazena a imagem capturada no Storage do Supabase e registra os dados de captura no banco de dados
    """

    try:
        supabase = SupabaseDB()
        db = supabase.init_supabase()

        if db:
            caminho_local = "captured_images/current_capture.jpg" 
            nome_no_storage = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".jpg"

            public_url = supabase.upload_image(db, "registros", caminho_local, nome_no_storage)

            if public_url:                
                novo_registro = {
                    "imagem": public_url,
                    "fases_detectadas": f"{fases}",
                    "angulo_definido": f"{angulo} graus"
                }
                
                dados_salvos = supabase.save_to_database(db, "registro", novo_registro)
                print(f"Registro Salvo no Banco de dados Supabase {dados_salvos}")

            else:
                log_results(
                    status="FALHA",
                    data=f"Falha ao obter URL da imagem."
                )
        else:
            log_results(
                status="FALHA",
                data=f"Falha ao inicializar o cliente Supabase."
            )

    except Exception as e:
        # Registra o log
        log_results(
            status="FALHA",
            data=f"Erro em salva no banco: {e}"
        )


# ----------------------------------------------------------------------
# FUNÇÃO DE CAPTURA E SALVAMENTO
# ----------------------------------------------------------------------
def capture_and_save_image(url: str, save_path: str):
    """
    Faz uma requisição HTTP para a ESP32-CAM, decodifica e salva a imagem.
    Retorna True se for bem-sucedido, False caso contrário.
    """

    print(f"📸 Tentando capturar imagem de: {url}")
    try:
        os.makedirs(os.path.dirname(IMAGE_SAVE_PATH), exist_ok=True)

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
    