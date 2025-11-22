import os
from supabase import create_client, Client

from configs.config import SUPABASE_URL, SUPABASE_KEY


class SupabaseDB:
    def init_supabase(self) -> Client:
        """Inicializa e retorna o cliente do Supabase"""
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            return client
        except Exception as e:
            print(f"Erro ao conectar: {e}")
            return None
        
    def upload_image(self, supabase: Client, bucket_name: str, file_path: str, file_name: str):
        """
        Faz upload da imagem para o Storage e retorna a URL pública.
        """
        try:
            # 1. Ler o arquivo binário
            with open(file_path, 'rb') as f:
                # Note o uso de 'from_' (com underline) pois 'from' é reservado em Python
                response = supabase.storage.from_(bucket_name).upload(
                    path=file_name,
                    file=f,
                    file_options={"content-type": "image/jpeg"} # Ajuste conforme necessário
                )
            
            # 2. Gerar a URL pública
            # O método get_public_url retorna a URL diretamente ou dentro de um objeto dependendo da versão
            public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_name)
            
            # Verifique se sua versão retorna string direta ou objeto.
            # Geralmente é uma string direta nas versões mais novas.
            return public_url_response

        except Exception as e:
            print(f"Erro no upload: {e}")
            return None

    def save_to_database(self, supabase: Client, table_name: str, user_data: dict):
        """
        Salva os dados (incluindo a URL da imagem) na tabela do banco.
        """
        try:
            response = supabase.table(table_name).insert(user_data).execute()
            return response.data
        except Exception as e:
            print(f"Erro ao salvar no banco: {e}")
            return None