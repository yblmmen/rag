import os
from netmiko import ConnectHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 네트워크 RAG 관련 파일 경로 (자산관리 RAG와 분리) ---
RAW_DATA_FILE = "network_raw_data.txt"
FAISS_INDEX_PATH = "network_faiss_index" # 폴더명 변경

def fetch_network_data(device_info, commands_to_run):
    """Netmiko를 사용해 네트워크 장비에서 데이터를 수집합니다."""
    all_output = ""
    print("네트워크 장비 데이터 수집 시작...")
    try:
        with ConnectHandler(**device_info) as net_connect:
            net_connect.enable()
            prompt = net_connect.find_prompt()
            for cmd in commands_to_run:
                print(f"'{cmd}' 명령어 실행 중...")
                output = net_connect.send_command(cmd, expect_string=prompt, read_timeout=200)
                all_output += f"\n--- {cmd} ---\n{output}\n"
        with open(RAW_DATA_FILE, "w", encoding="utf-8") as f:
            f.write(all_output)
        print("네트워크 데이터 수집 완료.")
        return True, None
    except Exception as e:
        error_message = f"네트워크 데이터 수집 중 오류: {e}"
        print(error_message)
        return False, error_message

def build_network_vector_store():
    """수집된 네트워크 데이터를 기반으로 FAISS 벡터 스토어를 구축합니다."""
    if not os.path.exists(RAW_DATA_FILE):
        return False, f"{RAW_DATA_FILE}이 없습니다."
    try:
        print("네트워크 벡터 스토어 구축 시작...")
        loader = TextLoader(RAW_DATA_FILE, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("네트워크 벡터 스토어 구축 완료.")
        return True, None
    except Exception as e:
        error_message = f"네트워크 벡터 스토어 구축 중 오류: {e}"
        print(error_message)
        return False, error_message