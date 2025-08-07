# run_all.py
import os
import subprocess
import time
import sys
from setup_components import setup_terminal_component

def ensure_node_modules():
    """Node.js 의존성 설치 및 빌드"""
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "terminal", "frontend")
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    print("Building frontend...")
    subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)

def main():
    """WebSocket 서버와 Streamlit 앱을 동시에 실행"""
    # components/terminal/ 디렉토리 생성
    print("Setting up terminal component...")
    setup_terminal_component()

    # Node.js 의존성 설치 및 빌드
    ensure_node_modules()

    # WebSocket 서버 실행 (백그라운드)
    print("Starting WebSocket server...")
    websocket_process = subprocess.Popen(
        [sys.executable, "websocket_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    # Streamlit 앱 실행 (포그라운드)
    print("Starting Streamlit app...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    # 프로세스 상태 모니터링
    try:
        # 두 프로세스가 모두 실행될 때까지 잠시 대기
        time.sleep(5)

        while True:
            if streamlit_process.poll() is not None:
                print("Streamlit app stopped.")
                break
            if websocket_process.poll() is not None:
                print("WebSocket server stopped.")
                # 웹소켓 서버가 중지되어도 Streamlit은 계속 실행되도록 break 제거
            time.sleep(1)

    except KeyboardInterrupt:
        print("Terminating processes...")
        websocket_process.terminate()
        streamlit_process.terminate()
        websocket_process.wait()
        streamlit_process.wait()
        print("All processes terminated.")

if __name__ == "__main__":
    main()