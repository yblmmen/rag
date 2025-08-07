# setup_components.py
import os

def setup_terminal_component():
    """components/terminal/ 디렉토리와 파일을 자동 생성"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    terminal_dir = os.path.join(base_dir, "components", "terminal")
    frontend_dir = os.path.join(terminal_dir, "frontend")
    build_dir = os.path.join(frontend_dir, "build")

    # 디렉토리 생성
    os.makedirs(terminal_dir, exist_ok=True)
    os.makedirs(frontend_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)

    # __init__.py
    init_content = """\
import os
import streamlit.components.v1 as components

_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_PARENT_DIR, "frontend", "build")

_terminal_component = components.declare_component(
    "terminal_component",
    path=_BUILD_DIR
)

def terminal_component(websocket_url, host, username, password, port, key=None):
    return _terminal_component(
        websocket_url=websocket_url,
        host=host,
        username=username,
        password=password,
        port=port,
        key=key
    )
"""
    with open(os.path.join(terminal_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(init_content)

    # package.json
    package_json_content = """\
{
    "name": "terminal-component",
    "version": "0.1.0",
    "dependencies": {
        "xterm": "^5.3.0",
        "xterm-addon-fit": "^0.8.0",
        "socket.io-client": "^4.7.5"
    },
    "scripts": {
        "build": "mkdir -p build/static && cp index.html build/ && cp index.js build/static && cp -r node_modules build/static"
    }
}
"""
    with open(os.path.join(frontend_dir, "package.json"), "w", encoding="utf-8") as f:
        f.write(package_json_content)

    # index.html
    index_html_content = """\
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="static/node_modules/xterm/css/xterm.css" />
</head>
<body>
    <div id="terminal"></div>
    <script src="static/node_modules/xterm/lib/xterm.js"></script>
    <script src="static/node_modules/xterm-addon-fit/lib/xterm-addon-fit.js"></script>
    <script src="static/node_modules/socket.io-client/dist/socket.io.js"></script>
    <script src="static/index.js"></script>
</body>
</html>
"""
    with open(os.path.join(frontend_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html_content)

    # index.js
    index_js_content = """\
const terminal = new Terminal({
    cursorBlink: true,
    theme: { background: '#1e1e1e' }
});
const fitAddon = new FitAddon.FitAddon();
terminal.loadAddon(fitAddon);

terminal.open(document.getElementById('terminal'));
fitAddon.fit();

const socket = io(Streamlit.connectionInfo.componentProps.websocket_url);

socket.on('connect', () => {
    socket.emit('start_terminal', {
        host: Streamlit.connectionInfo.componentProps.host,
        username: Streamlit.connectionInfo.componentProps.username,
        password: Streamlit.connectionInfo.componentProps.password,
        port: Streamlit.connectionInfo.componentProps.port
    });
});

socket.on('terminal_output', (data) => {
    terminal.write(data.data);
});

terminal.onData((data) => {
    socket.emit('terminal_input', { data: data });
});

Streamlit.setFrameHeight(400);
"""
    with open(os.path.join(frontend_dir, "index.js"), "w", encoding="utf-8") as f:
        f.write(index_js_content)

if __name__ == "__main__":
    setup_terminal_component()