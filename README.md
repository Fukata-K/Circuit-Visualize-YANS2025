# Circuit-Visualize-YANS2025

## 🌐 デモサイト

以下の URL からデモをブラウザ上で直接体験できます：

🔗 [https://circuit-visualize-yans2025-xxszotqmqkoetu3jwnt3gk.streamlit.app/](https://circuit-visualize-yans2025-xxszotqmqkoetu3jwnt3gk.streamlit.app/)


## macOS でのインストールと起動 （uv + Homebrew）

```bash
brew install graphviz

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

streamlit run streamlit_app.py
```

## pygraphviz インストール失敗時の対処 （macOS, uv）

```bash
uv pip uninstall pygraphviz
uv pip install pip

GV=$(brew --prefix graphviz)
export CPATH="$GV/include"
export LIBRARY_PATH="$GV/lib"
export PKG_CONFIG_PATH="$GV/lib/pkgconfig"

python -m pip install --no-binary pygraphviz \
  --config-settings="--global-option=build_ext" \
  --config-settings="--global-option=-I${GV}/include" \
  --config-settings="--global-option=-L${GV}/lib" \
  --config-settings="--global-option=-R${GV}/lib" \
  pygraphviz
```
