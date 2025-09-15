import streamlit as st

st.set_page_config(page_title="Circuit Visualize Demo", layout="wide")

home = st.Page("demo/pages/home.py", title="ホーム", icon="🏠")
explain = st.Page("demo/pages/explain_circuit.py", title="サーキットとは？", icon="ℹ️")
circuit = st.Page("demo/pages/set_operation.py", title="サーキット集合演算", icon="🔬")
task = st.Page("demo/pages/task.py", title="タスクの説明", icon="🎯")
example = st.Page("demo/pages/example.py", title="分析事例", icon="📝")

nav = st.navigation([home, explain, circuit, task, example])
nav.run()
