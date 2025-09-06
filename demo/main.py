import streamlit as st
import torch
from transformer_lens import HookedTransformer

from demo.home import display_home, display_sidebar_navigation, set_page
from demo.multi_circuits import display_multi_circuits
from demo.multi_set_operation_circuits import display_circuit_multi_set_operation
from demo.pairwise_set_operation_circuits import display_circuit_pairwise_set_operation
from demo.single_circuit import display_single_circuit


@st.cache_resource
def load_model(model_name="gpt2-small", device=torch.device("cpu")):
    print(f"Using device: {device}")
    return HookedTransformer.from_pretrained(model_name, device=device)


# Streamlit UI 設定
st.set_page_config(page_title="Circuit Visualize Demo", layout="wide")
st.title("Circuit Visualize Demo (YANS2025)")
if "page" not in st.session_state:
    st.session_state.page = "home"

# モデルの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-small"
model = load_model(model_name=model_name, device=device)

# ページ切り替えボタンと判定
if st.session_state.page == "home":
    display_home()

elif st.session_state.page == "single":
    st.header("Single Circuit Page")
    display_single_circuit(model)

elif st.session_state.page == "multi":
    st.header("Multi Circuits Page")
    display_multi_circuits(model)

elif st.session_state.page == "pairwise_union":
    st.header("Pairwise Union Page")
    display_circuit_pairwise_set_operation(model, mode="union")

elif st.session_state.page == "pairwise_intersection":
    st.header("Pairwise Intersection Page")
    display_circuit_pairwise_set_operation(model, mode="intersection")

elif st.session_state.page == "pairwise_difference":
    st.header("Pairwise Difference Page")
    display_circuit_pairwise_set_operation(model, mode="difference")

# 開発中
# elif st.session_state.page == "weighted_difference":
#     st.header("Weighted Difference Page")
#     display_circuit_pairwise_set_operation(model, mode="weighted_difference")

elif st.session_state.page == "multi_union":
    st.header("Multi Union Page")
    display_circuit_multi_set_operation(model, mode="union")

elif st.session_state.page == "multi_intersection":
    st.header("Multi Intersection Page")
    display_circuit_multi_set_operation(model, mode="intersection")

elif st.session_state.page == "multi_difference":
    st.header("Multi Difference Page")
    display_circuit_multi_set_operation(model, mode="difference")

# サイドバーにページナビゲーションを表示
display_sidebar_navigation()
