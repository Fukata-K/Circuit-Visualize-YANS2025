import streamlit as st
import torch
from transformer_lens import HookedTransformer

from demo.multi_circuits import display_multi_circuits
from demo.multi_set_operation_circuits import display_circuit_multi_set_operation
from demo.pairwise_set_operation_circuits import display_circuit_pairwise_set_operation
from demo.single_circuit import display_single_circuit


@st.cache_resource
def load_model(model_name="gpt2-small", device=torch.device("cpu")):
    print(f"Using device: {device}")
    return HookedTransformer.from_pretrained(model_name, device=device)


def set_page(page_name):
    st.session_state.page = page_name


# Streamlit UI 設定
st.set_page_config(layout="wide")
st.title("Circuit Demo")
if "page" not in st.session_state:
    st.session_state.page = "home"

# モデルの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-small"
model = load_model(model_name=model_name, device=device)

# ページ切り替えボタンと判定
if st.session_state.page == "home":
    if st.button("Single Graph"):
        set_page("single")
        st.rerun()

    if st.button("Multi Graphs"):
        set_page("multi")
        st.rerun()

    if st.button("Pairwise Union"):
        set_page("pairwise_union")
        st.rerun()

    if st.button("Pairwise Intersection"):
        set_page("pairwise_intersection")
        st.rerun()

    if st.button("Pairwise Difference"):
        set_page("pairwise_difference")
        st.rerun()

    # 開発中
    # if st.button("Weighted Difference"):
    #     set_page("weighted_difference")
    #     st.rerun()

    if st.button("Multi Union"):
        set_page("multi_union")
        st.rerun()

    if st.button("Multi Intersection"):
        set_page("multi_intersection")
        st.rerun()

    if st.button("Multi Difference"):
        set_page("multi_difference")
        st.rerun()

elif st.session_state.page == "single":
    st.header("Single Graph Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_single_circuit(model)

elif st.session_state.page == "multi":
    st.header("Multi Graphs Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_multi_circuits(model)

elif st.session_state.page == "pairwise_union":
    st.header("Pairwise Union Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_pairwise_set_operation(model, mode="union")

elif st.session_state.page == "pairwise_intersection":
    st.header("Pairwise Intersection Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_pairwise_set_operation(model, mode="intersection")

elif st.session_state.page == "pairwise_difference":
    st.header("Pairwise Difference Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_pairwise_set_operation(model, mode="difference")

# 開発中
# elif st.session_state.page == "weighted_difference":
#     st.header("Weighted Difference Page")
#     if st.button("Back to Home"):
#         set_page("home")
#         st.rerun()
#     display_circuit_pairwise_set_operation(model, mode="weighted_difference")

elif st.session_state.page == "multi_union":
    st.header("Multi Union Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_multi_set_operation(model, mode="union")

elif st.session_state.page == "multi_intersection":
    st.header("Multi Intersection Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_multi_set_operation(model, mode="intersection")

elif st.session_state.page == "multi_difference":
    st.header("Multi Difference Page")
    if st.button("Back to Home"):
        set_page("home")
        st.rerun()
    display_circuit_multi_set_operation(model, mode="difference")
