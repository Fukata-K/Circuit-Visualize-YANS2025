import streamlit as st
import torch
from transformer_lens import HookedTransformer


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model(model_name: str = "gpt2-small", device: torch.device | None = None):
    device = device or pick_device()
    print(f"Using device: {device}")
    return HookedTransformer.from_pretrained(model_name, device=device)
