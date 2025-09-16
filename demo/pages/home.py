from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from transformer_lens import HookedTransformer

from demo.core import load_model
from demo.figure_utils import (
    format_text,
    generate_individual_circuits,
    get_available_relations,
    get_svg_path_from_config,
)
from demo.html_utils import create_svg_html_content


def display_settings_info(config: dict[str, Any]) -> None:
    """設定情報を表示する共通関数."""
    if config["edge_selection_mode"] == "Top-N Edges":
        st.info(f"**Top-N**：{config['topn']} edges")
    elif config["score_threshold"] is not None:
        st.info(f"**性能の閾値**：{int(config['score_threshold'] * 100)}%")
        st.warning(
            "注意：エッジ数が多すぎる場合は自動的にトリムされます。(上限 3000 本)"
        )


def display_svg_grid(relations: list[str], config: dict[str, Any]) -> None:
    """SVG を格子状に表示する関数."""

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    images_per_row = config.get("images_per_row", 1)

    for rel_chunk in chunk_list(relations, images_per_row):
        num_cols = len(rel_chunk)
        pad = images_per_row - num_cols
        padded_chunk = list(rel_chunk) + [None] * pad

        # ヘッダー行
        header_cols = st.columns(images_per_row)
        for i, relation in enumerate(padded_chunk):
            if relation:
                display_name = format_text(relation)
                header_cols[i].markdown(
                    f'<div style="text-align: center;"><b style="font-size: 1.3em;">{display_name}</b></div>',
                    unsafe_allow_html=True,
                )

        # SVG行
        svg_cols = st.columns(images_per_row)
        for i, relation in enumerate(padded_chunk):
            if relation:
                svg_path = get_svg_path_from_config(
                    relation,
                    config["edge_selection_mode"],
                    config["topn"],
                    config["score_threshold"],
                )
                html_content = create_svg_html_content(
                    svg_path, max_height=config["max_height"]
                )
                with svg_cols[i]:
                    components.html(
                        html_content, height=config["max_height"] + 20, scrolling=False
                    )


def render_circuits_sidebar() -> dict[str, Any]:
    """通常のサーキット表示用のサイドバー."""
    relations = get_available_relations()

    # サーキット選択
    st.sidebar.markdown("##### 表示するサーキット：")
    selected_relations = [
        rel
        for rel in relations
        if st.sidebar.checkbox(format_text(rel), value=(rel in relations[:2]))
    ]

    # エッジ選択方法
    edge_selection_mode = st.sidebar.radio(
        "エッジ選択方法：",
        ["Top-N Edges", "Performance"],
        help="Top-N Edges は N 本のエッジを選択し、Performance は性能が閾値を超えるまでエッジを追加します。",
    )

    config = {
        "selected_relations": selected_relations,
        "edge_selection_mode": edge_selection_mode,
    }

    config["topn"] = (
        st.sidebar.slider("エッジ数：", 100, 1000, 200, 100)
        if edge_selection_mode == "Top-N Edges"
        else 200
    )
    config["score_threshold"] = (
        st.sidebar.slider(
            "閾値 (%)：",
            0,
            100,
            50,
            10,
            help="サーキットに求める Exact Match の正答率を設定します。",
        )
        / 100.0
        if edge_selection_mode == "Performance"
        else None
    )
    config["max_height"] = st.sidebar.slider(
        "画像の高さ (px):",
        400,
        1200,
        600,
        50,
        help="表示するサーキット画像の高さを調整します。",
    )
    config["images_per_row"] = st.sidebar.slider(
        "1行あたりの画像数：", 1, len(relations), 1
    )

    config["generate_button"] = st.sidebar.button(
        "サーキットを表示", use_container_width=True
    )

    return config


def display_normal_circuits(model: HookedTransformer, config: dict[str, Any]) -> None:
    """通常のサーキット表示."""
    selected_relations = config["selected_relations"]

    if not selected_relations:
        st.warning("表示するサーキットを選択してください。")
        return

    if config["generate_button"]:
        with st.spinner("サーキット生成中..."):
            generate_individual_circuits(model, selected_relations, config)

    display_settings_info(config)

    # SVG 存在チェック
    all_exist = all(
        get_svg_path_from_config(
            rel,
            config["edge_selection_mode"],
            config["topn"],
            config["score_threshold"],
        ).exists()
        for rel in selected_relations
    )

    if all_exist:
        display_svg_grid(selected_relations, config)
    else:
        st.warning("「サーキットを表示」ボタンをクリックしてください。")


st.title("🏠️ ホーム")
st.markdown("""
### Circuit Visualize Demo へようこそ！

本デモでは、Transformer 型言語モデル ([GPT-2 Small](https://huggingface.co/openai-community/gpt2)) における「サーキット」の可視化を行ないます。

サーキットについての説明は「[ℹ️ サーキットとは？](./explain_circuit)」ページをご覧ください。

また、サーキットの可視化はタスク内容と密接に関連しているため、「[🎯 タスクの説明](./task)」ページもあわせてご覧ください。

以下では、基本的なサーキット表示機能を提供します。より高度な集合演算機能については「[🔬 サーキット集合演算](./set_operation)」ページをご利用ください。

本ページや「[🔬 サーキット集合演算](./set_operation)」ページで提供するサーキット可視化ツールを使用した具体的な分析事例については「[📝 分析事例](./example)」ページをご覧ください。
""")

st.warning(
    "各ページへの遷移やサーキット可視化ツールの設定は画面左のサイドバーから行なえます。"
)

st.divider()

# メイン実行
model = load_model()
config = render_circuits_sidebar()
display_normal_circuits(model, config)

st.divider()

st.markdown("""
本デモは [YANS2025](https://yans.anlp.jp/entry/yans2025) の [S2-P03]「言語モデルの内部機序分析に向けたサーキットの可視化」
深田 圭一 (東北大), Heinzerling Benjamin (理研/東北大), 坂口 慶祐 (東北大/理研) のものです。
本デモのソースコードは [GitHub リポジトリ](https://github.com/Fukata-K/Circuit-Visualize-YANS2025) で公開されています。
""")
