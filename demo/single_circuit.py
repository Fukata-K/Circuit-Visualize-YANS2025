from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from transformer_lens import HookedTransformer

from demo.figure_utils import generate_circuit_svg, get_svg_path
from demo.html_utils import create_svg_html_content


def display_single_circuit(model: HookedTransformer) -> None:
    """
    単独の Circuit の表示を行う関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.

    Returns:
        None
    """
    # 利用可能な Relation name を取得
    data_dir = Path("data/filtered_gpt2_small")
    available_relations = [f.stem for f in data_dir.glob("**/*.csv")]
    available_relations.sort()

    # プルダウンメニューで Relation name を選択
    st.sidebar.header("Settings")
    relation = st.sidebar.selectbox("Relation:", available_relations, index=0)

    # エッジ選択方法の設定
    edge_selection_mode = st.sidebar.radio(
        "Edge Selection Mode:", ["Top-N Edges", "Performance Threshold"], index=0
    )

    # 選択されたモードに応じてパラメータを設定
    if edge_selection_mode == "Top-N Edges":
        topn = st.sidebar.slider(
            "Top-N Edges:", min_value=100, max_value=500, value=200, step=100
        )
        score_threshold = None
    else:
        score_threshold = st.sidebar.slider(
            "Performance Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            format="%.2f",
        )
        topn = 200  # フォールバック用のデフォルト値

    # SVG 最大高さ設定
    max_height = st.sidebar.slider(
        "SVG Max Height (px):", min_value=400, max_value=1200, value=800, step=50
    )

    # 描画実行ボタン
    generate_button = st.sidebar.button("Generate Circuit")

    # 現在の設定をキーとして SVG ファイル名を生成
    svg_path = get_svg_path(
        base_relation=relation,
        topn=topn if edge_selection_mode == "Top-N Edges" else None,
        perf_percent=score_threshold
        if edge_selection_mode == "Performance Threshold"
        else None,
    )

    # ボタンが押されたときのみ描画処理を実行
    if generate_button:
        # SVG ファイルが既に存在するかチェック
        if not svg_path.exists():
            with st.spinner("Generating Circuit..."):
                generate_circuit_svg(
                    model=model,
                    relation_name=relation,
                    svg_path=svg_path,
                    topn=topn,
                    score_threshold=score_threshold,
                )

    # 選択された設定を表示
    st.info(f"**Relation**: {relation}")
    if edge_selection_mode == "Top-N Edges":
        st.info(f"**Top-N**: {topn} edges")
    else:
        st.info(f"**Performance Threshold**: {score_threshold:.2f}")

    # SVG ファイルが存在する場合のみ表示
    if svg_path.exists():
        # HTML 表示
        html_content = create_svg_html_content(svg_path, max_height=max_height)
        components.html(html_content, height=max_height + 20, scrolling=False)
    else:
        st.warning("Click 'Generate Circuit' button to create the visualization.")
