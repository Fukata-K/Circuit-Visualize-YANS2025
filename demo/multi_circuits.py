from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from transformer_lens import HookedTransformer

from demo.figure_utils import generate_circuit_svg, get_svg_path
from demo.html_utils import create_svg_html_content


def display_multi_circuits(model: HookedTransformer) -> None:
    """
    複数の Circuit を並べて表示する関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.

    Returns:
        None
    """
    # 利用可能な Relation name を取得
    data_dir = Path("data/filtered_gpt2_small")
    available_relations = [f.stem for f in data_dir.glob("**/*.csv")]
    available_relations.sort()

    # チェックボックスで Relation name を選択 (順序固定)
    st.sidebar.header("Settings")
    default_checked = set(available_relations[:2])
    selected_relations = []
    for rel in available_relations:
        checked = st.sidebar.checkbox(rel, value=(rel in default_checked))
        if checked:
            selected_relations.append(rel)
    if not selected_relations:
        st.warning("Please select at least one relation.")
        return

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

    # 1行あたりの画像数スライダー
    max_cols = len(available_relations) if len(available_relations) > 0 else 1
    images_per_row = st.sidebar.slider(
        "Images per Row:",
        min_value=1,
        max_value=max_cols,
        value=min(2, max_cols),
        step=1,
    )
    # SVG 最大高さ設定
    max_height = st.sidebar.slider(
        "SVG Max Height (px):", min_value=400, max_value=1200, value=800, step=50
    )

    # 描画実行ボタン
    generate_button = st.sidebar.button("Generate Circuits")

    # ボタンが押されたときのみ描画処理を実行
    if generate_button:
        with st.spinner("Generating Circuits..."):
            for relation in selected_relations:
                svg_path = get_svg_path(
                    base_relation=relation,
                    topn=topn if edge_selection_mode == "Top-N Edges" else None,
                    perf_percent=score_threshold
                    if edge_selection_mode == "Performance Threshold"
                    else None,
                )

                if not svg_path.exists():
                    generate_circuit_svg(
                        model=model,
                        relation_name=relation,
                        svg_path=svg_path,
                        topn=topn,
                        score_threshold=score_threshold,
                    )

    # 選択された設定を表示
    if edge_selection_mode == "Top-N Edges":
        st.info(f"**Top-N**: {topn} edges")
    else:
        st.info(f"**Performance Threshold**: {score_threshold:.2f}")
    for i, relation in enumerate(selected_relations):
        st.info(f"**Relation {i + 1}**: {relation}")

    # 複数行でラベルと SVG 画像を表示
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # 全ての SVG ファイルが存在するかチェック
    def get_relation_svg_path(relation):
        return get_svg_path(
            base_relation=relation,
            topn=topn if edge_selection_mode == "Top-N Edges" else None,
            perf_percent=score_threshold
            if edge_selection_mode == "Performance Threshold"
            else None,
        )

    all_svg_exist = all(
        get_relation_svg_path(relation).exists() for relation in selected_relations
    )

    if all_svg_exist:
        for rel_chunk in chunk_list(selected_relations, images_per_row):
            num_cols = len(rel_chunk)
            # ダミーでカラム数を揃える
            pad = images_per_row - num_cols
            padded_rel_chunk = list(rel_chunk) + [None] * pad
            # ラベル行
            header_cols = st.columns(images_per_row)
            for i, relation in enumerate(padded_rel_chunk):
                if relation is not None:
                    header_cols[i].markdown(
                        f'<div style="text-align: center;"><b style="font-size: 1.3em;">{relation}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    header_cols[i].markdown("", unsafe_allow_html=True)
            # SVG 画像行
            svg_cols = st.columns(images_per_row)
            for i, base_relation in enumerate(padded_rel_chunk):
                if base_relation is not None:
                    svg_path = get_relation_svg_path(base_relation)
                    html_content = create_svg_html_content(
                        svg_path, max_height=max_height
                    )
                    with svg_cols[i]:
                        components.html(
                            html_content, height=max_height + 20, scrolling=False
                        )
                else:
                    with svg_cols[i]:
                        st.markdown("")
    else:
        st.warning("Click 'Generate Circuits' button to create the visualizations.")
