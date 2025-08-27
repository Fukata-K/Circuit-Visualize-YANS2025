from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from transformer_lens import HookedTransformer

from demo.figure_utils import (
    generate_circuit_multi_set_operation_svg,
    generate_circuit_svg,
    get_svg_path,
)
from demo.html_utils import create_svg_html_content


def display_circuit_multi_set_operation(
    model: HookedTransformer,
    device: torch.device,
    mode: str = "intersection",
) -> None:
    """
    Circuit の各種集合演算の表示を行う関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        device (torch.device): 使用するデバイス (例: "cpu" or "cuda").
        mode (str): 集合演算の種類. "intersection", "union", "difference", "weighted_difference" から選択.

    Returns:
        None
    """

    # 利用可能な Relation name を取得
    data_dir = Path("data/filtered_gpt2_small")
    available_relations = [f.stem for f in data_dir.glob("**/*.csv")]
    available_relations.sort()

    # Base Relation を 1つ選択
    st.sidebar.header("Settings")
    default_base = (
        "landmark_in_country"
        if "landmark_in_country" in available_relations
        else available_relations[0]
    )
    base_relation = st.sidebar.selectbox(
        "Base Relation:",
        options=available_relations,
        index=available_relations.index(default_base)
        if default_base in available_relations
        else 0,
    )

    # Other Relations を複数選択
    st.sidebar.subheader("Other Relations")
    default_others = [
        r
        for r in ["landmark_on_continent"]
        if r in available_relations and r != base_relation
    ]
    other_relations = []
    for rel in available_relations:
        if rel != base_relation:  # Base Relation 以外のみ表示
            checked = st.sidebar.checkbox(rel, value=(rel in default_others))
            if checked:
                other_relations.append(rel)

    if not other_relations:
        st.warning("Please select at least one other relation.")
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
    else:  # Performance Threshold
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

    # ヘルパー関数定義 (必要な変数が定義された後)
    def get_individual_svg_path(relation_name):
        """個別の Circuit SVGパス を取得"""
        return get_svg_path(
            base_relation=relation_name,
            topn=topn if edge_selection_mode == "Top-N Edges" else None,
            perf_percent=score_threshold
            if edge_selection_mode == "Performance Threshold"
            else None,
        )

    def get_multi_operation_svg_path():
        """複数の論理演算結果の SVG パスを取得"""
        return get_svg_path(
            base_relation=base_relation,
            other_relation=other_relations,
            set_operation_mode=mode,
            topn=topn if edge_selection_mode == "Top-N Edges" else None,
            perf_percent=score_threshold
            if edge_selection_mode == "Performance Threshold"
            else None,
        )

    # 描画実行ボタン
    generate_button = st.sidebar.button("Generate Set Operations")

    # ボタンが押されたときのみ描画処理を実行
    if generate_button:
        with st.spinner("Generating set operation circuits..."):
            # まず Base Relation の個別 Circuit を生成
            svg_path = get_individual_svg_path(base_relation)

            if not svg_path.exists():
                generate_circuit_svg(
                    model=model,
                    device=device,
                    relation_name=base_relation,
                    svg_path=svg_path,
                    topn=topn,
                    score_threshold=score_threshold,
                )

            # Other Relations の個別 Circuit を生成
            for other_relation in other_relations:
                svg_path = get_individual_svg_path(other_relation)

                if not svg_path.exists():
                    generate_circuit_svg(
                        model=model,
                        device=device,
                        relation_name=other_relation,
                        svg_path=svg_path,
                        topn=topn,
                        score_threshold=score_threshold,
                    )

            # Base Relation と全ての Other Relations の集合演算の SVG ファイルを生成
            svg_path = get_multi_operation_svg_path()
            if not svg_path.exists():
                generate_circuit_multi_set_operation_svg(
                    model=model,
                    device=device,
                    base_relation_name=base_relation,
                    other_relation_names=other_relations,
                    mode=mode,
                    svg_path=svg_path,
                    topn=topn,
                    score_threshold=score_threshold,
                )

    # 選択された設定を表示
    if edge_selection_mode == "Top-N Edges":
        st.info(f"**Top-N**: {topn} edges")
    else:
        st.info(f"**Performance Threshold**: {score_threshold:.2f}")
    st.info(f"**Base Relation**: {base_relation}")
    for i, relation in enumerate(other_relations):
        st.info(f"**Other Relation {i + 1}**: {relation}")

    # 全ての SVG ファイルが存在するかチェック
    def check_all_svg_exist():
        # Base Relation の個別 Circuit をチェック
        svg_path = get_individual_svg_path(base_relation)
        if not svg_path.exists():
            return False

        # Other Relations の個別 Circuit をチェック
        for other_relation in other_relations:
            svg_path = get_individual_svg_path(other_relation)
            if not svg_path.exists():
                return False

        # Base Relation と全ての Other Relations の集合演算をチェック
        svg_path = get_multi_operation_svg_path()
        if not svg_path.exists():
            return False

        return True

    if check_all_svg_exist():
        # 集合演算結果を表示
        st.subheader(f"{mode.title()} Result")
        other_relations_str = " & ".join(other_relations)
        st.markdown(f"**{base_relation} {mode} ({other_relations_str})**")
        svg_path = get_multi_operation_svg_path()
        html_content = create_svg_html_content(svg_path, max_height=max_height)
        components.html(html_content, height=max_height + 20, scrolling=False)

        # Base Relation の個別 Circuit を表示
        st.subheader("Base Relation Circuit")
        st.markdown(f"**{base_relation}**")
        svg_path = get_individual_svg_path(base_relation)
        html_content = create_svg_html_content(svg_path, max_height=max_height)
        components.html(html_content, height=max_height + 20, scrolling=False)

        # Other Relations の個別 Circuit を表示
        st.subheader("Other Relations Circuits")
        for other_relation in other_relations:
            st.markdown(f"**{other_relation}**")
            svg_path = get_individual_svg_path(other_relation)
            html_content = create_svg_html_content(svg_path, max_height=max_height)
            components.html(html_content, height=max_height + 20, scrolling=False)
    else:
        st.warning(
            "Click 'Generate Set Operations' button to create the visualizations."
        )
