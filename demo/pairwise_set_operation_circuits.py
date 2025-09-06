from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from transformer_lens import HookedTransformer

from demo.figure_utils import (
    create_all_circuit_pairwise_set_operation_svg,
    generate_circuit_svg,
    get_svg_path,
)
from demo.html_utils import create_svg_html_content


def display_circuit_pairwise_set_operation(
    model: HookedTransformer,
    mode: str = "intersection",
) -> None:
    """
    Circuit の各種集合演算の表示を行う関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        mode (str): 集合演算の種類. "intersection", "union", "difference", "weighted_difference" から選択.

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

    # SVG 最大高さ設定
    max_height = st.sidebar.slider(
        "SVG Max Height (px):", min_value=400, max_value=1200, value=800, step=50
    )

    # ヘルパー関数定義 (必要な変数が定義された後)
    def get_individual_svg_path(relation_name):
        """個別の Circuit SVG パスを取得"""
        return get_svg_path(
            base_relation=relation_name,
            topn=topn if edge_selection_mode == "Top-N Edges" else None,
            perf_percent=score_threshold
            if edge_selection_mode == "Performance Threshold"
            else None,
        )

    def get_relation_svg_path(base_relation, other_relation=None):
        """ペアごとの SVG パスを取得"""
        if base_relation == other_relation or other_relation is None:
            # 対角線の場合：個別 Circuit
            return get_individual_svg_path(base_relation)
        else:
            # 集合演算の場合
            return get_svg_path(
                base_relation=base_relation,
                other_relation=other_relation,
                set_operation_mode=mode,
                topn=topn if edge_selection_mode == "Top-N Edges" else None,
                perf_percent=score_threshold
                if edge_selection_mode == "Performance Threshold"
                else None,
            )

    # 描画実行ボタン
    generate_button = st.sidebar.button("Generate Set Circuits", use_container_width=True)

    # ボタンが押されたときのみ描画処理を実行
    if generate_button:
        with st.spinner("Generating set operation Circuits..."):
            # まず対角線用の個別 Circuit を生成
            for relation in selected_relations:
                svg_path = get_individual_svg_path(relation)

                if not svg_path.exists():
                    generate_circuit_svg(
                        model=model,
                        relation_name=relation,
                        svg_path=svg_path,
                        topn=topn,
                        score_threshold=score_threshold,
                    )

            # 全てのペアについて集合演算の SVG ファイルを生成
            create_all_circuit_pairwise_set_operation_svg(
                model=model,
                relation_list=selected_relations,
                topn=topn,
                score_threshold=score_threshold,
                mode=mode,
            )

    # 選択された設定を表示
    if edge_selection_mode == "Top-N Edges":
        st.info(f"**Top-N**: {topn} edges")
    else:
        st.info(f"**Performance Threshold**: {score_threshold:.2f}")
    for i, relation in enumerate(selected_relations):
        st.info(f"**Relation {i + 1}**: {relation}")

    # 全ての SVG ファイルが存在するかチェック
    def check_all_svg_exist():
        for base_relation in selected_relations:
            for other_relation in selected_relations:
                svg_path = get_relation_svg_path(base_relation, other_relation)
                if not svg_path.exists():
                    return False
        return True

    if check_all_svg_exist():
        # グリッド表示: base_relation を縦, other_relation を横
        num_cols = len(selected_relations)
        # 1行目: 空セル + other_relation ラベル
        col_widths = [0.03] + [0.97 / num_cols] * num_cols  # 1列目 3%, 残り均等
        header_cols = st.columns(col_widths)
        header_cols[0].markdown("<b></b>", unsafe_allow_html=True)
        for j, other_relation in enumerate(selected_relations):
            header_cols[j + 1].markdown(
                f'<div style="text-align: center;"><b style="font-size: 1.3em;">{other_relation}</b></div>',
                unsafe_allow_html=True,
            )

        # 2行目以降: base_relation ラベル + SVG グリッド
        for i, base_relation in enumerate(selected_relations):
            # 1列目だけ幅を狭くする
            col_widths = [0.03] + [0.97 / num_cols] * num_cols  # 1列目 3%, 残り均等
            row_cols = st.columns(col_widths)
            # 行ラベル (縦書き・90 度回転・狭い幅)
            row_cols[0].markdown(
                f'<div style="display: flex; align-items: center; justify-content: center; height: {max_height + 20}px;"><span style="transform: rotate(-90deg); white-space: nowrap; min-width: 20px; max-width: 30px; text-align: left; display: inline-block;"><b style="font-size: 1.3em;">{base_relation}</b></span></div>',
                unsafe_allow_html=True,
            )
            for j, other_relation in enumerate(selected_relations):
                svg_path = get_relation_svg_path(base_relation, other_relation)
                with row_cols[j + 1]:
                    html_content = create_svg_html_content(
                        svg_path, max_height=max_height
                    )
                    components.html(
                        html_content, height=max_height + 20, scrolling=False
                    )
    else:
        st.warning(
            "Click 'Generate Set Operations' button to create the visualizations."
        )
