from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from transformer_lens import HookedTransformer

from demo.core import load_model
from demo.figure_utils import (
    create_all_circuit_pairwise_set_operation_svg,
    format_text,
    generate_circuit_multi_set_operation_svg,
    generate_individual_circuits,
    get_svg_path_from_config,
)
from demo.html_utils import create_svg_html_content

st.title("🔬 サーキット集合演算")
st.markdown("""
複数のサーキット間での集合演算（積集合・和集合・差集合）を実行し、可視化します。

- **Aggregation**：選択したサーキットの集合演算結果を1枚の画像として表示
- **Pairwise**：選択したサーキット同士の全ペアの集合演算結果をテーブル表示

このページのツールを用いた具体的な分析事例は「[📝 分析事例](./example)」ページの「分析事例2」で紹介しています。

サーキット図の見方については「[🎯 タスクの説明](./task)」ページの下部をご覧ください。
""")

st.warning(
    "各ページへの遷移やサーキット可視化ツールの設定は画面左のサイドバーから行なえます。"
)

st.divider()


def display_settings_info(config: dict[str, Any]) -> None:
    """設定情報を表示する共通関数."""
    if config["edge_selection_mode"] == "Top-N Edges":
        st.info(f"**Top-N**：{config['topn']} edges")
    elif config["score_threshold"] is not None:
        st.info(f"**性能の閾値**：{int(config['score_threshold'] * 100)}%")
        st.warning(
            "注意：エッジ数が多すぎる場合は自動的にトリムされます。(上限 3000 本)"
        )


def render_circuits_sidebar() -> dict[str, Any]:
    """集合演算専用のサイドバー."""
    # 利用可能な関係の取得
    from demo.figure_utils import get_available_relations

    relations = get_available_relations()

    circuit_display_mode = st.sidebar.radio(
        "表示モード：",
        ["Aggregation", "Pairwise"],
        help="Aggregation：選択したサーキットの集合演算結果を1枚の画像として表示． / Pairwise：選択したサーキット同士の全ペアの集合演算をテーブル表示．",
    )
    config: dict[str, Any] = {"circuit_display_mode": circuit_display_mode}

    # 共通：まず表示するサーキットを選択
    st.sidebar.markdown("##### 表示するサーキット：")
    selected_relations = [
        rel
        for rel in relations
        if st.sidebar.checkbox(format_text(rel), value=(rel in relations[:2]))
    ]
    config["selected_relations"] = selected_relations

    if circuit_display_mode == "Aggregation":
        # 選択されたサーキットの中からベースを選択
        if selected_relations:
            base_relation_display = st.sidebar.selectbox(
                "ベースサーキット:",
                [format_text(r) for r in selected_relations],
                help="集合演算のベースとなるサーキットを選択してください",
            )
            base_relation = selected_relations[
                [format_text(r) for r in selected_relations].index(
                    base_relation_display
                )
            ]
            other_relations = [
                rel for rel in selected_relations if rel != base_relation
            ]

            config["base_relation"] = base_relation
            config["other_relations"] = other_relations
        else:
            config["base_relation"] = None
            config["other_relations"] = []

    config["set_operation_mode"] = st.sidebar.selectbox(
        "集合演算：", ["intersection", "union", "difference"]
    )

    # 共通設定
    edge_selection_mode = st.sidebar.radio(
        "エッジ選択方法：", ["Top-N Edges", "Performance"]
    )

    config["edge_selection_mode"] = edge_selection_mode
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

    config["generate_button"] = st.sidebar.button(
        "サーキットを表示", use_container_width=True
    )

    return config


def display_multi_set_operation(
    model: HookedTransformer, config: dict[str, Any]
) -> None:
    """Aggregation 集合演算の表示."""
    selected_relations = config["selected_relations"]
    base_relation = config.get("base_relation")
    other_relations = config.get("other_relations", [])
    set_operation_mode = config["set_operation_mode"]

    if not selected_relations:
        st.warning("表示するサーキットを選択してください。")
        return

    if not base_relation:
        st.warning("ベースサーキットを選択してください。")
        return

    if config["generate_button"]:
        with st.spinner("サーキット生成中..."):
            generate_individual_circuits(model, selected_relations, config)

            # 集合演算SVG生成
            svg_path = get_svg_path_from_config(
                base_relation,
                config["edge_selection_mode"],
                config["topn"],
                config["score_threshold"],
                other_relations,
                set_operation_mode,
            )
            if not svg_path.exists():
                generate_circuit_multi_set_operation_svg(
                    model=model,
                    base_relation_name=base_relation,
                    other_relation_names=other_relations,
                    mode=set_operation_mode,
                    svg_path=svg_path,
                    topn=config["topn"],
                    score_threshold=config["score_threshold"],
                )

    display_settings_info(config)
    st.info(f"**集合演算**： {set_operation_mode}")
    st.info(f"**ベース**： {format_text(base_relation)}")
    for i, rel in enumerate(other_relations):
        st.info(f"**その他 {i + 1}**： {format_text(rel)}")

    # 集合演算結果表示
    multi_svg_path = get_svg_path_from_config(
        base_relation,
        config["edge_selection_mode"],
        config["topn"],
        config["score_threshold"],
        other_relations,
        set_operation_mode,
    )

    if multi_svg_path.exists():
        st.subheader(f"{set_operation_mode.title()} の結果")
        html_content = create_svg_html_content(
            multi_svg_path, max_height=config["max_height"]
        )
        components.html(html_content, height=config["max_height"] + 20, scrolling=False)

        st.subheader("演算前の各サーキット")
        for relation in selected_relations:
            st.markdown(f"**{format_text(relation)}**")
            svg_path = get_svg_path_from_config(
                relation,
                config["edge_selection_mode"],
                config["topn"],
                config["score_threshold"],
            )
            html_content = create_svg_html_content(
                svg_path, max_height=config["max_height"]
            )
            components.html(
                html_content, height=config["max_height"] + 20, scrolling=False
            )
    else:
        st.warning("「サーキットを表示」ボタンをクリックしてください。")


def display_pairwise_set_operation(
    model: HookedTransformer, config: dict[str, Any]
) -> None:
    """Pairwise の集合演算の表示."""
    selected_relations = config["selected_relations"]
    set_operation_mode = config["set_operation_mode"]

    if not selected_relations:
        st.warning("表示するサーキットを選択してください。")
        return

    if config["generate_button"]:
        with st.spinner("サーキット生成中..."):
            generate_individual_circuits(model, selected_relations, config)
            create_all_circuit_pairwise_set_operation_svg(
                model=model,
                relation_list=selected_relations,
                topn=config["topn"],
                score_threshold=config["score_threshold"],
                mode=set_operation_mode,
            )

    display_settings_info(config)
    st.info(f"**集合演算**： {set_operation_mode.title()}")
    for i, rel in enumerate(selected_relations):
        st.info(f"**サーキット {i + 1}**： {format_text(rel)}")

    # グリッド表示
    def get_pair_svg_path(base_rel: str, other_rel: str) -> Path:
        if base_rel == other_rel:
            return get_svg_path_from_config(
                base_rel,
                config["edge_selection_mode"],
                config["topn"],
                config["score_threshold"],
            )
        return get_svg_path_from_config(
            base_rel,
            config["edge_selection_mode"],
            config["topn"],
            config["score_threshold"],
            other_rel,
            set_operation_mode,
        )

    all_exist = all(
        get_pair_svg_path(base, other).exists()
        for base in selected_relations
        for other in selected_relations
    )

    if all_exist:
        num_cols = len(selected_relations)
        col_widths = [0.03] + [0.97 / num_cols] * num_cols

        # ヘッダー
        header_cols = st.columns(col_widths)
        header_cols[0].markdown("<b></b>", unsafe_allow_html=True)
        for j, rel in enumerate(selected_relations):
            header_cols[j + 1].markdown(
                f'<div style="text-align: center;"><b>{format_text(rel)}</b></div>',
                unsafe_allow_html=True,
            )

        # データ行
        for i, base_rel in enumerate(selected_relations):
            row_cols = st.columns(col_widths)
            row_cols[0].markdown(
                f'<div style="display: flex; align-items: center; justify-content: center; height: {config["max_height"] + 20}px;"><span style="transform: rotate(-90deg);"><b>{format_text(base_rel)}</b></span></div>',
                unsafe_allow_html=True,
            )
            for j, other_rel in enumerate(selected_relations):
                svg_path = get_pair_svg_path(base_rel, other_rel)
                with row_cols[j + 1]:
                    html_content = create_svg_html_content(
                        svg_path, max_height=config["max_height"]
                    )
                    components.html(
                        html_content, height=config["max_height"] + 20, scrolling=False
                    )
    else:
        st.warning("「サーキットを表示」ボタンをクリックしてください。")


def display_unified_circuits(model: HookedTransformer, config: dict[str, Any]) -> None:
    """統合されたサーキット表示."""
    mode = config["circuit_display_mode"]

    if mode == "Aggregation":
        display_multi_set_operation(model, config)
    elif mode == "Pairwise":
        display_pairwise_set_operation(model, config)
    else:
        st.error(f"不明なモード： {mode}")


# メイン実行
model = load_model()
config = render_circuits_sidebar()
display_unified_circuits(model, config)
