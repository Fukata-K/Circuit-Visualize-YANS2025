from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from transformer_lens import ActivationCache, HookedTransformer

from analysis.analysis_logits import load_logit_analysis_results
from analysis.head_scoring import get_head_scores
from circuit.circuit_utils import Circuit
from visual_style import (
    BLUE,
    BORDER,
    EDGE_WIDTH,
    FONTSIZE,
    GRAY,
    GREEN,
    NODE,
    NODE_BORDER_WIDTH,
    NODE_HEIGHT,
    NODE_WIDTH,
    RBMIX,
    RED,
    Color,
)


def get_color_from_score(score_red: float, score_blue: float) -> str:
    """
    スコアに基づいて色を計算するヘルパー関数.

    Args:
        score_red  (float): 赤のスコア (0.0 から 1.0)
        score_blue (float): 青のスコア (0.0 から 1.0)

    Returns:
        str: 16進カラーコード (例: "#A1B2C3")
    """
    s = score_red
    r = score_blue
    R = (RBMIX.r * s + BLUE.r * (1 - s)) * r + (RED.r * s + NODE.r * (1 - s)) * (1 - r)
    G = (RBMIX.g * s + BLUE.g * (1 - s)) * r + (RED.g * s + NODE.g * (1 - s)) * (1 - r)
    B = (RBMIX.b * s + BLUE.b * (1 - s)) * r + (RED.b * s + NODE.b * (1 - s)) * (1 - r)
    return Color(R, G, B).to_hex()


def get_color_from_logits_rank(rank: float, max_rank: float) -> str:
    """
    target の rank に基づいて色を計算するヘルパー関数.
    順位が良い (数値が小さい) ほど緑色, 悪い (数値が大きい) ほどデフォルト色になる.

    Args:
        rank (float): target のランク
        max_rank (float): 最大ランク

    Returns:
        str: 16進カラーコード (例: "#A1B2C3")
    """
    log_rank = np.log(rank + 1e-8)
    log_max_rank = np.log(max_rank + 1e-8)
    normalized_log_rank = log_rank / log_max_rank if log_max_rank > 0 else 0.0

    # 順位が良いほど緑, 悪いほどデフォルト色になるように調整
    R = BORDER.r * normalized_log_rank + GREEN.r * (1 - normalized_log_rank)
    G = BORDER.g * normalized_log_rank + GREEN.g * (1 - normalized_log_rank)
    B = BORDER.b * normalized_log_rank + GREEN.b * (1 - normalized_log_rank)
    return Color(R, G, B).to_hex()


def make_node_color_dict(
    circuit: Circuit,
    red_scores: dict[str, float | None],
    blue_scores: dict[str, float | None],
    input_color: str = GRAY.to_hex(),
    mlp_color: str = NODE.to_hex(),
    output_color: str = GRAY.to_hex(),
    default_color: str = NODE.to_hex(),
) -> dict[str, str]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> 色コード の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        red_scores  (dict[str, float]): Attention Head 名をキー, 赤スコアを値とする辞書
        blue_scores (dict[str, float]): Attention Head 名をキー, 青スコアを値とする辞書
        input_color   (str): 入力ノードの色
        mlp_color     (str): MLP ノードの色
        output_color  (str): 出力ノードの色
        default_color (str): その他ノードの色

    Returns:
        dict[str, str]: ノード名をキー, 16進カラーコードを値とする辞書
    """
    colors: dict[str, str] = {}
    for node in circuit.nodes.values():
        if node.name == "input":  # Input
            colors[node.name] = input_color
        elif node.name.startswith("a"):  # Attention Head
            colors[node.name] = get_color_from_score(
                red_scores.get(node.name, 0.0) or 0.0,
                blue_scores.get(node.name, 0.0) or 0.0,
            )
        elif node.name.startswith("m"):  # MLP
            colors[node.name] = mlp_color
        elif node.name == "logits":  # Output
            colors[node.name] = output_color
        else:
            colors[node.name] = default_color
    return colors


def make_border_color_dict(
    circuit: Circuit,
    rank_dict: dict[str, float] | None = None,
    max_rank: float = 50257,
    default_color: str = BORDER.to_hex(),
) -> dict[str, str]:
    """
    Circuit オブジェクトと各ノードのランク情報から ノード名 -> ボーダーカラー の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        rank_dict (dict[str, float]): ノード名をキー, ランクを値とする辞書
        max_rank (float): 最大ランク (default: 50257)
        default_color (str): ランク情報がないノードのデフォルトボーダーカラー

    Returns:
        dict[str, str]: ノード名をキー, ボーダーカラーコードを値とする辞書
    """
    border_colors: dict[str, str] = {}
    if rank_dict is None:
        for node in circuit.nodes.values():
            border_colors[node.name] = default_color
    else:
        for node in circuit.nodes.values():
            border_colors[node.name] = get_color_from_logits_rank(
                rank_dict.get(node.name, max_rank), max_rank
            )
        border_colors["input"] = get_color_from_logits_rank(max_rank, max_rank)
        border_colors["logits"] = border_colors.get(
            f"m{circuit.cfg['n_layers'] - 1}", default_color
        )
    return border_colors


def make_node_size_scale_dict(
    circuit: Circuit,
    score: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> サイズ倍率 の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        score (dict[str, float]): Attention Head 名をキー, スコアを値とする辞書

    Returns:
        dict[str, float]: ノード名をキー, サイズ倍率を値とする辞書
    """
    size_scale_dict: dict[str, float] = {}
    for node in circuit.nodes.values():
        scale = 1.0 + score.get(node.name, 0.0) if score is not None else 1.0
        size_scale_dict[node.name] = scale
    return size_scale_dict


def make_node_alpha_dict(
    circuit: Circuit,
    alpha_strength: float = 0.9,
) -> dict[str, str]:
    """
    Circuit オブジェクトから ノード名 -> 透明度 (16進文字列) の辞書を生成する関数.
    グラフに含まれるノード (node.in_graph が True) は完全不透明, 含まれないノードは半透明になる.
    alpha_strength で半透明ノードの透明度強度 (0.0-1.0) を指定可能.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        alpha_strength (float): 半透明ノードの透明度強度 (0.0 で完全不透明, 1.0 で完全透明に近くなる)

    Returns:
        dict[str, str]: ノード名をキー, 透明度 (16進文字列) を値とする辞書
    """
    alpha_dict: dict[str, str] = {}
    alpha_strength = max(0.0, min(1.0, alpha_strength))  # 0.0-1.0 の範囲にクリップ
    for node in circuit.nodes.values():
        alpha = 255 if node.in_graph else int(255 * (1 - alpha_strength))
        alpha_dict[node.name] = f"{alpha:02x}"  # 2桁の 16進数に変換
    return alpha_dict


def make_node_shape_dict(
    circuit: Circuit,
    score: dict[str, float | None] | None = None,
    score_threshold: float = 0.7,
    base_shape: str = "box",
    over_shape: str = "diamond",
) -> dict[str, str]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> ノード形状 の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        score (dict[str, float]): Attention Head 名をキー, スコアを値とする辞書
        score_threshold (float): スコアの閾値 (0.0-1.0)
        base_shape (str): スコアが閾値未満のノードの基本形状 (default: "box")
        over_shape (str): スコアが閾値以上のノードの形状 (default: "diamond")

    Returns:
        dict[str, str]: ノード名をキー, ノード形状を値とする辞書
    """
    shape_dict: dict[str, str] = {}
    for node in circuit.nodes.values():
        shape = base_shape  # デフォルトは base_shape
        if score is not None:
            node_score = score.get(node.name, 0.0) or 0.0
            if node_score >= score_threshold:
                shape = over_shape  # スコアが閾値以上なら over_shape
        shape_dict[node.name] = shape
    return shape_dict


def save_circuit_image(
    model: HookedTransformer,
    cache: dict | ActivationCache | None,
    df: pd.DataFrame,
    circuit: Circuit,
    relation_name: str,
    head_scores_dir: str | Path,
    output_path: str | Path = "circuit_attention.svg",
    metric: Literal["mean", "pearson", "auc"] = "pearson",
    prompt_col: str = "clean",
    relation_type: str | None = None,
    use_self_attention: bool = True,
    self_attention_threshold: float = 0.7,
    use_fillcolor: bool = True,
    use_size: bool = False,
    use_alpha: bool = False,
    alpha_strength: float = 0.9,
    urls: dict[str, str] | None = None,
    fontsize: int = FONTSIZE,
    node_width: float = NODE_WIDTH,
    node_height: float = NODE_HEIGHT,
    node_border_width: float = NODE_BORDER_WIDTH,
    edge_width: float = EDGE_WIDTH,
    display_not_in_graph: bool = False,
) -> None:
    """
    指定したモデル・キャッシュ・データフレーム・Circuit に対して, 各 Attention Head の Self / Subject / Relation スコアに基づき
    ノード色を決定し, Graphviz 形式で画像として保存する関数.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (Union[dict, ActivationCache, None]): Attention Pattern のキャッシュ
            注意: cache に None を渡す場合は各種 Attention Score が計算済みであることを前提とする
        df (pd.DataFrame): データセット
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        relation_name (str): Relation 名 (例: "city_in_country")
        head_scores_dir (str): Attention Head スコアが保存されているディレクトリ
        output_path (str): 画像ファイルの保存先パス
        metric (str): "mean", "pearson", "auc" のいずれか (スコア計算指標)
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対象部分に対する Pearson 相関係数を正規化したもの
            - "auc"    : 対象部分に対する AUC 値を正規化したもの
        prompt_col (str): プロンプトが格納されているカラム名
        relation_type (str): Relation タイプ (例: "factual", "bias")
        use_self_attention (bool): Self Attention Score を形状として反映させるか
        self_attention_threshold (float): 強い Self Attention とみなす閾値
            補足: pearson の場合は 0.7, auc の場合は 0.6 で強い Self Attention とみなす
            注意: 正規化の影響で auc の 0.6 は元の値の 0.8 に相当する
        use_fillcolor (bool): ノードの塗りつぶし色を指定するか
        use_size      (bool): ノードのサイズ倍率を指定するか
        use_alpha     (bool): ノードの透明度を指定するか
        alpha_strength (float): ノードの透明度強度 (0.0-1.0)
        urls (dict[str, str]): ノード名をキー, 画像表示用の JavaScript URL を値とする辞書
        fontsize  (int): ノードの基本フォントサイズ
        node_width   (float): ノードの基本幅
        node_height  (float): ノードの基本高さ
        node_border_width (float): ノードの枠線幅
        edge_width (float): エッジの幅
        display_not_in_graph (bool): グラフに含まれないノードも表示するか

    Returns:
        None
    """
    if use_fillcolor:
        # Subject Score の読み込み (ファイルが存在しない場合は計算する)
        head_subject_scores = get_head_scores(
            scores_dir=head_scores_dir,
            score_type="subject_scores",
            metric=metric,
            relation_type=relation_type,
            relation_name=relation_name,
            model=model,
            cache=cache,
            df=df,
            prompt_col=prompt_col,
        )

        # Relation Score の読み込み (ファイルが存在しない場合は計算する)
        head_relation_scores = get_head_scores(
            scores_dir=head_scores_dir,
            score_type="relation_scores",
            metric=metric,
            relation_type=relation_type,
            relation_name=relation_name,
            model=model,
            cache=cache,
            df=df,
            prompt_col=prompt_col,
        )

        # Subject / Relation Score 及び Target の出力順位からノードの塗りつぶし色を計算
        fillcolors = make_node_color_dict(
            circuit,
            red_scores=head_subject_scores,
            blue_scores=head_relation_scores,
        )
    else:
        fillcolors = None

    # ノードのボーダーカラーを計算
    try:
        layer_ranks = (
            load_logit_analysis_results(
                base_dir="out/logits",
                relation_type=relation_type,
                relation_name=relation_name,
                analysis_level="layer",
            )
            .get("layer", {})
            .get("ranks", {})
        )
    except Exception:
        layer_ranks = {}

    try:
        head_ranks = (
            load_logit_analysis_results(
                base_dir="out/logits",
                relation_type=relation_type,
                relation_name=relation_name,
                analysis_level="head",
            )
            .get("head", {})
            .get("ranks", {})
        )
    except Exception:
        head_ranks = {}

    # 辞書を安全に結合
    rank_dict = {**layer_ranks, **head_ranks}
    border_colors = make_border_color_dict(
        circuit,
        rank_dict=rank_dict if rank_dict else None,
        max_rank=model.cfg.d_vocab,
    )

    # ノードのサイズ倍率を計算 (開発中)
    size_scales = make_node_size_scale_dict(circuit, score=None) if use_size else None

    # 透明度を計算
    alphas = (
        make_node_alpha_dict(circuit, alpha_strength=alpha_strength)
        if use_alpha
        else None
    )

    if use_self_attention:
        # Self Attention Score の読み込み (ファイルが存在しない場合は計算する)
        head_self_scores = get_head_scores(
            scores_dir=head_scores_dir,
            score_type="self_scores",
            metric=metric,
            relation_type=relation_type,
            relation_name=relation_name,
            model=model,
            cache=cache,
            df=df,
            prompt_col=prompt_col,
        )

        # ノードの形状を計算
        shapes = make_node_shape_dict(
            circuit,
            score=head_self_scores,
            score_threshold=self_attention_threshold,
        )
    else:
        shapes = None

    # pygraphviz が利用可能かチェック
    import importlib.util

    if importlib.util.find_spec("pygraphviz") is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        circuit.to_svg_with_node_styles(
            output_path,
            border_colors=border_colors,
            fillcolors=fillcolors,
            alphas=alphas,
            size_scales=size_scales,
            shapes=shapes,
            urls=urls,
            fontsize=fontsize,
            node_width=node_width,
            node_height=node_height,
            node_border_width=node_border_width,
            edge_width=edge_width,
            display_not_in_graph=display_not_in_graph,
        )
        print(f"Saved SVG file: {output_path}")
    else:
        print("No pygraphviz installed; skipping SVG export")
