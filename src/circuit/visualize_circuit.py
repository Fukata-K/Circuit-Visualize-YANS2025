import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformer_lens import ActivationCache, HookedTransformer

from analysis.analysis_logits import load_logit_analysis_results
from analysis.head_scoring import get_head_scores
from circuit.circuit_utils import Circuit


def get_color_from_score(score_red: float, score_blue: float) -> str:
    """
    スコアに基づいて色を計算するヘルパー関数.

    Args:
        score_red  (float): 赤のスコア (0.0 から 1.0)
        score_blue (float): 青のスコア (0.0 から 1.0)

    Returns:
        str: 16進カラーコード (例: "#A1B2C3")
    """
    r = int(255 - max(0, score_blue - score_red) * 255)
    b = int(255 - max(0, score_red - score_blue) * 255)
    g = int(255 - max(score_red, score_blue) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_color_from_logits_rank(rank: float, max_rank: float) -> str:
    """
    target の rank に基づいて色を計算するヘルパー関数.
    順位が良い (数値が小さい) ほど緑色, 悪い (数値が大きい) ほど白色になる.

    Args:
        rank (float): target のランク
        max_rank (float): 最大ランク

    Returns:
        str: 16進カラーコード (例: "#A1B2C3")
    """
    log_rank = np.log(rank + 1e-8)
    log_max_rank = np.log(max_rank + 1e-8)
    normalized_log_rank = log_rank / log_max_rank if log_max_rank > 0 else 0.0

    # 順位が良いほど緑, 悪いほど白になるように調整
    r = int(255 * normalized_log_rank)
    g = 255
    b = int(255 * normalized_log_rank)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_node_color_dict(
    circuit: Circuit,
    red_scores: Dict[str, Optional[float]],
    blue_scores: Dict[str, Optional[float]],
    input_color: str = "#808080",
    mlp_color: str = "#CCFFCC",
    output_color: str = "#FFD700",
    default_color: str = "#FFFFFF",
    rank_dict: Optional[Dict[str, float]] = None,
    max_rank: float = 50257,
) -> Dict[str, str]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> 色コード の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        red_scores  (Dict[str, float]): Attention Head 名をキー, 赤スコアを値とする辞書
        blue_scores (Dict[str, float]): Attention Head 名をキー, 青スコアを値とする辞書
        input_color   (str): 入力ノードの色 (default: グレー)
        mlp_color     (str): MLP ノードの色 (default: 薄緑)
        output_color  (str): 出力ノードの色 (default: 金色)
        default_color (str): その他ノードの色 (default: 白, 通常は存在しない)

    Returns:
        Dict[str, str]: ノード名をキー, 16進カラーコードを値とする辞書
    """
    colors: Dict[str, str] = {}
    for node in circuit.nodes.values():
        if node.name == "input":
            colors[node.name] = input_color
        elif node.name.startswith("a"):  # Attention Head
            colors[node.name] = get_color_from_score(
                red_scores.get(node.name, 0.0) or 0.0,
                blue_scores.get(node.name, 0.0) or 0.0,
            )
        elif node.name.startswith("m"):  # MLP
            if rank_dict is None:
                colors[node.name] = mlp_color
            else:
                colors[node.name] = get_color_from_logits_rank(
                    rank_dict.get(node.name, max_rank), max_rank
                )
        elif node.name == "logits":
            colors[node.name] = output_color
        else:
            colors[node.name] = default_color
    return colors


def make_border_color_dict(
    circuit: Circuit,
    rank_dict: Optional[Dict[str, float]] = None,
    max_rank: float = 50257,
    default_color: str = "#000000",
) -> Dict[str, str]:
    """
    Circuit オブジェクトと各ノードのランク情報から ノード名 -> ボーダーカラー の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        rank_dict (Dict[str, float]): ノード名をキー, ランクを値とする辞書
        max_rank (float): 最大ランク (default: 50257)
        default_color (str): ランク情報がないノードのデフォルトボーダーカラー (default: black)

    Returns:
        Dict[str, str]: ノード名をキー, ボーダーカラーコードを値とする辞書
    """
    border_colors: Dict[str, str] = {}
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
    score: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> サイズ倍率 の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        score (Dict[str, float]): Attention Head 名をキー, スコアを値とする辞書

    Returns:
        Dict[str, float]: ノード名をキー, サイズ倍率を値とする辞書
    """
    size_scale_dict: Dict[str, float] = {}
    for node in circuit.nodes.values():
        scale = 1.0 + score.get(node.name, 0.0) if score is not None else 1.0
        size_scale_dict[node.name] = scale
    return size_scale_dict


def make_node_alpha_dict(
    circuit: Circuit,
    alpha_strength: float = 0.9,
) -> Dict[str, str]:
    """
    Circuit オブジェクトから ノード名 -> 透明度 (16進文字列) の辞書を生成する関数.
    グラフに含まれるノード (node.in_graph が True) は完全不透明, 含まれないノードは半透明になる.
    alpha_strength で半透明ノードの透明度強度 (0.0-1.0) を指定可能.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        alpha_strength (float): 半透明ノードの透明度強度 (0.0 で完全不透明, 1.0 で完全透明に近くなる)

    Returns:
        Dict[str, str]: ノード名をキー, 透明度 (16進文字列) を値とする辞書
    """
    alpha_dict: Dict[str, str] = {}
    alpha_strength = max(0.0, min(1.0, alpha_strength))  # 0.0-1.0 の範囲にクリップ
    for node in circuit.nodes.values():
        alpha = 255 if node.in_graph else int(255 * (1 - alpha_strength))
        alpha_dict[node.name] = f"{alpha:02x}"  # 2桁の 16進数に変換
    return alpha_dict


def make_node_shape_dict(
    circuit: Circuit,
    score: Optional[Dict[str, Optional[float]]] = None,
    score_threshold: float = 0.7,
    base_shape: str = "box",
    over_shape: str = "diamond",
) -> Dict[str, str]:
    """
    Circuit オブジェクトと各ノードのスコア情報から ノード名 -> ノード形状 の辞書を生成する関数.

    Args:
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        score (Dict[str, float]): Attention Head 名をキー, スコアを値とする辞書
        score_threshold (float): スコアの閾値 (0.0-1.0)
        base_shape (str): スコアが閾値未満のノードの基本形状 (default: "box")
        over_shape (str): スコアが閾値以上のノードの形状 (default: "diamond")

    Returns:
        Dict[str, str]: ノード名をキー, ノード形状を値とする辞書
    """
    shape_dict: Dict[str, str] = {}
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
    cache: Union[Dict, ActivationCache, None],
    df: pd.DataFrame,
    circuit: Circuit,
    relation_name: str,
    head_scores_dir: Union[str, Path],
    output_path: Union[str, Path] = "circuit_attention.svg",
    metric: Literal["mean", "pearson", "auc"] = "pearson",
    prompt_col: str = "clean",
    relation_type: Optional[str] = None,
    use_self_attention: bool = True,
    self_attention_threshold: float = 0.7,
    use_fillcolor: bool = False,
    use_size: bool = False,
    use_alpha: bool = False,
    alpha_strength: float = 0.9,
    urls: Optional[Dict[str, str]] = None,
    base_width: float = 0.75,
    base_height: float = 0.5,
    base_fontsize: int = 14,
    penwidth: float = 1.0,
    display_not_in_graph: bool = False,
) -> None:
    """
    指定したモデル・キャッシュ・データフレーム・Circuit に対して, 各 Attention Head の Self / Subject / Relation スコアに基づき
    ノード色を決定し, Graphviz 形式で画像として保存する関数.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (Union[Dict, ActivationCache, None]): Attention Pattern のキャッシュ
            注意: cache に None を渡す場合は各種 Attention Score が計算済みであることを前提とする
        df (pd.DataFrame): データセット
        circuit (Circuit): 可視化対象の Circuit オブジェクト
        output_path (str): 画像ファイルの保存先パス
        metric (str): "mean", "pearson", "auc" のいずれか (スコア計算指標)
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対象部分に対する Pearson 相関係数を正規化したもの
            - "auc"    : 対象部分に対する AUC 値を正規化したもの
        prompt_col (str): プロンプトが格納されているカラム名
        head_scores_dir (str): Attention Head スコアが保存されているディレクトリ
        relation_type (str): Relation タイプ (例: "factual", "bias")
        relation_name (str): Relation 名 (例: "city_in_country")
        use_self_attention (bool): Self Attention Score を形状として反映させるか
        self_attention_threshold (float): 強い Self Attention とみなす閾値
            補足: pearson の場合は 0.7, auc の場合は 0.6 で強い Self Attention とみなす
            注意: 正規化の影響で auc の 0.6 は元の値の 0.8 に相当する
        use_fillcolor (bool): ノードの塗りつぶし色を指定するか
        use_size      (bool): ノードのサイズ倍率を指定するか
        use_alpha     (bool): ノードの透明度を指定するか
        alpha_strength (float): ノードの透明度強度 (0.0-1.0)
        urls (Dict[str, str]): ノード名をキー, 画像表示用の JavaScript URL を値とする辞書
        base_width   (float): ノードの基本幅
        base_height  (float): ノードの基本高さ
        base_fontsize  (int): ノードの基本フォントサイズ
        penwidth (float): エッジの幅
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

        # Target の出力順位を読み込み
        try:
            target_rank = load_logit_analysis_results(
                base_dir="out/logits",
                relation_type=relation_type,
                relation_name=relation_name,
                analysis_level="layer",
            )["layer"]["ranks"]
        except Exception as e:
            target_rank = None
            print(f"Error loading target ranks: {e}")

        # Subject / Relation Score 及び Target の出力順位からノードの塗りつぶし色を計算
        fillcolors = make_node_color_dict(
            circuit,
            red_scores=head_subject_scores,
            blue_scores=head_relation_scores,
            rank_dict=target_rank,
            max_rank=model.cfg.d_vocab,
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

    # 画像として保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # pygraphviz が利用可能かチェック
    import importlib.util

    if importlib.util.find_spec("pygraphviz") is not None:
        circuit.to_svg_with_node_styles(
            output_path,
            border_colors=border_colors,
            fillcolors=fillcolors,
            alphas=alphas,
            size_scales=size_scales,
            shapes=shapes,
            urls=urls,
            base_width=base_width,
            base_height=base_height,
            base_fontsize=base_fontsize,
            node_border_width=10.0,
            edge_width=penwidth,
            display_not_in_graph=display_not_in_graph,
        )
        print(f"Saved SVG file: {output_path}")
    else:
        print("No pygraphviz installed; skipping SVG export")


def apply_score_threshold_to_graph(
    score_list: List[Tuple[int, int, float]], score_threshold: float, graph: Circuit
) -> Circuit:
    """
    スコアリストから閾値以上になる部分での Circuit オブジェクトを返す.

    Args:
        score_list: (topn_param, actual_edge_count, score) のタプルのリスト
            - topn_param: apply_topn() に渡すパラメータ値
            - actual_edge_count: 実際に生成されるグラフのエッジ数
            - score: そのエッジ数での評価スコア
        score_threshold: スコアの閾値
        graph: 処理対象の Circuit オブジェクト

    Returns:
        Circuit: 閾値以上のスコアを持つエッジが適用された Circuit オブジェクト

    Note:
        - 元の Circuit オブジェクトが変更されるため必要に応じて事前にコピーを作成する必要がある
        - topn_param と actual_edge_count が異なるのは孤立ノード除去の影響
    """
    # 閾値以上のスコアを持つ最初の topn_param を取得
    target_topn_param = None
    for topn_param, _, score in score_list:
        if score >= score_threshold:
            target_topn_param = topn_param
            break

    # 閾値以上のスコアが見つからない場合はそのまま返す
    if target_topn_param is None:
        return graph

    # top-n エッジを適用
    graph.apply_topn(target_topn_param)
    return graph


def save_all_circuit_images(
    dataset_paths: Union[List[str], List[Path]],
    cache_dir: str,
    output_dir: str,
    circuit_score_dir: str,
    scored_graphs_dir: str,
    metric: Literal["mean", "pearson", "auc"],
    model: HookedTransformer,
    device: torch.device,
    head_scores_dir: Union[str, Path],
    prompt_col: str = "clean",
    self_attention_threshold: float = 0.7,
    score_threshold: Optional[float] = None,
    topn_param: int = 200,
    max_edge_count: int = 1000,
    alpha_strength: float = 0.9,
    base_width: float = 1.5,
    base_height: float = 0.6,
    base_fontsize: int = 24,
    display_not_in_graph: bool = False,
) -> None:
    """
    複数のデータセットパスに対して Circuit の可視化画像を一括で保存する関数.

    Args:
        dataset_paths (List[str]): 入力データセット (CSV ファイル) のパス一覧
        cache_dir (str): キャッシュが保存されているディレクトリ
        output_dir (str): 画像出力ディレクトリ
        circuit_score_dir (str): Circuit の性能 (Exact Match など) が保存されているディレクトリ
        scored_graphs_dir (str): スコア付きグラフが保存されているディレクトリ
        metric (str): スコア計算指標 ("mean", "pearson", "auc" のいずれか)
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対象部分に対する Pearson 相関係数を正規化したもの
            - "auc"    : 対象部分に対する AUC 値を正規化したもの
        model (HookedTransformer): トークナイズ用モデル
        device (torch.device): モデル・キャッシュのロード先デバイス
        prompt_col (str): プロンプトが格納されているカラム名
        head_scores_dir (str): Attention Head スコアが保存されているディレクトリ
        self_attention_threshold (float): 強い Self Attention とみなす閾値
            補足: pearson の場合は 0.7, auc の場合は 0.6 で強い Self Attention とみなす
            注意: 正規化の影響で auc の 0.6 は元の値の 0.8 に相当する
        score_threshold (float): Circuit に求めるスコアの閾値
            - None の場合は topn_param を用いた apply_topn() を適用した Circuit を使用
        topn_param (int): apply_topn() に渡すパラメータ値
            - 例: 200 ならば上位 200 エッジを残す
            - 注意: 実際のエッジ数は孤立ノード除去により topn_param より少なくなる場合がある
        max_edge_count (int): 可視化する Circuit の最大エッジ数. この数を超える場合は可視化をスキップする
        alpha_strength (float): ノードの透明度強度 (0.0-1.0)
        base_width (float): ノードの基本幅
        base_height (float): ノードの基本高さ
        base_fontsize (int): ノードの基本フォントサイズ
        display_not_in_graph (bool): グラフに含まれないノードも表示するか

    Returns:
        None
    """
    for dataset_path in dataset_paths:
        try:
            relation_type = Path(dataset_path).parent.name
            relation_name = Path(dataset_path).stem

            # キャッシュの読み込み (要改善: cache が分割されている場合の対応)
            cache_path = sorted(
                Path(f"{cache_dir}/{relation_type}").glob(f"{relation_name}*.pt")
            )[0]
            cache = torch.load(cache_path, map_location=device, weights_only=False)

            # データセットの読み込み
            df = pd.read_csv(dataset_path)

            # edge_score_list: (topn_param, actual_edge_count, score)
            # topn_param: apply_topn() に渡すパラメータ値
            # actual_edge_count: 実際に生成されるグラフのエッジ数 (孤立ノード除去により topn_param より少なくなる場合がある)
            score_path = f"{circuit_score_dir}/{relation_type}/{relation_name}_node_edge_scores.pkl"
            score_path = Path(score_path)
            with open(score_path, "rb") as f:
                score_list = pickle.load(f)
            edge_score_list = [
                (topn_param, actual_edge_count, score)
                for topn_param, _, (actual_edge_count, score) in score_list
            ]

            # Circuit の読み込みとスコア閾値の適用
            circuit_path = f"{scored_graphs_dir}/{relation_type}/{relation_name}.pt"
            circuit = Circuit.from_pt(circuit_path)
            if score_threshold is None:
                circuit.apply_topn(topn_param)
            else:
                circuit = apply_score_threshold_to_graph(
                    edge_score_list, score_threshold, circuit
                )

            # 画像出力パス
            output_path = (
                f"{output_dir}/{metric}/{relation_type}/{relation_name}_{metric}.svg"
            )

            if circuit.count_included_edges() <= max_edge_count:
                # 可視化画像の保存
                save_circuit_image(
                    model=model,
                    cache=cache,
                    df=df,
                    circuit=circuit,
                    output_path=output_path,
                    metric=metric,
                    prompt_col=prompt_col,
                    head_scores_dir=head_scores_dir,
                    relation_type=relation_type,
                    relation_name=relation_name,
                    use_self_attention=True,
                    self_attention_threshold=self_attention_threshold,
                    use_fillcolor=True,
                    use_size=True,
                    use_alpha=True,
                    alpha_strength=alpha_strength,
                    base_width=base_width,
                    base_height=base_height,
                    base_fontsize=base_fontsize,
                    display_not_in_graph=display_not_in_graph,
                )
            else:
                print(
                    f"Skip: {relation_type}/{relation_name} Circuit has {circuit.count_included_edges()}",
                    "edges and is not suitable for visualization",
                )
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            continue


def main(
    model_name: str = "gpt2-small",
    dataset_base_dir: str = "data/filtered_gpt2_small",
    cache_dir: str = "out/cache",
    output_dir: str = "out/circuit_svg",
    metric: Literal["mean", "pearson", "auc"] = "pearson",
    head_scores_dir: str = "out/head_scores",
    circuit_score_dir: str = "out/circuit_scores/raw",
    scored_graphs_dir: str = "out/scored_graphs_gpt2_small/pt",
    self_attention_threshold: float = 0.7,
    score_threshold: Optional[float] = 0.8,
    topn_param: int = 200,
    max_edge_count: int = 1000,
    alpha_strength: float = 0.9,
    base_width: float = 1.5,
    base_height: float = 0.6,
    base_fontsize: int = 24,
    display_not_in_graph: bool = False,
) -> None:
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルの設定
    model = HookedTransformer.from_pretrained(model_name, device=device)

    dataset_dir = Path(dataset_base_dir)
    dataset_paths = sorted(list(dataset_dir.glob("**/*.csv")))

    save_all_circuit_images(
        dataset_paths=dataset_paths,
        cache_dir=cache_dir,
        output_dir=output_dir,
        circuit_score_dir=circuit_score_dir,
        scored_graphs_dir=scored_graphs_dir,
        metric=metric,
        model=model,
        device=device,
        head_scores_dir=head_scores_dir,
        self_attention_threshold=self_attention_threshold,
        score_threshold=score_threshold,
        topn_param=topn_param,
        max_edge_count=max_edge_count,
        alpha_strength=alpha_strength,
        base_width=base_width,
        base_height=base_height,
        base_fontsize=base_fontsize,
        display_not_in_graph=display_not_in_graph,
    )


if __name__ == "__main__":
    main(
        model_name="gpt2-small",
        dataset_base_dir="data/filtered_gpt2_small",
        cache_dir="out/cache",
        output_dir="out/circuit_svg/top200",
        metric="pearson",
        head_scores_dir="out/head_scores",
        circuit_score_dir="out/circuit_scores/raw",
        scored_graphs_dir="out/scored_graphs_gpt2_small/pt",
        self_attention_threshold=0.7,
        score_threshold=None,
        topn_param=200,
        max_edge_count=5000,
        alpha_strength=0.95,
        base_width=1.5,
        base_height=0.6,
        base_fontsize=24,
        display_not_in_graph=False,
    )
