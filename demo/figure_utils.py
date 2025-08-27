import pickle
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from circuit.circuit_utils import Circuit
from circuit.visualize_circuit import save_circuit_image
from paths import (
    get_attention_image_path,
    get_cache_path,
    get_circuit_score_path,
    get_dataframe_path,
    get_scored_graph_path,
    get_svg_path,
    image_to_base64,
)

MAX_EDGE_COUNT = 3000


def apply_score_threshold_to_graph(
    score_list: list[tuple[int, int, float]], score_threshold: float, graph: Circuit
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
    if not score_list:
        print("Warning: score_list is empty, returning graph as-is")
        return graph

    # 閾値以上のスコアを持つ最初の topn_param を取得
    target_topn_param = None
    for topn_param, actual_edge_count, score in score_list:
        if score >= score_threshold:
            target_topn_param = topn_param
            print(
                f"Found score {score:.4f} >= threshold {score_threshold} at topn={topn_param} (actual_edges={actual_edge_count})"
            )
            break

    # 閾値以上のスコアが見つからない場合はそのまま返す
    if target_topn_param is None:
        available_scores = [score for _, _, score in score_list[:5]]
        max_score = max(score for _, _, score in score_list) if score_list else 0
        print(
            f"Warning: No score >= {score_threshold} found. Max available score: {max_score:.4f}. Top 5 scores: {available_scores}"
        )
        return graph

    # top-n エッジを適用
    graph.apply_topn(target_topn_param)
    print(
        f"Applied topn={target_topn_param} to graph based on score threshold {score_threshold}"
    )
    return graph


def apply_circuit_threshold(
    circuit: Circuit,
    relation_name: str,
    score_threshold: Optional[float] = None,
    topn: int = 200,
) -> Circuit:
    """
    Circuit にスコアしきい値または topn を適用する関数.

    Args:
        circuit (Circuit): 処理対象の Circuit オブジェクト.
        relation_name (str): Relation 名 (例: "city_in_country").
        score_threshold (float, optional): スコアの閾値. None の場合は topn を適用.
        topn (int): フォールバック用のトップ N 要素数.

    Returns:
        Circuit: しきい値または topn が適用された Circuit オブジェクト.
    """
    if score_threshold is not None:
        try:
            score_path = get_circuit_score_path(relation_name)
        except Exception as e:
            print(f"Warning: {e}. Using topn={topn} instead for {relation_name}.")
            circuit.apply_topn(topn)
            return circuit

        try:
            with open(score_path, "rb") as f:
                score_list = pickle.load(f)
            edge_score_list = [
                (topn_param, actual_edge_count, score)
                for topn_param, _, (actual_edge_count, score) in score_list
            ]
            circuit = apply_score_threshold_to_graph(
                edge_score_list, score_threshold, circuit
            )
        except Exception as e:
            print(f"Warning: Error loading score file: {e}. Using topn={topn} instead.")
            circuit.apply_topn(topn)
    else:
        circuit.apply_topn(topn)

    return circuit


def generate_circuit_svg(
    model: HookedTransformer,
    device: torch.device,
    relation_name: str,
    cache_base_dir: str = "out/cache",
    df_base_dir: str = "data/filtered_gpt2_small",
    circuit_base_dir: str = "out/scored_graphs_gpt2_small/pt",
    svg_path: Union[str, Path] = "demo/figures/circuit.svg",
    topn: int = 200,
    score_threshold: Optional[float] = None,
    attention_patterns_dir: str = "out/attention_patterns",
    head_scores_dir: str = "out/head_scores",
) -> None:
    """
    Circuit の読み込みと SVG の生成を行う関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        device (torch.device): 使用するデバイス (例: "cpu" or "cuda").
        relation_name (str): Relation 名 (例: "city_in_country").
        cache_base_dir (str): キャッシュファイルのベースディレクトリ.
        df_base_dir (str): データフレームファイルのベースディレクトリ.
        circuit_base_dir (str): Circuit ファイルのベースディレクトリ.
        svg_path (str): 出力 SVG ファイルのパス.
        topn (int): Circuit で使用するトップ N 要素数.
        score_threshold (float, optional): スコアの閾値. None の場合は適用しない.
        attention_patterns_dir (str): Attention Pattern 画像のディレクトリ.
        head_scores_dir (str): Head Score のディレクトリ.

    Returns:
        None
    """

    # キャッシュの読み込み (要改善: cache が分割されている場合の対応)
    cache_path = get_cache_path(relation_name, base_dir=cache_base_dir)
    cache = torch.load(cache_path, map_location=device, weights_only=False)

    # DataFrame の読み込み
    df_path = get_dataframe_path(relation_name, base_dir=df_base_dir)
    df = pd.read_csv(df_path)

    # Circuit の読み込み
    circuit_path = get_scored_graph_path(relation_name, base_dir=circuit_base_dir)
    circuit = Circuit.from_pt(circuit_path)

    # スコアしきい値または topn を適用
    circuit = apply_circuit_threshold(circuit, relation_name, score_threshold, topn)

    # Attention Pattern 画像の URL を生成
    urls = {}
    for node in circuit.nodes.values():
        if node.name.startswith("a"):
            layer = node.layer
            head = int(node.name.split(".h")[1])
            image_path = get_attention_image_path(
                relation_name, layer, head, base_dir=attention_patterns_dir
            )
            urls[node.name] = f"javascript:showImage('{image_to_base64(image_path)}')"
        else:
            urls[node.name] = ""

    # エッジ数が多すぎる場合は Circuit をリセット (何も表示されない画像を生成)
    if circuit.count_included_edges() > MAX_EDGE_COUNT:
        circuit.reset()

    # Circuit の SVG を生成して保存
    save_circuit_image(
        model=model,
        cache=cache,
        df=df,
        circuit=circuit,
        output_path=svg_path,
        metric="pearson",
        prompt_col="clean",
        head_scores_dir=head_scores_dir,
        relation_name=relation_name,
        use_self_attention=True,
        self_attention_threshold=0.7,
        use_fillcolor=True,
        use_size=True,
        use_alpha=True,
        alpha_strength=0.8,
        urls=urls,
        base_width=1.5,
        base_height=0.6,
        base_fontsize=24,
        penwidth=5,
        display_not_in_graph=False,
    )


def generate_circuit_multi_set_operation_svg(
    model: HookedTransformer,
    device: torch.device,
    base_relation_name: str,
    other_relation_names: list[str],
    mode: str = "intersection",
    cache_base_dir: str = "out/cache",
    df_base_dir: str = "data/filtered_gpt2_small",
    circuit_base_dir: str = "out/scored_graphs_gpt2_small/pt",
    svg_path: Union[str, Path] = "demo/figures/circuit.svg",
    topn: int = 200,
    score_threshold: Optional[float] = None,
    attention_patterns_dir: str = "out/attention_patterns",
    head_scores_dir: str = "out/head_scores",
) -> None:
    """
    基準となる 1つの Circuit に対して, 複数の Circuit の集合演算 (積集合・和集合・差集合) を行い, SVG として保存する関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        device (torch.device): 使用するデバイス (例: "cpu" or "cuda").
        base_relation_name (str): 基準となる Relation 名 (例: "city_in_country").
        other_relation_names (list[str]): 集合演算対象の Relation 名 (例: ["landmark_in_country"]).
        mode (str): 集合演算の種類.
            - "intersection" (積集合)
            - "union" (和集合)
            - "difference" (差集合)
        cache_base_dir (str): キャッシュファイルのベースディレクトリ.
        df_base_dir (str): データフレームファイルのベースディレクトリ.
        circuit_base_dir (str): Circuit ファイルのベースディレクトリ.
        svg_path (str): 出力 SVG ファイルのパス.
        topn (int): Circuit で使用するトップ N 要素数.
        score_threshold (float, optional): スコアの閾値. None の場合は適用しない.
        attention_patterns_dir (str): Attention Pattern 画像のディレクトリ.
        head_scores_dir (str): Head Score のディレクトリ.

    Returns:
        None
    """

    # キャッシュの読み込み (要改善: cache が分割されている場合の対応)
    cache_path = get_cache_path(base_relation_name, base_dir=cache_base_dir)
    cache = torch.load(cache_path, map_location=device, weights_only=False)

    # DataFrame の読み込み
    df_path = get_dataframe_path(base_relation_name, base_dir=df_base_dir)
    df = pd.read_csv(df_path)

    # Circuit の読み込みと SVG の生成
    base_circuit_path = get_scored_graph_path(
        base_relation_name, base_dir=circuit_base_dir
    )
    other_circuit_paths = [
        get_scored_graph_path(rel_name, base_dir=circuit_base_dir)
        for rel_name in other_relation_names
    ]
    base_circuit = Circuit.from_pt(base_circuit_path)
    other_circuits = [Circuit.from_pt(path) for path in other_circuit_paths]

    # base_circuit にスコアしきい値または topn を適用
    base_circuit = apply_circuit_threshold(
        base_circuit, base_relation_name, score_threshold, topn
    )

    # other_circuit にスコアしきい値または topn を適用
    for other_relation_name, other_circuit in zip(other_relation_names, other_circuits):
        other_circuit = apply_circuit_threshold(
            other_circuit, other_relation_name, score_threshold, topn
        )

    # Attention Pattern 画像の URL を生成
    urls = {}
    for node in base_circuit.nodes.values():
        if node.name.startswith("a"):
            layer = node.layer
            head = int(node.name.split(".h")[1])
            image_path = get_attention_image_path(
                base_relation_name, layer, head, base_dir=attention_patterns_dir
            )
            urls[node.name] = f"javascript:showImage('{image_to_base64(image_path)}')"
        else:
            urls[node.name] = ""

    # 集合演算
    if mode == "intersection":
        base_circuit.intersection_circuits(other_circuits)
    elif mode == "union":
        base_circuit.union_circuits(other_circuits)
    elif mode == "difference":
        base_circuit.difference_circuits(other_circuits)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # エッジ数が多すぎる場合は Circuit をリセット (何も表示されない画像を生成)
    if base_circuit.count_included_edges() > MAX_EDGE_COUNT:
        base_circuit.reset()
    save_circuit_image(
        model=model,
        cache=cache,
        df=df,
        circuit=base_circuit,
        output_path=svg_path,
        metric="pearson",
        prompt_col="clean",
        head_scores_dir=head_scores_dir,
        relation_name=base_relation_name,
        use_self_attention=True,
        self_attention_threshold=0.7,
        use_fillcolor=True,
        use_size=True,
        use_alpha=True,
        alpha_strength=0.8,
        urls=urls,
        base_width=1.5,
        base_height=0.6,
        base_fontsize=24,
        penwidth=5,
        display_not_in_graph=False,
    )


def generate_circuit_pairwise_set_operation_svg(
    model: HookedTransformer,
    device: torch.device,
    base_relation_name: str,
    other_relation_name: str,
    mode: str = "intersection",
    cache_base_dir: str = "out/cache",
    df_base_dir: str = "data/filtered_gpt2_small",
    circuit_base_dir: str = "out/scored_graphs_gpt2_small/pt",
    svg_path: Union[str, Path] = "demo/figures/circuit.svg",
    topn: int = 200,
    score_threshold: Optional[float] = None,
    attention_patterns_dir: str = "out/attention_patterns",
    head_scores_dir: str = "out/head_scores",
) -> None:
    """
    2つの Circuit に対して集合演算 (積集合・和集合・差集合・重み付き差集合) を行い, SVG として保存する関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        device (torch.device): 使用するデバイス (例: "cpu" or "cuda").
        base_relation_name (str): 基準となる Relation 名 (例: "city_in_country").
        other_relation_name (str): 集合演算対象の Relation 名 (例: "landmark_in_country").
        mode (str): 集合演算の種類.
            - "intersection" (積集合)
            - "union" (和集合)
            - "difference" (差集合)
            - "weighted_difference" (重み付き差集合)
        cache_base_dir (str): キャッシュファイルのベースディレクトリ.
        df_base_dir (str): データフレームファイルのベースディレクトリ.
        circuit_base_dir (str): Circuit ファイルのベースディレクトリ.
        svg_path (str): 出力 SVG ファイルのパス.
        topn (int): Circuit で使用するトップ N 要素数.
        score_threshold (float, optional): スコアの閾値. None の場合は適用しない.
        attention_patterns_dir (str): Attention Pattern 画像のディレクトリ.
        head_scores_dir (str): Head Score のディレクトリ.

    Returns:
        None
    """

    # キャッシュの読み込み (要改善: cache が分割されている場合の対応)
    cache_path = get_cache_path(base_relation_name, base_dir=cache_base_dir)
    cache = torch.load(cache_path, map_location=device, weights_only=False)

    # DataFrame の読み込み
    df_path = get_dataframe_path(base_relation_name, base_dir=df_base_dir)
    df = pd.read_csv(df_path)

    # Circuit の読み込みと SVG の生成
    base_circuit_path = get_scored_graph_path(
        base_relation_name, base_dir=circuit_base_dir
    )
    other_circuit_path = get_scored_graph_path(
        other_relation_name, base_dir=circuit_base_dir
    )
    base_circuit = Circuit.from_pt(base_circuit_path)
    other_circuit = Circuit.from_pt(other_circuit_path)

    # base_circuit にスコアしきい値または topn を適用
    base_circuit = apply_circuit_threshold(
        base_circuit, base_relation_name, score_threshold, topn
    )

    # other_circuit にスコアしきい値または topn を適用
    other_circuit = apply_circuit_threshold(
        other_circuit, other_relation_name, score_threshold, topn
    )

    # Attention Pattern 画像の URL を生成
    urls = {}
    for node in base_circuit.nodes.values():
        if node.name.startswith("a"):
            layer = node.layer
            head = int(node.name.split(".h")[1])
            image_path = get_attention_image_path(
                base_relation_name, layer, head, base_dir=attention_patterns_dir
            )
            urls[node.name] = f"javascript:showImage('{image_to_base64(image_path)}')"
        else:
            urls[node.name] = ""

    # 集合演算
    if mode == "intersection":
        base_circuit.intersection_circuits(other_circuit)
    elif mode == "union":
        base_circuit.union_circuits(other_circuit)
    elif mode == "difference":
        base_circuit.difference_circuits(other_circuit)
    elif mode == "weighted_difference":
        base_circuit.weighted_difference_circuits(other_circuit)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # エッジ数が多すぎる場合は Circuit をリセット (何も表示されない画像を生成)
    if base_circuit.count_included_edges() > MAX_EDGE_COUNT:
        base_circuit.reset()
    save_circuit_image(
        model=model,
        cache=cache,
        df=df,
        circuit=base_circuit,
        output_path=svg_path,
        metric="pearson",
        prompt_col="clean",
        head_scores_dir=head_scores_dir,
        relation_name=base_relation_name,
        use_self_attention=True,
        self_attention_threshold=0.7,
        use_fillcolor=True,
        use_size=True,
        use_alpha=True,
        alpha_strength=0.8,
        urls=urls,
        base_width=1.5,
        base_height=0.6,
        base_fontsize=24,
        penwidth=5,
        display_not_in_graph=False,
    )


def create_all_circuit_pairwise_set_operation_svg(
    model: HookedTransformer,
    device: torch.device,
    relation_list: list[str],
    topn: int = 200,
    score_threshold: Optional[float] = None,
    mode: str = "intersection",
) -> None:
    """
    全ての Relation の組み合わせに対して集合演算 (積集合・和集合・差集合・重み付き差集合) の SVG ファイルを生成する関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル.
        device (torch.device): 使用するデバイス (例: "cpu" or "cuda").
        relation_list (list[str]): Relation 名のリスト.
        topn (int): Circuit で使用するトップ N 要素数.
        score_threshold (float, optional): スコアの閾値. None の場合は適用しない.
        mode (str): 集合演算の種類. "intersection", "union", "difference", "weighted_difference" から選択.

    Returns:
        None
    """
    for base_relation in relation_list:
        for other_relation in relation_list:
            if base_relation == other_relation:
                continue

            # SVG パスを生成 (スコアしきい値または top-n)
            svg_path = get_svg_path(
                base_relation=base_relation,
                other_relation=other_relation,
                set_operation_mode=mode,
                topn=topn if score_threshold is None else None,
                perf_percent=score_threshold,
            )

            if not svg_path.exists():
                generate_circuit_pairwise_set_operation_svg(
                    model=model,
                    device=device,
                    base_relation_name=base_relation,
                    other_relation_name=other_relation,
                    mode=mode,
                    svg_path=svg_path,
                    topn=topn,
                    score_threshold=score_threshold,
                )
