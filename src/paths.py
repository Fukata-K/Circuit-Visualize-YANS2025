import base64
import os
from pathlib import Path
from typing import Dict, Optional, Union

DATA_DIR = Path("data")
OUT_DIR = Path("out")
DATAFRAME_DIR = DATA_DIR / "filtered_gpt2_small"
SCORED_GRAPH_DIR = OUT_DIR / "scored_graphs_gpt2_small" / "pt"
HEAD_SCORE_DIR = OUT_DIR / "head_scores"
CIRCUIT_SCORE_DIR = OUT_DIR / "circuit_scores" / "raw"
CACHE_DIR = OUT_DIR / "cache"
ATTENTION_PATTERN_DIR = OUT_DIR / "attention_patterns"
LOGITS_DIR = OUT_DIR / "logits"
SVG_DIR = Path("demo/figures")


def get_dataframe_path(
    relation_name: str,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = DATAFRAME_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    保存時:
        get_dataframe_path("city_in_country", relation_type="factual", ensure_parent=True)
    読み込み時:
        get_dataframe_path("city_in_country")

    - relation_name に対応する CSV ファイルのパスを返す.
    - relation_type 指定時: base/{relation_type}/{relation_name}.csv
    - 未指定時: base/*/{relation_name}.csv を探索 (1件のみ許容)

    Args:
        relation_name (str): 一意のリレーション名 (例: "city_in_country").
        relation_type (Optional[str]): サブディレクトリ (例: "factual", "commonsense" など). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: DATAFRAME_DIR).
        ensure_parent (bool): True なら保存前に親ディレクトリを作成する.

    Returns:
        Path: 該当する CSV ファイルの Path.

    Raises:
        FileNotFoundError: 見つからなかった場合.
        RuntimeError: 複数一致した場合.
    """
    base = Path(base_dir)
    filename = f"{relation_name}.csv"

    if relation_type is not None:
        path = base / relation_type / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    matches = list(base.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"No dataframe file found for relation_name='{relation_name}' under {base}"
        )
    if len(matches) > 1:
        # どの relation_type か曖昧なのでエラー
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple dataframe files found for relation_name='{relation_name}':{candidates}"
        )
    return matches[0]


def get_circuit_score_path(
    relation_name: str,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = CIRCUIT_SCORE_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    保存時:
        get_circuit_score_path("city_in_country", relation_type="factual", ensure_parent=True)
    読み込み時:
        get_circuit_score_path("city_in_country")

    Args:
        relation_name (str): 一意のリレーション名 (例: "city_in_country").
        relation_type (Optional[str]): サブディレクトリ (例: "factual", "commonsense" など). 未指定なら自動検出.
        base_dir (Path | str): ルート (default: CIRCUIT_SCORE_DIR).
        ensure_parent (bool): True なら保存前に親ディレクトリを作成する.

    Returns:
        Path: 対応する .pkl の Path.
            - .pkl の中身: list[tuple[int, tuple[int, float], tuple[int, float]]]
            - 各段階での (指定エッジ数, (ノード数, Score), (実際のエッジ数, Score)) のリスト

    Raises:
        FileNotFoundError: 自動検出時に見つからない場合.
        RuntimeError: 自動検出時に複数ヒットして一意でない場合.
    """
    base = Path(base_dir)
    filename = f"{relation_name}_node_edge_scores.pkl"

    if relation_type is not None:
        path = base / relation_type / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    matches = list(base.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"No score file found for relation_name='{relation_name}' under {base}"
        )
    if len(matches) > 1:
        # どの relation_type か曖昧なのでエラー
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple score files found for relation_name='{relation_name}':{candidates}"
        )
    return matches[0]


def get_cache_path(
    relation_name: str,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = CACHE_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    保存時:
        get_cache_path("city_in_country", relation_type="factual", ensure_parent=True)
    読み込み:
        get_cache_path("city_in_country")

    - relation_type 指定時は base/{relation_type} 直下で f"{relation_name}*.pt" を検索
    - 未指定時は base/*/ に対して同パターン検索
    - 複数一致時はソート後の先頭を返す (将来の分割キャッシュ対応は別途実装)

    Args:
        relation_name (str): 一意のリレーション名 (例: "city_in_country").
        relation_type (Optional[str]): サブディレクトリ (例: "factual", "commonsense" など). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: CACHE_DIR).
        ensure_parent (bool): True なら保存前に親ディレクトリを作成する.

    Returns:
        Path: 一致した .pt ファイルの Path.

    Raises:
        FileNotFoundError: 一致ファイルが見つからない場合.
    """
    base = Path(base_dir)
    pattern = f"{relation_name}*.pt"

    if relation_type is not None:
        cache_dir = base / relation_type
        if ensure_parent:
            cache_dir.mkdir(parents=True, exist_ok=True)
        matches = sorted(cache_dir.glob(pattern))
    else:
        # relation_type 未指定 -> 1 階層下 (*/) を検索
        matches = sorted(base.glob(f"*/{pattern}"))

    if not matches:
        raise FileNotFoundError(
            f"No cache file found for relation_name='{relation_name}' under {base}"
        )

    # TODO: 分割キャッシュ (複数 .pt) を統合読み込みする処理を実装
    return matches[0]


def get_scored_graph_path(
    relation_name: str,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = SCORED_GRAPH_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    保存時:
        get_scored_graph_path("city_in_country", relation_type="factual", ensure_parent=True)
    読み込み時:
        get_scored_graph_path("city_in_country")

    - relation_name に対応する Circuit の .pt ファイルのパスを返す.
    - relation_type 指定時: base/{relation_type}/{relation_name}.pt
    - 未指定時: base/*/{relation_name}.pt を探索 (1件のみ許容)

    Args:
        relation_name (str): 一意のリレーション名 (例: "city_in_country").
        relation_type (Optional[str]): サブディレクトリ (例: "factual", "commonsense" など). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: SCORED_GRAPH_DIR).
        ensure_parent (bool): True なら保存前に親ディレクトリを作成する.

    Returns:
        Path: 対応する .pt ファイルの Path.

    Raises:
        FileNotFoundError: 見つからなかった場合.
        RuntimeError: 複数一致した場合.
    """
    base = Path(base_dir)
    filename = f"{relation_name}.pt"

    if relation_type is not None:
        path = base / relation_type / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    matches = list(base.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"No circuit file found for relation_name='{relation_name}' under {base}"
        )
    if len(matches) > 1:
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple circuit files found for relation_name='{relation_name}':{candidates}"
        )
    return matches[0]


def get_head_score_path(
    relation_name: str,
    score_type: str,
    metric: str,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = HEAD_SCORE_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    保存時:
        get_head_score_path("city_in_country", score_type="self_scores", metric="pearson",
                            relation_type="factual", ensure_parent=True)
    読み込み時:
        get_head_score_path("city_in_country", score_type="self_scores", metric="pearson")

    Args:
        relation_name (str): リレーション名 (例: "city_in_country").
        score_type (str): "self_scores" | "subject_scores" | "relation_scores" など.
        metric (str): 指標名 (例: "pearson", "auc").
        relation_type (Optional[str]): リレーション種別 (例: "factual", "bias"). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: out/head_scores).
        ensure_parent (bool): True のとき保存前に親ディレクトリを作成.

    Returns:
        Path: 対応する .pkl の Path.

    Raises:
        FileNotFoundError: 自動検出時に見つからない場合
        RuntimeError: 自動検出時に複数ヒットして一意でない場合
    """
    base = Path(base_dir)
    filename = f"{relation_name}.pkl"

    if relation_type is not None:
        path = base / score_type / metric / relation_type / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    pattern = f"{score_type}/{metric}/*/{filename}"
    matches = list(base.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No head score file found for relation_name='{relation_name}', "
            f"score_type='{score_type}', metric='{metric}' under {base}"
        )
    if len(matches) > 1:
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            "Multiple head score files found for "
            f"relation_name='{relation_name}', score_type='{score_type}', metric='{metric}':"
            f"{candidates}"
        )
    return matches[0]


def get_attention_image_path(
    relation_name: str,
    layer: int,
    head: int,
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = ATTENTION_PATTERN_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    Attention パターン画像のパスを取得する.

    保存時:
        get_attention_image_path("city_in_country", layer=1, head=5, relation_type="factual", ensure_parent=True)

    読み込み時:
        get_attention_image_path("city_in_country", layer=1, head=5)

    Args:
        relation_name (str): 一意のリレーション名 (例: "city_in_country").
        layer (int): Attention Layer のインデックス.
        head (int): Attention Head のインデックス.
        relation_type (Optional[str]): サブディレクトリ (例: "factual", "commonsense" など). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: out/attention_patterns).
        ensure_parent (bool): True なら保存前に親ディレクトリを作成する.

    Returns:
        Path: 該当する PNG ファイルの Path.

    Raises:
        FileNotFoundError: 自動検出時に見つからない場合.
        RuntimeError: 自動検出時に複数ヒットして一意でない場合.
    """
    base = Path(base_dir)
    filename = f"layer{layer:02d}_head{head:02d}.png"

    if relation_type is not None:
        path = base / relation_type / relation_name / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    matches = list(base.glob(f"*/{relation_name}/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"No attention image found for relation_name='{relation_name}', layer={layer}, head={head} under {base}"
        )
    if len(matches) > 1:
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple attention images found for relation_name='{relation_name}', layer={layer}, head={head}:{candidates}"
        )
    return matches[0]


def get_logit_result_path(
    relation_name: str,
    level: str,  # "layer" | "head"
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = LOGITS_DIR,
    ensure_parent: bool = False,
) -> Path:
    """
    logit 分析結果 (単一レベル) の .pkl パスを返す.

    保存時:
        get_logit_result_path("city_in_country", level="layer", relation_type="factual", ensure_parent=True)

    読み込み時:
        get_logit_result_path("city_in_country", level="head")

    Args:
        relation_name (str): 例 "city_in_country".
        level (str): "layer" または "head".
        relation_type (Optional[str]): 未指定なら自動検出 (base/{level}/*/{relation_name}.pkl を探索).
        base_dir (Path | str): 既定は out/logits.
        ensure_parent (bool): True のとき保存前に親ディレクトリを作成.

    Returns:
        Path: 対応する .pkl の Path

    Raises:
        ValueError: level が不正
        FileNotFoundError: 自動検出で見つからない
        RuntimeError: 自動検出で複数一致
    """
    if level not in {"layer", "head"}:
        raise ValueError(f"level must be 'layer' or 'head', got '{level}'")

    base = Path(base_dir)
    filename = f"{relation_name}.pkl"

    if relation_type is not None:
        path = base / level / relation_type / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # relation_type 未指定: 自動検出
    matches = list((base / level).glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(
            f"No logit result found for relation_name='{relation_name}', level='{level}' under {base / level}"
        )
    if len(matches) > 1:
        candidates = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple logit results found for relation_name='{relation_name}', level='{level}':{candidates}"
        )
    return matches[0]


def get_logit_result_paths(
    relation_name: str,
    analysis_level: str = "both",  # "layer" | "head" | "both"
    relation_type: Optional[str] = None,
    base_dir: Union[str, Path] = LOGITS_DIR,
    ensure_parent: bool = False,
) -> Dict[str, Path]:
    """
    analysis_level に応じて必要なレベルのパスをまとめて返す.

    Args:
        relation_name (str): リレーション名 (例: "city_in_country").
        analysis_level (str): "layer" | "head" | "both". 両方指定時は両方のパスを返す.
        relation_type (Optional[str]): リレーション種別 (例: "factual", "bias"). 未指定なら自動検出.
        base_dir (Path | str): ルートディレクトリ (default: LOGITS_DIR).
        ensure_parent (bool): True のとき保存前に親ディレクトリを作成.

    Returns:
        Dict[str, Path]: {"layer": Path, "head": Path} のうち必要なキーのみ.
    """
    valid = {"layer", "head", "both"}
    if analysis_level not in valid:
        raise ValueError(
            f"analysis_level must be one of {valid}, got '{analysis_level}'"
        )

    levels = ["layer", "head"] if analysis_level == "both" else [analysis_level]
    return {
        lvl: get_logit_result_path(
            relation_name=relation_name,
            level=lvl,
            relation_type=relation_type,
            base_dir=base_dir,
            ensure_parent=ensure_parent,
        )
        for lvl in levels
    }


def get_svg_path(
    base_relation: str,
    other_relation: Optional[str | list[str]] = None,
    set_operation_mode: Optional[str] = None,
    topn: Optional[int] = None,
    perf_percent: Optional[float] = None,
    base_dir: Union[str, Path] = SVG_DIR,
) -> Path:
    """
    SVG ファイルのパスを生成する関数.

    Args:
        base_relation (str): 基本となる Relation 名.
        other_relation (str | list[str], optional): 他の Relation 名または Relation 名のリスト (default: None).
        set_operation_mode (str, optional): 集合演算の種類 (default: None).
            - "union" (和集合)
            - "intersection" (積集合)
            - "difference" (差集合)
            - "weighted_difference" (重み付き差集合)
        topn (int, optional): トップ N 要素数 (default: None).
        perf_percent (float, optional): パフォーマンスの閾値. 0.0-1.0 の場合は自動的に 0-100 に変換される (default: None).
        base_dir (str): SVG ファイルのベースディレクトリ.

    Returns:
        Path: 生成された SVG ファイルのパス.

    Raises:
        ValueError: topn と perf_percent の両方が指定された場合, またはどちらも指定されなかった場合.
    """
    if topn is not None and perf_percent is not None:
        raise ValueError("topn and perf_percent cannot both be specified.")
    if topn is None and perf_percent is None:
        raise ValueError("Either topn or perf_percent must be specified.")

    # other_relation が str の場合はリストに変換
    if isinstance(other_relation, str):
        other_relation = [other_relation]

    # perf_percent が 0.0-1.0 の範囲の場合は自動的にパーセント表記に変換
    if perf_percent is not None and 0.0 <= perf_percent <= 1.0:
        perf_percent = int(perf_percent * 100)

    # perf_percent が指定されている場合は性能閾値モード, そうでなければ top-n モード
    suffix = f"_p{perf_percent}" if perf_percent is not None else f"_n{topn}"

    # other_relation が指定されている場合は各要素を連結
    if other_relation:
        other_part = "".join(f"_{rel}" for rel in other_relation)
        filename = f"{base_relation}{other_part}{suffix}"
    else:
        filename = f"{base_relation}{suffix}"

    svg_path = Path(base_dir) / (set_operation_mode or "single") / filename
    return svg_path.with_suffix(".svg")


def image_to_base64(image_path: Union[str, Path]) -> Optional[str]:
    """
    画像ファイルを Base64 エンコードされた文字列に変換する関数.

    Args:
        image_path (Union[str, Path]): 画像ファイルのパス.

    Returns:
        Optional[str]: Base64 エンコードされた文字列. ファイルが存在しない場合は None.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


def main():
    # テスト用の関数
    try:
        path = get_cache_path("city_in_country_3")
        print(f"Cache file path: {path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
