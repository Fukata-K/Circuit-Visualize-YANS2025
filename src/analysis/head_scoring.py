import pickle
from pathlib import Path
from typing import Callable, Literal, Union

import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from transformer_lens import ActivationCache, HookedTransformer

from analysis.analysis_utils import get_attention_pattern, get_substring_token_span
from paths import get_head_score_path


def convert_score(value: float | None, metric: str) -> float | None:
    """
    指定したスコア指標に応じて値を正規化・変換する関数.

    - "pearson": 負の値を 0 に丸める (正の相関のみ可視化)
    - "auc": 0.5 を基準に 0-1 の範囲に変換 (0.5 以下は 0, 0.5 より大きい場合は 0-1 に正規化)
    - other: 0-1 にクリップして返す

    Args:
        value (float | None): 元のスコア値
        metric (str): "pearson", "auc" などの指標名

    Returns:
        float | None: 変換後のスコア値 (数値を返す場合は必ず 0-1 の範囲に収まる)
    """
    if value is None:
        return None
    if metric == "pearson":
        return max(0, value)
    elif metric == "auc":
        return max(0, value - 0.5) * 2
    else:
        return min(1.0, max(0.0, value))


def flatten_lower_triangle_with_diag_labels(
    attn: torch.Tensor,
) -> tuple[list[float], list[int]]:
    """
    下三角成分 (対角含む) を flat なリストにし, 各要素が対角成分かどうかのラベル (1:対角, 0:非対角) も同じ順序で返す関数.

    Args:
        attn (torch.Tensor): [seq_len, seq_len] 形式の Attention Pattern

    Returns:
        tuple[list[float], list[int]]:
            - 下三角成分 (行優先でフラット化) の値リスト
            - 同じ順序でのラベルリスト (1:対角, 0:非対角)
    """
    seq_len = attn.size(0)
    values = []
    labels = []
    for i in range(seq_len):
        for j in range(i + 1):
            values.append(attn[i, j].item())
            labels.append(1 if i == j else 0)
    return values, labels


def calc_corr_auc(
    values: list[float], labels: list[int]
) -> tuple[float | None, float | None]:
    """
    値リストとラベルリストから Pearson 相関係数と ROC AUC スコアを計算して返す関数.

    Args:
        values (list[float]): 予測値のリスト
        labels (list[int])  : 2値ラベルのリスト (1:正例, 0:負例)

    Returns:
        tuple[float | None, float | None]:
            (Pearson 相関係数, ROC AUC スコア)
            計算できない場合は None
    """
    try:
        pearson_corr, _ = pearsonr(values, labels)
    except Exception:
        pearson_corr = None
    try:
        auc_score = roc_auc_score(labels, values)
    except Exception:
        auc_score = None
    return pearson_corr, auc_score  # type: ignore


def reduce_attention_by_span_lastrow(
    attn_pattern: torch.Tensor, span: tuple[int, int], mode: str = "last"
) -> tuple[list[float], list[int]]:
    """
    [seq_len, seq_len] 形式の Attention Pattern の最終行に対してスパン処理を行い,
    1行分の Attention 値リストとラベルリストを返す関数.

    ラベルは end 位置のみ 1, それ以外は 0 (mode = "last"/"sum"/"mean"),
    またはスパン内全て 1 (mode = "all") とする．

    Args:
        attn_pattern (torch.Tensor): [seq_len, seq_len] 形式の Attention Pattern
        span (tuple[int, int]): (start, end) スパン (両端含む, トークンインデックス)
        mode (str):
            "last" ... スパン内の最後 (end 位置) の値のみ残す
            "all"  ... スパン内は全て 1ラベル, 値はそのまま
            "sum"  ... スパン内の合計値で代表
            "mean" ... スパン内の平均値で代表

    Returns:
        tuple[list[float], list[int]]: 新しい Attention 値リストとラベルリスト (長さ seq_len)
    """
    seq_len = attn_pattern.size(1)
    start, end = span
    attn = attn_pattern[-1].tolist()  # 最終行

    if mode == "last":
        new_attn = attn[:start] + [attn[end]] + attn[end + 1 :]
        new_label = [0] * start + [1] + [0] * (seq_len - end - 1)
    elif mode == "all":
        new_attn = attn.copy()
        new_label = [1 if start <= i <= end else 0 for i in range(seq_len)]
    elif mode == "sum":
        span_sum = sum(attn[start : end + 1])
        new_attn = attn[:start] + [span_sum] + attn[end + 1 :]
        new_label = [0] * start + [1] + [0] * (seq_len - end - 1)
    elif mode == "mean":
        span_vals = attn[start : end + 1]
        span_mean = sum(span_vals) / len(span_vals) if len(span_vals) > 0 else 0.0
        new_attn = attn[:start] + [span_mean] + attn[end + 1 :]
        new_label = [0] * start + [1] + [0] * (seq_len - end - 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return new_attn, new_label


def collect_head_attention_and_labels(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    df: pd.DataFrame,
    prompt_col: str = "clean",
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """
    各 Attention Head ごとに全データ・全トークンの Attention 値とラベルを収集する共通関数.

    Returns:
        tuple[dict[str, list[float]], dict[str, list[int]]]:
            (Attention Head ごとの値リスト, Attention Head ごとのラベルリスト)
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    head_names = [
        f"a{layer}.h{head}" for layer in range(num_layers) for head in range(num_heads)
    ]
    heads_attn = {head_name: [] for head_name in head_names}
    heads_label = {head_name: [] for head_name in head_names}

    for data_idx in range(len(df)):
        prompt = df.iloc[data_idx][prompt_col]
        seq_len = len(model.to_tokens(prompt, prepend_bos=False)[0])
        for layer in range(num_layers):
            for head in range(num_heads):
                head_name = f"a{layer}.h{head}"
                attn_pattern = get_attention_pattern(
                    cache, data_idx, layer, head, seq_len=seq_len
                )
                values, labels = flatten_lower_triangle_with_diag_labels(attn_pattern)
                heads_attn[head_name].extend(values)
                heads_label[head_name].extend(labels)
    return heads_attn, heads_label


def compute_head_self_attention_score(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    df: pd.DataFrame,
    metric: str = "mean",
    prompt_col: str = "clean",
) -> dict[str, float | None]:
    """
    各 Attention Head ごとに Self-Attention の度合いを指定した指標で計算して返す関数.
    この関数で返されるスコアは 0-1 の範囲に正規化されている.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): Attention Pattern のキャッシュ
        df (pd.DataFrame): データセット (各行にプロンプト列が必要)
        metric (str): "mean", "pearson", "auc" のいずれか
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対角成分とラベルの Pearson 相関係数を convert_score で変換した値
            - "auc"    : 対角成分とラベルの ROC AUC スコアを convert_score で変換した値
        prompt_col (str): プロンプトが格納されているカラム名 (default: "clean")

    Returns:
        dict[str, float | None]: {head_name: 指定指標の値, ...}
    """
    assert metric in {"mean", "pearson", "auc"}, (
        "metric must be 'mean', 'pearson', or 'auc'"
    )

    heads_attn, heads_label = collect_head_attention_and_labels(
        model, cache, df, prompt_col
    )
    result = {}
    for head_name in heads_attn:
        if metric == "mean":
            # 対角成分のみ抽出して平均
            diag_values = [
                v
                for v, label in zip(heads_attn[head_name], heads_label[head_name])
                if label == 1
            ]
            score = sum(diag_values) / len(diag_values) if diag_values else float("nan")
        elif metric in {"pearson", "auc"}:
            pearson, auc = calc_corr_auc(heads_attn[head_name], heads_label[head_name])
            score = pearson if metric == "pearson" else auc
        else:
            score = None
        result[head_name] = convert_score(score, metric)
    return result


def compute_head_span_score(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    df: pd.DataFrame,
    span_getter: Callable[[HookedTransformer, pd.Series], tuple[int, int]],
    metric: str = "pearson",
    prompt_col: str = "clean",
    mode: str = "last",
) -> dict[str, float | None]:
    """
    各 Attention Head ごとに, 指定したスパン (Subject / Relation 等) に対するスコアを計算して返す共通関数.
    この関数で返されるスコアは 0-1 の範囲に正規化されている.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): Attention Pattern のキャッシュ
        df (pd.DataFrame): データセット
        span_getter (callable): (model, row) -> (start, end) なスパン取得関数
        metric (str): "mean", "pearson", "auc" のいずれか
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対角成分とラベルの Pearson 相関係数を convert_score で変換した値
            - "auc"    : 対角成分とラベルの ROC AUC スコアを convert_score で変換した値
        prompt_col (str): プロンプトが格納されているカラム名
        mode (str): reduce_attention_by_span_lastrow の mode

    Returns:
        dict[str, float | None]: {head_name: 指定指標の値, ...}
    """
    assert metric in {"mean", "pearson", "auc"}, (
        "metric must be 'mean', 'pearson', or 'auc'"
    )

    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    head_names = [
        f"a{layer}.h{head}" for layer in range(num_layers) for head in range(num_heads)
    ]
    heads_attn = {head_name: [] for head_name in head_names}
    heads_label = {head_name: [] for head_name in head_names}

    for idx, row in df.iterrows():
        prompt = row[prompt_col]
        seq_len = len(model.to_tokens(prompt, prepend_bos=False)[0])
        try:
            span = span_getter(model, row)
        except Exception as e:
            print(f"Warning: span_getter failed at idx={idx}, prompt='{prompt}': {e}")
            continue
        if span[1] >= seq_len:
            print(
                f"Warning: span out of range at idx={idx}, span={span}, seq_len={seq_len}"
            )
            continue

        data_idx = int(idx) if isinstance(idx, (int, float)) else 0
        for layer in range(num_layers):
            for head in range(num_heads):
                head_name = f"a{layer}.h{head}"
                attn_pattern = get_attention_pattern(
                    cache, data_idx, layer, head, seq_len=seq_len
                )
                values, labels = reduce_attention_by_span_lastrow(
                    attn_pattern, span, mode=mode
                )
                heads_attn[head_name].extend(values)
                heads_label[head_name].extend(labels)

    result = {}
    for head_name in head_names:
        if metric == "mean":
            target_values = [
                v
                for v, label in zip(heads_attn[head_name], heads_label[head_name])
                if label == 1
            ]
            score = (
                sum(target_values) / len(target_values)
                if target_values
                else float("nan")
            )
        elif metric in {"pearson", "auc"}:
            pearson, auc = calc_corr_auc(heads_attn[head_name], heads_label[head_name])
            score = pearson if metric == "pearson" else auc
        else:
            score = None
        result[head_name] = convert_score(score, metric)
    return result


def subject_span_getter(model: HookedTransformer, row: pd.Series) -> tuple[int, int]:
    """
    DataFrame の 1行から Subject スパンを取得する関数

    Args:
        model (HookedTransformer): トークナイズ用モデル
        row (pd.Series): データフレームの1行

    Returns:
        tuple[int, int]: Subject のトークンスパン
    """
    return get_substring_token_span(model, row["clean"], row["subject"])


def relation_span_getter(model: HookedTransformer, row: pd.Series) -> tuple[int, int]:
    """
    DataFrame の 1行から Relation スパンを取得する関数

    Args:
        model (HookedTransformer): トークナイズ用モデル
        row (pd.Series): データフレームの1行

    Returns:
        tuple[int, int]: Relation のトークンスパン

    Notes:
        ここでは "Prompt の Subject 以降" を Relation とみなす
        データセットの形式が変わった場合にはこの関数も変更が必要
    """
    prompt = row["clean"]
    subject = row["subject"]
    relation = prompt.split(subject)[-1].strip()
    return get_substring_token_span(model, prompt, relation)


def compute_head_subject_score(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    df: pd.DataFrame,
    metric: str = "pearson",
    prompt_col: str = "clean",
    mode: str = "last",
) -> dict[str, float | None]:
    """
    各 Attention Head ごとに Subject スパンに対するスコアを計算して返す関数.
    この関数で返されるスコアは 0-1 の範囲に正規化されている.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): Attention Pattern のキャッシュ
        df (pd.DataFrame): データセット
        metric (str): "mean", "pearson", "auc" のいずれか
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対角成分とラベルの Pearson 相関係数を convert_score で変換した値
            - "auc"    : 対角成分とラベルの ROC AUC スコアを convert_score で変換した値
        prompt_col (str): プロンプトが格納されているカラム名
        mode (str): reduce_attention_by_span_lastrow の mode

    Returns:
        dict[str, float | None]: {head_name: 指定指標の値, ...}
    """
    return compute_head_span_score(
        model,
        cache,
        df,
        subject_span_getter,
        metric=metric,
        prompt_col=prompt_col,
        mode=mode,
    )


def compute_head_relation_score(
    model: HookedTransformer,
    cache: dict | ActivationCache,
    df: pd.DataFrame,
    metric: str = "pearson",
    prompt_col: str = "clean",
    mode: str = "last",
) -> dict[str, float | None]:
    """
    各 Attention Head ごとに Relation スパンに対するスコアを計算して返す関数.
    この関数で返されるスコアは 0-1 の範囲に正規化されている.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        cache (dict | ActivationCache): Attention Pattern のキャッシュ
        df (pd.DataFrame): データセット
        metric (str): "mean", "pearson", "auc" のいずれか
            - "mean"   : 対象部分に対する平均 Attention 値
            - "pearson": 対角成分とラベルの Pearson 相関係数を convert_score で変換した値
            - "auc"    : 対角成分とラベルの ROC AUC スコアを convert_score で変換した値
        prompt_col (str): プロンプトが格納されているカラム名
        mode (str): reduce_attention_by_span_lastrow の mode

    Returns:
        dict[str, float | None]: {head_name: 指定指標の値, ...}
    """
    return compute_head_span_score(
        model,
        cache,
        df,
        relation_span_getter,
        metric=metric,
        prompt_col=prompt_col,
        mode=mode,
    )


def load_head_scores(
    scores_dir: Union[str, Path],
    score_type: str,
    metric: str,
    relation_name: str,
) -> dict[str, float | None] | None:
    """
    保存されたヘッドスコアを pickle ファイルから読み込む関数.

    Args:
        scores_dir (str): スコアファイルが保存されているディレクトリ
        score_type (str): スコアタイプ ("self_scores", "subject_scores", "relation_scores")
        metric (str): スコアの指標名 (例: "pearson", "auc")
        relation_name (str): Relation 名 (例: "city_in_country")

    Returns:
        dict[str, float | None]: {head_name: score, ...} の辞書
        パスが存在しない場合は None を返す

    Examples:
        >>> self_scores = load_head_scores("out/head_scores", "self_scores", "pearson", "city_in_country")
        >>> subject_scores = load_head_scores("out/head_scores", "subject_scores", "pearson", "city_in_country")
    """
    if any(x is None for x in [scores_dir, score_type, metric, relation_name]):
        return None
    scores_path = get_head_score_path(
        relation_name, score_type, metric, base_dir=scores_dir
    )

    if not scores_path.exists():
        return None

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    return scores


def get_head_scores(
    scores_dir: Union[str, Path],
    score_type: Literal["self_scores", "subject_scores", "relation_scores"],
    metric: Literal["mean", "pearson", "auc"],
    relation_name: str,
    relation_type: str | None = None,
    model: HookedTransformer | None = None,
    cache: dict | ActivationCache | None = None,
    df: pd.DataFrame | None = None,
    prompt_col: str = "clean",
    save_scores: bool = False,
) -> dict[str, float | None]:
    """
    保存されたヘッドスコアを取得する関数. 該当するスコアが存在しない場合は計算を行なう.

    Args:
        scores_dir (str): スコアファイルが保存されているディレクトリ
        score_type (str): スコアタイプ ("self_scores", "subject_scores", "relation_scores")
        metric (str): スコアの指標名 (例: "pearson", "auc")
        relation_name (str): Relation 名 (例: "city_in_country")
        relation_type (str | None): Relation タイプ (例: "factual", "commonsense")
        model (HookedTransformer | None): トークナイズ用モデル (スコアが存在しない場合に必要)
        cache (dict | ActivationCache | None): Attention Pattern のキャッシュ (スコアが存在しない場合に必要)
        df (pd.DataFrame | None): データセット (スコアが存在しない場合に必要)
        prompt_col (str): プロンプトが格納されているカラム名 (default: "clean")
        save_scores (bool): スコアを計算した場合に保存するかどうか (default: False)

    Returns:
        dict[str, float | None]: {head_name: score, ...} の辞書
    """
    scores = load_head_scores(scores_dir, score_type, metric, relation_name)
    if scores is not None:
        return scores

    if model is None or cache is None or df is None:
        raise ValueError(
            "model, cache, and df must be provided if scores are not already computed"
        )

    if score_type == "self_scores":
        scores = compute_head_self_attention_score(
            model, cache, df, metric=metric, prompt_col=prompt_col
        )
    elif score_type == "subject_scores":
        scores = compute_head_subject_score(
            model, cache, df, metric=metric, prompt_col=prompt_col
        )
    elif score_type == "relation_scores":
        scores = compute_head_relation_score(
            model, cache, df, metric=metric, prompt_col=prompt_col
        )
    else:
        raise ValueError(f"Unknown score type: {score_type}")

    if save_scores:
        score_path = get_head_score_path(
            relation_name,
            score_type,
            metric,
            relation_type=relation_type,
            base_dir=scores_dir,
            ensure_parent=True,
        )
        with open(score_path, "wb") as f:
            pickle.dump(scores, f)
        print(f"Scores saved to: {score_path}")

    return scores


def main(
    model_name: str = "gpt2-small",
    cache_dir: str = "out/cache",
    dataset_dir: str = "data/filtered_gpt2_small",
    output_dir: str = "out/head_scores",
    prompt_col: str = "clean",
    metric: Literal["mean", "pearson", "auc"] = "pearson",
):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルの設定
    model = HookedTransformer.from_pretrained(model_name, device=device)

    # ディレクトリ内の全ての CSV ファイルを取得
    dataset_csv_files_dir = Path(dataset_dir)
    dataset_csv_files = sorted(list(dataset_csv_files_dir.glob("**/*.csv")))
    print(f"Found {len(dataset_csv_files)} dataset files.")

    for dataset_path in dataset_csv_files:
        # データセットの読み込み
        df = pd.read_csv(dataset_path)

        # データセット名と種類を取得
        p = Path(dataset_path)
        relation_type = p.parent.name
        relation_name = p.stem

        # 観察したい cache の読み込み (glob でファイルを検索)
        cache_pattern = Path(cache_dir) / relation_type / f"{relation_name}*.pt"
        cache_files = list(cache_pattern.parent.glob(cache_pattern.name))

        if not cache_files:
            print(f"Warning: No cache files found for pattern: {cache_pattern}")
            continue

        # 複数のファイルがある場合は最初のファイルを使用 (要改善: 複数ファイルがある場合の適切な処理を加える必要あり)
        cache_path = cache_files[0]
        if len(cache_files) > 1:
            print(f"Info: Multiple cache files found, using: {cache_path}")

        cache = torch.load(cache_path, map_location=device, weights_only=False)

        # 各スコアを別々の pickle ファイルとして保存
        for score_type in ["self_scores", "subject_scores", "relation_scores"]:
            get_head_scores(
                scores_dir=output_dir,
                score_type=score_type,  # type: ignore
                metric=metric,
                relation_type=relation_type,
                relation_name=relation_name,
                model=model,
                cache=cache,
                df=df,
                prompt_col=prompt_col,
                save_scores=True,  # スコアを保存する
            )


if __name__ == "__main__":
    main(
        model_name="gpt2-small",
        cache_dir="out/cache",
        dataset_dir="data/filtered_gpt2_small",
        output_dir="out/head_scores",
        prompt_col="clean",
        metric="pearson",
    )
