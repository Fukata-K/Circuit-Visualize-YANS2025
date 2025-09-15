import pickle

from paths import LOGITS_DIR, get_logit_result_paths


def load_logit_analysis_results(
    relation_name: str,
    analysis_level: str = "both",
    relation_type: str | None = None,
    base_dir: str = str(LOGITS_DIR),
) -> dict[str, dict[str, dict[str, float]]]:
    """
    save_logit_analysis_results 関数で保存された pickle ファイルから logit 分析結果を読み込む関数.

    指定されたディレクトリから層レベルまたは Head レベル (あるいは両方) の
    予測性能データを読み込み, 構造化された辞書として返す.

    Args:
        relation_name (str): Relation 名 (例: "city_in_country", "company_hq")
        analysis_level (str, optional): 読み込むデータのレベル.
            "layer": 層レベルのみ, "head": Headレベルのみ, "both": 両方. Defaults to "both".
        relation_type (str | None): Relation タイプ (例: "factual", "commonsense"). Defaults to None.
        base_dir (str): 結果が保存されているベースディレクトリパス. Defaults to LOGITS_DIR.

    Returns:
        dict[str, dict[str, dict[str, float]]]: 読み込まれた分析結果
            - "layer" が含まれる場合: {"layer": {"ranks": {...}, "probs": {...}}}
            - "head" が含まれる場合: {"head": {"ranks": {...}, "probs": {...}}}
            - "both" の場合: 上記両方を含む辞書

    Raises:
        FileNotFoundError: 指定されたファイルが見つからない場合
        ValueError: analysis_level が無効な値の場合
        pickle.UnpicklingError: pickle ファイルの読み込みに失敗した場合

    Examples:
        >>> # 層レベルと Head レベル両方のデータを読み込み
        >>> results = load_logit_analysis_results(
        ...     relation_name="city_in_country",
        ...     relation_type="factual"
        ... )
        >>> print(f"Layer m0 average rank: {results['layer']['ranks']['m0']:.2f}")
        >>> print(f"Head a0.h0 average rank: {results['head']['ranks']['a0.h0']:.2f}")

        >>> # 層レベルのデータのみを読み込み
        >>> layer_results = load_logit_analysis_results(
        ...     relation_name="city_in_country",
        ...     analysis_level="layer",
        ...     relation_type="factual"
        ... )
        >>> print(f"Layer ranks: {layer_results['layer']['ranks']}")

        >>> # Head レベルのデータのみを読み込み
        >>> head_results = load_logit_analysis_results(
        ...     relation_name="city_in_country",
        ...     analysis_level="head",
        ...     relation_type="factual"
        ... )
        >>> print(f"Head probs: {head_results['head']['probs']}")

    Note:
        読み込み対象のファイル構造:
        - Layer レベル: {base_dir}/layer/{relation_type}/{relation_name}.pkl
        - Head レベル: {base_dir}/head/{relation_type}/{relation_name}.pkl

        各ファイルは以下の形式でデータが保存されている:
        - {"ranks": {...}, "probs": {...}}

        返される辞書の構造:
        - Layer データ: ranks は {"m0": float, "m1": float, ...}
        - Head データ: ranks は {"a0.h0": float, "a0.h1": float, ...}
        - probs も同様の構造
    """
    # analysis_level の妥当性チェック
    valid_levels = {"layer", "head", "both"}
    if analysis_level not in valid_levels:
        raise ValueError(
            f"analysis_level must be one of {valid_levels}, got '{analysis_level}'"
        )

    results = {}

    paths = get_logit_result_paths(
        relation_name=relation_name,
        analysis_level=analysis_level,
        relation_type=relation_type,
        base_dir=base_dir,
    )

    # 層レベルのデータを読み込み
    if "layer" in paths:
        layer_path = paths["layer"]

        if not layer_path.exists():
            raise FileNotFoundError(f"Layer results file not found: {layer_path}")

        print(f"Loading layer results from: {layer_path}")

        try:
            with open(layer_path, "rb") as f:
                layer_data = pickle.load(f)
            results["layer"] = layer_data
            print(f"Layer results loaded: {len(layer_data['ranks'])} layers")
        except Exception as e:
            raise pickle.UnpicklingError(
                f"Failed to load layer results from {layer_path}: {e}"
            ) from e

    # Head レベルのデータを読み込み
    if "head" in paths:
        head_path = paths["head"]

        if not head_path.exists():
            raise FileNotFoundError(f"Head results file not found: {head_path}")

        print(f"Loading head results from: {head_path}")

        try:
            with open(head_path, "rb") as f:
                head_data = pickle.load(f)
            results["head"] = head_data
            print(f"Head results loaded: {len(head_data['ranks'])} heads")
        except Exception as e:
            raise pickle.UnpicklingError(
                f"Failed to load head results from {head_path}: {e}"
            ) from e

    print(f"Logit analysis results loaded for {relation_type}/{relation_name}")
    return results
