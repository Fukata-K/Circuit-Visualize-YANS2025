import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
from transformer_lens import HookedTransformer

from paths import LOGITS_DIR, get_logit_result_paths


def _validate_indices(
    model: HookedTransformer,
    data_idx: int,
    seq_len: int,
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> None:
    """
    引数の妥当性を検証する内部関数.

    Args:
        model (HookedTransformer): HookedTransformer モデル
        data_idx (int): データインデックス
        seq_len (int): シーケンス長
        layer (Optional[int]): レイヤー番号 (オプション)
        head (Optional[int]): ヘッド番号 (オプション)

    Raises:
        ValueError: 引数が無効な場合
    """
    if data_idx < 0:
        raise ValueError(f"data_idx must be non-negative, got {data_idx}")

    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")

    if layer is not None:
        if not (0 <= layer < model.cfg.n_layers):
            raise ValueError(
                f"layer must be in range [0, {model.cfg.n_layers - 1}], got {layer}"
            )

    if head is not None:
        if not (0 <= head < model.cfg.n_heads):
            raise ValueError(
                f"head must be in range [0, {model.cfg.n_heads - 1}], got {head}"
            )

    # layer と head の組み合わせを検証
    if layer is None and head is not None:
        raise ValueError(
            "If 'head' is specified, 'layer' must also be specified. "
            "To analyze a specific head, please specify both 'layer' and 'head'."
        )


def compute_layer_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
    layer: Optional[int] = None,
) -> torch.Tensor:
    """
    各 layer の出力を logit へ変換する関数.

    引数の組み合わせによって自動的に最適な計算方法を選択する.
    - layer 指定: 層レベル分析
    - 未指定: 全体分析

    Note: この関数は各層の出力に Final Layer Normalization を適用してから Unembedding を行う.
    これにより実際の GPT-2 モデルの処理フローと一致する正確な寄与度が計算される.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)
        layer (Optional[int], optional): 分析対象のレイヤー番号 (0 から model.cfg.n_layers-1). Defaults to None.

    Returns:
        torch.Tensor:
            - 単一層分析: [d_vocab]
            - 全体分析: [n_layers, d_vocab]

    Raises:
        ValueError: 引数が無効な場合
        KeyError: 必要なキャッシュキーが存在しない場合
    """
    # 引数の妥当性を検証
    _validate_indices(model, data_idx, seq_len, layer)

    with torch.no_grad():
        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        if layer is not None:
            # 指定された Layer の出力 ([d_model]) を取得
            cache_key = f"blocks.{layer}.hook_resid_post"
            if cache_key not in cache:
                raise KeyError(f"Cache key '{cache_key}' not found")

            layer_out = cache[cache_key][data_idx, seq_len - 1]

            # Final Layer Normalization を適用してから Unembedding する
            # これにより実際のモデルの処理フローと一致する
            normalized_layer_out = model.ln_final(layer_out.unsqueeze(0)).squeeze(0)

            # 最終的な logit ([d_vocab]) を計算
            logit = normalized_layer_out @ W_U
            if b_U is not None:
                logit = logit + b_U

            return logit

        else:
            n_layers = model.cfg.n_layers

            # 全ての層の出力を取得してスタック
            layer_out_list = []
            for layer_idx in range(n_layers):
                cache_key = f"blocks.{layer_idx}.hook_resid_post"
                if cache_key not in cache:
                    raise KeyError(f"Cache key '{cache_key}' not found")

                layer_out = cache[cache_key][data_idx, seq_len - 1]
                layer_out_list.append(layer_out)

            # layer_out を [n_layers, d_model] の形にスタック
            layer_out = torch.stack(layer_out_list, dim=0)

            # 各出力に Final Layer Normalization を適用
            # バッチ処理のために次元を調整: [n_layers, d_model] -> [n_layers, 1, d_model]
            layer_out_batched = layer_out.unsqueeze(1)
            normalized_layer_out_list = []
            for layer_idx in range(n_layers):
                normalized = model.ln_final(layer_out_batched[layer_idx]).squeeze(0)
                normalized_layer_out_list.append(normalized)
            normalized_layer_out = torch.stack(normalized_layer_out_list, dim=0)

            # einsum を使用して効率的に計算: normalized_layer_out @ W_U
            # normalized_layer_out: [n_layers, d_model], W_U: [d_model, d_vocab]
            # 結果: [n_layers, d_vocab]
            logits = torch.einsum("lm,mv->lv", normalized_layer_out, W_U)

            # バイアス項が存在する場合は各層に追加
            if b_U is not None:
                logits = logits + b_U.unsqueeze(0)

            return logits


def compute_mlp_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
    layer: Optional[int] = None,
) -> torch.Tensor:
    """
    MLP の logit への寄与を計算する関数.

    引数の組み合わせによって自動的に最適な計算方法を選択する.
    - layer 指定: 層レベル分析
    - 未指定: 全体分析

    Note: この関数は MLP 出力に Final Layer Normalization を適用してから Unembedding を行う.
    これにより実際の GPT-2 モデルの処理フローと一致する正確な寄与度が計算される.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)
        layer (Optional[int], optional): 分析対象のレイヤー番号 (0 から model.cfg.n_layers-1). Defaults to None.

    Returns:
        torch.Tensor:
            - 単一 MLP 分析: [d_vocab]
            - 全体分析: [n_layers, d_vocab]

    Raises:
        ValueError: 引数が無効な場合
        KeyError: 必要なキャッシュキーが存在しない場合
    """
    # 引数の妥当性を検証
    _validate_indices(model, data_idx, seq_len, layer)

    with torch.no_grad():
        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        if layer is not None:
            # 指定された Layer の MLP 出力 ([d_model]) を取得
            cache_key = f"blocks.{layer}.hook_mlp_out"
            if cache_key not in cache:
                raise KeyError(f"Cache key '{cache_key}' not found")

            mlp_out = cache[cache_key][data_idx, seq_len - 1]

            # Final Layer Normalization を適用してから Unembedding する
            # これにより実際のモデルの処理フローと一致する
            normalized_mlp_out = model.ln_final(mlp_out.unsqueeze(0)).squeeze(0)

            # 最終的な logit ([d_vocab]) を計算
            logit = normalized_mlp_out @ W_U
            if b_U is not None:
                logit = logit + b_U

            return logit

        else:
            n_layers = model.cfg.n_layers

            # 全ての層の MLP 出力を取得してスタック
            mlp_out_list = []
            for layer_idx in range(n_layers):
                cache_key = f"blocks.{layer_idx}.hook_mlp_out"
                if cache_key not in cache:
                    raise KeyError(f"Cache key '{cache_key}' not found")

                mlp_out = cache[cache_key][data_idx, seq_len - 1]
                mlp_out_list.append(mlp_out)

            # mlp_out を [n_layers, d_model] の形にスタック
            mlp_out = torch.stack(mlp_out_list, dim=0)

            # 各 MLP 出力に Final Layer Normalization を適用
            # バッチ処理のために次元を調整: [n_layers, d_model] -> [n_layers, 1, d_model]
            mlp_out_batched = mlp_out.unsqueeze(1)
            normalized_mlp_out_list = []
            for layer_idx in range(n_layers):
                normalized = model.ln_final(mlp_out_batched[layer_idx]).squeeze(0)
                normalized_mlp_out_list.append(normalized)
            normalized_mlp_out = torch.stack(normalized_mlp_out_list, dim=0)

            # einsum を使用して効率的に計算: normalized_mlp_out @ W_U
            # normalized_mlp_out: [n_layers, d_model], W_U: [d_model, d_vocab]
            # 結果: [n_layers, d_vocab]
            logits = torch.einsum("lm,mv->lv", normalized_mlp_out, W_U)

            # バイアス項が存在する場合は各層に追加
            if b_U is not None:
                logits = logits + b_U.unsqueeze(0)

            return logits


def compute_head_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> torch.Tensor:
    """
    Attention Head の logit への寄与を計算する統合関数.

    引数の組み合わせによって自動的に最適な計算方法を選択する.
    - layer, head 両方指定: 単一 Head 分析
    - layer のみ指定: 層レベル分析
    - どちらも未指定: 全体俯瞰分析

    Note: この関数は Attention Head 出力に Final Layer Normalization を適用してから Unembedding を行う.
    これにより実際の GPT-2 モデルの処理フローと一致する正確な寄与度が計算される.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)
        layer (Optional[int], optional): 分析対象のレイヤー番号 (0 から model.cfg.n_layers-1). Defaults to None.
        head (Optional[int], optional): 分析対象の Head 番号 (0 から model.cfg.n_heads-1). Defaults to None.

    Returns:
        torch.Tensor:
            - 単一 Head 分析: [d_vocab]
            - 層レベル分析: [n_heads, d_vocab]
            - 全体俯瞰分析: [n_layers, n_heads, d_vocab]

    Raises:
        ValueError: 引数が無効な場合

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> _, cache = model.run_with_cache("The capital of France is")

        # 単一 Head 分析
        >>> logit = compute_head_logit_contribution(model, cache, 0, 5, layer=0, head=0)
        >>> print(f"Shape: {logit.shape}")  # [d_vocab]

        # 層レベル分析
        >>> logits = compute_head_logit_contribution(model, cache, 0, 5, layer=0)
        >>> print(f"Shape: {logits.shape}")  # [n_heads, d_vocab]

        # 全体俯瞰分析
        >>> logits = compute_head_logit_contribution(model, cache, 0, 5)
        >>> print(f"Shape: {logits.shape}")  # [n_layers, n_heads, d_vocab]
    """
    # 引数の妥当性を検証
    _validate_indices(model, data_idx, seq_len, layer, head)

    # 引数の組み合わせに応じて適切な関数を呼び出し
    if layer is not None and head is not None:
        # 単一 Head 分析
        return compute_single_head_logit_contribution(
            model, cache, data_idx, seq_len, layer, head
        )
    elif layer is not None:
        # 層レベル分析
        return compute_layer_heads_logit_contribution(
            model, cache, data_idx, seq_len, layer
        )
    else:
        # 全体俯瞰分析
        return compute_all_heads_logit_contribution(model, cache, data_idx, seq_len)


def compute_single_head_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
    layer: int,
    head: int,
) -> torch.Tensor:
    """
    単一の Attention Head の logit への寄与を計算する.

    指定された単一の Attention Head が最終的な語彙予測にどの程度貢献するかを計算する.

    処理範囲: 1つの Head
    最適用途: 特定の Head の詳細分析

    計算の流れ:
    1. 指定された Head の Attention 出力 z を取得
    2. その Head の出力重み W_O を適用
    3. Unembedding 行列 W_U を適用して logit を計算

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)
        layer (int): 分析対象のレイヤー番号 (0 から model.cfg.n_layers-1)
        head (int): 分析対象の Head 番号 (0 から model.cfg.n_heads-1)

    Returns:
        torch.Tensor: 語彙サイズの logit ベクトル [d_vocab]
    """
    with torch.no_grad():
        # キャッシュキーの存在確認
        cache_key = f"blocks.{layer}.attn.hook_z"
        if cache_key not in cache:
            raise KeyError(f"Cache key '{cache_key}' not found")

        # 指定された Head の Attention 出力 ([d_head]) を取得
        z = cache[cache_key][data_idx, seq_len - 1, head]

        # その Head の出力重み行列 ([d_head, d_model]) を取得
        W_O = model.W_O[layer][head]

        # Head の出力を残差ストリーム次元 ([d_model]) に変換
        out = z @ W_O

        # Final Layer Normalization を適用してから Unembedding する
        # これにより実際のモデルの処理フローと一致する
        normalized_out = model.ln_final(out.unsqueeze(0)).squeeze(0)

        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        # 最終的な logit ([d_vocab]) を計算
        logit = normalized_out @ W_U
        if b_U is not None:
            logit = logit + b_U

        return logit


def compute_layer_heads_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
    layer: int,
) -> torch.Tensor:
    """
    指定した層の全ての Head の logit への寄与を計算する.

    指定された層の全ての Attention Head が最終的な語彙予測にどの程度貢献するかを層レベルで一度に効率的に計算する.

    処理範囲: 1つの層の全ての Head
    最適用途: 特定の層の Head 間比較, 中程度の並列処理

    計算の流れ:
    1. 指定された層の全ての Head の Attention 出力 z を取得
    2. 全ての Head の出力重み W_O を一度に適用
    3. Unembedding 行列 W_U を適用して logits を計算

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)
        layer (int): 分析対象のレイヤー番号 (0 から model.cfg.n_layers-1)

    Returns:
        torch.Tensor: 各 Head の logits 行列 [n_heads, d_vocab]
    """
    with torch.no_grad():
        # キャッシュキーの存在確認
        cache_key = f"blocks.{layer}.attn.hook_z"
        if cache_key not in cache:
            raise KeyError(f"Cache key '{cache_key}' not found")

        # 指定された層の全ての Head の Attention 出力 ([n_heads, d_head]) を取得
        z = cache[cache_key][data_idx, seq_len - 1]

        # 出力重み行列 ([n_heads, d_head, d_model]) を取得
        W_O = model.W_O[layer]

        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        # 各 Head の出力を残差ストリーム次元に変換: z @ W_O
        # z: [n_heads, d_head], W_O: [n_heads, d_head, d_model]
        # 結果: [n_heads, d_model]
        head_outputs = torch.einsum("hd,hdm->hm", z, W_O)

        # 各 Head の出力に Final Layer Normalization を適用
        normalized_head_outputs = []
        for head_idx in range(head_outputs.shape[0]):
            normalized = model.ln_final(head_outputs[head_idx].unsqueeze(0)).squeeze(0)
            normalized_head_outputs.append(normalized)
        normalized_head_outputs = torch.stack(normalized_head_outputs, dim=0)

        # einsum を使用して効率的に計算: normalized_head_outputs @ W_U
        # normalized_head_outputs: [n_heads, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_heads, d_vocab]
        logits = torch.einsum("hm,mv->hv", normalized_head_outputs, W_U)

        # バイアス項が存在する場合は各 Head に追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0)

        return logits


def compute_all_heads_logit_contribution(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    data_idx: int,
    seq_len: int,
) -> torch.Tensor:
    """
    モデル内の全ての Head の logit への寄与を計算する.

    モデル内の全ての Attention Head が最終的な語彙予測にどの程度貢献するかを一度に効率的に計算する.

    処理範囲: モデル内の全ての Head
    最適用途: 全体俯瞰, 大規模並列処理, 最大パフォーマンス

    計算の流れ:
    1. モデル内の全ての層の Head の Attention 出力 z を取得してスタック
    2. 全ての Head に出力重み W_O を一度に適用
    3. Unembedding 行列 W_U を適用して logits を計算

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        data_idx (int): バッチ内のデータインデックス
        seq_len (int): シーケンス長 (1 から実際のシーケンス長)

    Returns:
        torch.Tensor: 各 Head の logits tensor [n_layers, n_heads, d_vocab]
    """
    with torch.no_grad():
        n_layers = model.cfg.n_layers

        # 全ての層の Head の Attention 出力を取得してスタック
        z_list = []
        for layer_idx in range(n_layers):
            cache_key = f"blocks.{layer_idx}.attn.hook_z"
            if cache_key not in cache:
                raise KeyError(f"Cache key '{cache_key}' not found")

            z = cache[cache_key][data_idx, seq_len - 1]
            z_list.append(z)

        # z を [n_layers, n_heads, d_head] の形にスタック
        z = torch.stack(z_list, dim=0)

        # 出力重み行列 ([n_layers, n_heads, d_head, d_model]) を取得
        W_O = model.W_O

        # Unembedding 行列とバイアス項を取得
        W_U = model.W_U  # [d_model, d_vocab]
        b_U = getattr(model, "b_U", None)  # [d_vocab] (存在しない場合は None)

        # 各 Head の出力を残差ストリーム次元に変換: z @ W_O
        # z: [n_layers, n_heads, d_head], W_O: [n_layers, n_heads, d_head, d_model]
        # 結果: [n_layers, n_heads, d_model]
        head_outputs = torch.einsum("lhd,lhdm->lhm", z, W_O)

        # 各 Head 出力に Final Layer Normalization を適用
        n_layers, n_heads, d_model = head_outputs.shape
        normalized_head_outputs = []
        for layer_idx in range(n_layers):
            layer_normalized = []
            for head_idx in range(n_heads):
                normalized = model.ln_final(
                    head_outputs[layer_idx, head_idx].unsqueeze(0)
                ).squeeze(0)
                layer_normalized.append(normalized)
            normalized_head_outputs.append(torch.stack(layer_normalized, dim=0))
        normalized_head_outputs = torch.stack(normalized_head_outputs, dim=0)

        # einsum を使用して効率的に計算: normalized_head_outputs @ W_U
        # normalized_head_outputs: [n_layers, n_heads, d_model], W_U: [d_model, d_vocab]
        # 結果: [n_layers, n_heads, d_vocab]
        logits = torch.einsum("lhm,mv->lhv", normalized_head_outputs, W_U)

        # バイアス項が存在する場合は各層・各ヘッドに追加
        if b_U is not None:
            logits = logits + b_U.unsqueeze(0).unsqueeze(0)

        return logits


def visualize_top_k_tokens(
    logits: torch.Tensor,
    model: HookedTransformer,
    top_k: int = 10,
    title: str = "Top-K Token Logits",
    figsize: Tuple[int, int] = (4, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
    show: bool = True,
    font_size: int = 10,
) -> Figure:
    """
    単一の logits から上位 K 個のトークンを 1次元 Heatmap で可視化する関数.

    Args:
        logits (torch.Tensor): 可視化する logits [d_vocab]
        model (HookedTransformer): トークナイザーを含むモデル
        top_k (int): 表示する上位トークン数. Defaults to 10.
        title (str): グラフのタイトル. Defaults to "Top-K Token Logits".
        figsize (Tuple[int, int]): 図のサイズ (width, height). Defaults to (4, 6).
        cmap (str): カラーマップ名. Defaults to "Blues".
        save_path (Optional[str]): 保存パス (None の場合は表示のみ). Defaults to None.
        show (bool): 図を表示するかどうか. Defaults to True.
        font_size (int): フォントサイズ. Defaults to 10.

    Returns:
        plt.Figure: 作成された matplotlib Figure オブジェクト
    """
    # logits を numpy 配列に変換
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.array(logits)

    # 1次元でない場合はエラー
    if logits_np.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits_np.ndim}D")

    # 上位 K 個のトークンインデックスを取得
    top_k_indices = np.argsort(logits_np)[-top_k:][::-1]  # 降順
    top_k_values = logits_np[top_k_indices]

    # 対応するトークンテキストを取得
    token_texts = []
    for idx in top_k_indices:
        try:
            if hasattr(model, "tokenizer") and model.tokenizer is not None:
                token_text = model.tokenizer.decode([idx])
            else:
                token_text = f"[TOKEN_{idx}]"
            # 特殊文字や空白を見やすく表示
            if token_text.strip() == "":
                token_text = "[SPACE]"
            elif len(token_text) > 20:  # 長いトークンは短縮
                token_text = token_text[:17] + "..."
            token_texts.append(token_text)
        except Exception:
            token_texts.append(f"[UNK_{idx}]")

    # 1次元 Heatmap 用のデータ準備 [top_k, 1] の形に変形
    heatmap_data = top_k_values.reshape(-1, 1)

    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)

    # カラーマップの範囲を設定
    vmax = max(heatmap_data)
    vmin = min(heatmap_data)

    # Heatmap の作成
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # カラーバーの追加
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Logit Value", fontsize=font_size)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # 軸ラベルの設定
    ax.set_yticks([])  # y軸のティックを非表示
    ax.set_xticks([])  # x軸は不要

    # 動的フォントサイズの計算
    def calculate_dynamic_font_size(figsize, top_k, token_texts, base_font_size):
        """
        画像サイズとトークン数に基づいて最適なフォントサイズを計算する関数.

        Args:
            figsize: 図のサイズ (width, height)
            top_k: 表示するトークン数
            token_texts: トークンテキストのリスト
            base_font_size: ベースとなるフォントサイズ

        Returns:
            int: 計算された最適なフォントサイズ
        """
        _, height = figsize

        # セルあたりの高さ (インチ単位)
        cell_height_inch = height / top_k

        # セルの高さをポイント単位に変換 (1インチ = 72ポイント)
        cell_height_points = cell_height_inch * 72

        # 最長トークンの文字数を考慮
        max_token_length = max(len(text) for text in token_texts) if token_texts else 5

        # フォントサイズの計算
        # セルの高さの 60% を目安とし, トークンの長さに応じて調整
        dynamic_size = cell_height_points * 0.6

        # 長いトークンの場合はフォントサイズを小さくする
        if max_token_length > 15:
            dynamic_size *= 0.6
        elif max_token_length > 10:
            dynamic_size *= 0.8

        # 最小・最大フォントサイズの制限
        min_font_size = 6
        max_font_size = min(base_font_size + 4, 20)

        calculated_size = max(min_font_size, min(max_font_size, dynamic_size))

        return int(calculated_size)

    # 動的フォントサイズを計算
    dynamic_font_size = calculate_dynamic_font_size(
        figsize, top_k, token_texts, font_size
    )

    # Heatmap 上にトークンを表示
    for i, (value, token_text) in enumerate(zip(top_k_values, token_texts)):
        # 正規化された値を計算 (0-1 の範囲)
        normalized_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5

        # 背景色とのコントラストを考慮してテキスト色を決定
        text_color = "white" if normalized_value > 0.5 else "black"

        ax.text(
            0,
            i,
            token_text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=dynamic_font_size,
            fontweight="bold",
        )

    # タイトル
    ax.set_title(title, fontsize=font_size + 2, pad=20)

    # レイアウトの調整
    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    # 図の表示
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def evaluate_layer_prediction_performance(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    prompt_col: str,
    target_col: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    各層での予測性能をデータセット全体で評価する関数.

    データセット内の各サンプルについて, 各層での target トークンの予測順位と確率を計算し,
    全サンプルでの平均値を返す.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        df (pd.DataFrame): 評価用データセット
        prompt_col (str): プロンプトが格納されている列名
        target_col (str): ターゲットトークンが格納されている列名

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]:
            - layer_ranks: 各層での平均順位 {"m0": rank, "m1": rank, ...}
            - layer_probs: 各層での平均確率 {"m0": prob, "m1": prob, ...}

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> # データフレームとキャッシュを準備
        >>> layer_ranks, layer_probs = evaluate_layer_prediction_performance(
        ...     model, cache, df, "prompt", "target"
        ... )
        >>> print(f"Layer 0 average rank: {layer_ranks['m0']:.2f}")
        >>> print(f"Layer 0 average probability: {layer_probs['m0']:.6f}")
    """
    n_layers = model.cfg.n_layers

    # 各層の累積値を初期化
    layer_ranks_sum = {f"m{layer_idx}": 0.0 for layer_idx in range(n_layers)}
    layer_probs_sum = {f"m{layer_idx}": 0.0 for layer_idx in range(n_layers)}

    # データセット内の各サンプルを処理
    for idx, row in df.iterrows():
        prompt = row[prompt_col]
        target = row[target_col]

        # プロンプトをトークン化
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False).squeeze(0)
        prompt_len = len(prompt_tokens)

        # idx を整数に変換
        data_idx = int(idx) if isinstance(idx, (int, float)) else 0

        # 各層での logit 寄与を計算
        layer_logits = compute_layer_logit_contribution(
            model, cache, data_idx=data_idx, seq_len=prompt_len
        )

        # 各層での予測順位と確率を計算・累積
        for layer_idx in range(n_layers):
            rank, prob = get_token_rank_and_probability(
                model, layer_logits[layer_idx], target
            )
            layer_ranks_sum[f"m{layer_idx}"] += rank
            layer_probs_sum[f"m{layer_idx}"] += prob

    # データ数で割って平均を計算
    n_samples = len(df)
    layer_ranks = {k: v / n_samples for k, v in layer_ranks_sum.items()}
    layer_probs = {k: v / n_samples for k, v in layer_probs_sum.items()}

    return layer_ranks, layer_probs


def evaluate_head_prediction_performance(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    prompt_col: str,
    target_col: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    各 Head での予測性能をデータセット全体で評価する関数.

    データセット内の各サンプルについて, 各 Head での target トークンの予測順位と確率を計算し,
    全サンプルでの平均値を返す.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache (Dict[str, torch.Tensor]): モデル実行時のアクティベーションキャッシュ
        df (pd.DataFrame): 評価用データセット
        prompt_col (str): プロンプトが格納されている列名
        target_col (str): ターゲットトークンが格納されている列名

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]:
            - layer_ranks: 各 Head での平均順位 {"a0.h0": rank, "a0.h1": rank, ...}
            - layer_probs: 各 Head での平均確率 {"a0.h0": prob, "a0.h1": prob, ...}

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> # データフレームとキャッシュを準備
        >>> layer_ranks, layer_probs = evaluate_head_prediction_performance(
        ...     model, cache, df, "prompt", "target"
        ... )
        >>> print(f"Head 0 average rank: {layer_ranks['a0.h0']:.2f}")
        >>> print(f"Head 0 average probability: {layer_probs['a0.h0']:.6f}")
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # 各層の累積値を初期化
    head_ranks_sum = {
        f"a{layer_idx}.h{head_idx}": 0.0
        for layer_idx in range(n_layers)
        for head_idx in range(n_heads)
    }
    head_probs_sum = {
        f"a{layer_idx}.h{head_idx}": 0.0
        for layer_idx in range(n_layers)
        for head_idx in range(n_heads)
    }

    # データセット内の各サンプルを処理
    for idx, row in df.iterrows():
        prompt = row[prompt_col]
        target = row[target_col]

        # プロンプトをトークン化
        prompt_tokens = model.to_tokens(prompt, prepend_bos=False).squeeze(0)
        prompt_len = len(prompt_tokens)

        # idx を整数に変換
        data_idx = int(idx) if isinstance(idx, (int, float)) else 0

        # 各 Head での logit 寄与を計算
        layer_logits = compute_head_logit_contribution(
            model, cache, data_idx=data_idx, seq_len=prompt_len
        )

        # 各 Head での予測順位と確率を計算・累積
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                rank, prob = get_token_rank_and_probability(
                    model, layer_logits[layer_idx][head_idx], target
                )
                head_ranks_sum[f"a{layer_idx}.h{head_idx}"] += rank
                head_probs_sum[f"a{layer_idx}.h{head_idx}"] += prob

    # データ数で割って平均を計算
    n_samples = len(df)
    head_ranks = {k: v / n_samples for k, v in head_ranks_sum.items()}
    head_probs = {k: v / n_samples for k, v in head_probs_sum.items()}

    return head_ranks, head_probs


def get_token_rank_and_probability(
    model: HookedTransformer,
    logits: torch.Tensor,
    text: str,
) -> tuple[int, float]:
    """
    指定されたテキストの最初のトークンについて, logits 内でのランクと softmax 確率を返す関数.

    テキストをトークン化し, 最初のトークンの logits 内での順位と確率を計算する.
    複数トークンに分割される場合は最初のトークンのみを対象とする.

    Args:
        model (HookedTransformer): トークナイザーを含むモデル
        logits (torch.Tensor): 語彙全体の logits [d_vocab]
        text (str): 対象となるテキスト (トークン化される)

    Returns:
        tuple[int, float]: (rank, probability) のタプル
            - rank: logits 内での順位 (1から始まる, 1が最高位)
            - probability: softmax 確率 (0-1の範囲)

    Raises:
        ValueError: logits が 1次元でない場合
        ValueError: テキストが空文字列またはトークン化できない場合

    Examples:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> logits = torch.randn(50257)
        >>> rank, prob = get_token_rank_and_probability(model, logits, "Tokyo")
        >>> print(f"'Tokyo'のランク: {rank}, 確率: {prob:.4f}")
        'Tokyo'のランク: 1234, 確率: 0.0023

        >>> rank, prob = get_token_rank_and_probability(model, logits, "New York")
        >>> print(f"'New York'の最初のトークンのランク: {rank}, 確率: {prob:.4f}")
        'New York'の最初のトークンのランク: 567, 確率: 0.0045
    """
    # 入力検証
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")

    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits.ndim}D")

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text cannot be empty or whitespace only")

    # テキストをトークン化
    try:
        # BOSトークンを付けずにトークン化
        token_ids = model.to_tokens(text, prepend_bos=False).squeeze(0)

        if len(token_ids) == 0:
            raise ValueError(f"Failed to tokenize text: '{text}'")

        # 最初のトークンの ID を取得
        target_token_id = token_ids[0].item()

    except Exception as e:
        raise ValueError(f"Failed to tokenize text '{text}': {e}") from e

    # logits を CPU に移動して numpy 配列に変換
    logits_np = logits.detach().cpu().numpy()

    # ランクの計算 (降順でソートして順位を取得)
    # argsort は昇順なので [::-1] で降順に変換
    sorted_indices = np.argsort(logits_np)[::-1]

    # 対象トークンの順位を見つける (0ベースから1ベースに変換)
    rank_positions = np.where(sorted_indices == target_token_id)[0]
    rank = rank_positions[0] + 1  # 1ベースの順位

    # softmax 確率の計算
    # 数値安定性のために max 値を引く
    logits_shifted = logits_np - np.max(logits_np)
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits)

    probability = softmax_probs[target_token_id]

    return rank, float(probability)


def save_logit_analysis_results(
    model: HookedTransformer,
    cache_base_dir: str,
    dataset_base_dir: str,
    output_base_dir: str,
    relation_type: str,
    relation_name: str,
    prompt_col: str = "clean",
    target_col: str = "object",
    device: Optional[torch.device] = None,
) -> None:
    """
    各種ロジット分析結果を計算して pickle ファイルとして保存する関数.

    指定されたディレクトリからキャッシュとデータセットを読み込み,
    層レベルと Head レベルの予測性能を評価して結果を構造化されたディレクトリに保存する.

    Args:
        model (HookedTransformer): 分析対象の HookedTransformer モデル
        cache_base_dir (str): キャッシュファイルが格納されているベースディレクトリパス
        dataset_base_dir (str): データセットファイルが格納されているベースディレクトリパス
        output_base_dir (str): 結果を保存するベースディレクトリパス
        relation_type (str): Relation タイプ (例: "factual", "commonsense")
        relation_name (str): Relation 名 (例: "city_in_country", "company_hq")
        prompt_col (str, optional): プロンプトが格納されている列名. Defaults to "clean".
        target_col (str, optional): ターゲットトークンが格納されている列名. Defaults to "object".
        device (Optional[torch.device], optional): 使用するデバイス. None の場合は自動選択. Defaults to None.

    Returns:
        None

    Raises:
        FileNotFoundError: 指定されたキャッシュまたはデータセットファイルが見つからない場合
        PermissionError: ファイルの書き込み権限がない場合
        ValueError: データセットの列が見つからない場合

    Examples:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> save_logit_analysis_results(
        ...     model=model,
        ...     cache_base_dir="out/cache",
        ...     dataset_base_dir="data/filtered_gpt2_small",
        ...     output_base_dir="out/logits",
        ...     relation_type="factual",
        ...     relation_name="city_in_country"
        ... )
        Loading cache from: out/cache/factual/city_in_country_0.pt
        Loading dataset from: data/filtered_gpt2_small/factual/city_in_country.csv
        Starting logit analysis for factual/city_in_country...
        Layer prediction performance saved to: out/logits/layer/factual/city_in_country.pkl
        Head prediction performance saved to: out/logits/head/factual/city_in_country.pkl

    Note:
        保存される結果ファイルの構造:
        - Layer レベル: {output_base_dir}/layer/{relation_type}/{relation_name}.pkl
        - Head レベル: {output_base_dir}/head/{relation_type}/{relation_name}.pkl

        各ファイルには以下の形式でデータが保存される:
        - {"ranks": {...}, "probs": {...}}

        キャッシュファイルは {cache_base_dir}/{relation_type}/{relation_name}*.pt の形式で検索される.
        データセットファイルは {dataset_base_dir}/{relation_type}/{relation_name}.csv から読み込まれる.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting logit analysis for {relation_type}/{relation_name}...")

    # キャッシュの読み込み
    cache_dir = Path(cache_base_dir) / relation_type
    cache_files = sorted(cache_dir.glob(f"{relation_name}*.pt"))

    if not cache_files:
        raise FileNotFoundError(
            f"No cache files found for {relation_type}/{relation_name} in {cache_dir}"
        )

    cache_path = cache_files[0]  # 最初にマッチしたファイルを使用
    print(f"Loading cache from: {cache_path}")

    try:
        cache = torch.load(cache_path, map_location=device, weights_only=False)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load cache from {cache_path}: {e}") from e

    # データセットの読み込み
    dataset_dir = Path(dataset_base_dir) / relation_type
    dataset_path = dataset_dir / f"{relation_name}.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load dataset from {dataset_path}: {e}"
        ) from e

    # 必要な列の存在確認
    if prompt_col not in df.columns:
        raise ValueError(
            f"Prompt column '{prompt_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

    print(f"Dataset loaded: {len(df)} samples")

    # 各層での予測性能を計算
    print("Evaluating layer prediction performance...")
    layer_ranks, layer_probs = evaluate_layer_prediction_performance(
        model, cache, df, prompt_col, target_col
    )

    # 各 Head での予測性能を計算
    print("Evaluating head prediction performance...")
    head_ranks, head_probs = evaluate_head_prediction_performance(
        model, cache, df, prompt_col, target_col
    )

    # 各層での結果を pickle ファイルで保存
    layer_output_dir = Path(output_base_dir) / "layer" / relation_type
    layer_output_dir.mkdir(parents=True, exist_ok=True)
    layer_output_path = layer_output_dir / f"{relation_name}.pkl"
    layer_results = {"ranks": layer_ranks, "probs": layer_probs}

    with open(layer_output_path, "wb") as f:
        pickle.dump(layer_results, f)
    print(f"Layer prediction performance saved to: {layer_output_path}")

    # 各 Head での結果を pickle ファイルで保存
    head_output_dir = Path(output_base_dir) / "head" / relation_type
    head_output_dir.mkdir(parents=True, exist_ok=True)
    head_output_path = head_output_dir / f"{relation_name}.pkl"
    head_results = {"ranks": head_ranks, "probs": head_probs}

    with open(head_output_path, "wb") as f:
        pickle.dump(head_results, f)
    print(f"Head prediction performance saved to: {head_output_path}")

    print(f"Logit analysis completed for {relation_type}/{relation_name}")


def load_logit_analysis_results(
    relation_name: str,
    analysis_level: str = "both",
    relation_type: Optional[str] = None,
    base_dir: str = str(LOGITS_DIR),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    save_logit_analysis_results 関数で保存された pickle ファイルから logit 分析結果を読み込む関数.

    指定されたディレクトリから層レベルまたは Head レベル (あるいは両方) の
    予測性能データを読み込み, 構造化された辞書として返す.

    Args:
        relation_name (str): Relation 名 (例: "city_in_country", "company_hq")
        analysis_level (str, optional): 読み込むデータのレベル.
            "layer": 層レベルのみ, "head": Headレベルのみ, "both": 両方. Defaults to "both".
        relation_type (Optional[str]): Relation タイプ (例: "factual", "commonsense"). Defaults to None.
        base_dir (str): 結果が保存されているベースディレクトリパス. Defaults to LOGITS_DIR.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: 読み込まれた分析結果
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


def main(
    model_name: str = "gpt2-small",
    cache_base_dir: str = "out/cache",
    dataset_base_dir: str = "data/filtered_gpt2_small",
    output_base_dir: str = "out/logits",
):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルの設定
    model = HookedTransformer.from_pretrained(model_name, device=device)

    for relation_name in [
        "city_in_country",
        "company_hq",
        "landmark_in_country",
        "landmark_on_continent",
        "plays_pro_sport",
        "product_by_company",
        "star_constellation_name",
    ]:
        # ロジット分析結果を保存
        save_logit_analysis_results(
            model=model,
            cache_base_dir=cache_base_dir,
            dataset_base_dir=dataset_base_dir,
            output_base_dir=output_base_dir,
            relation_type="factual",
            relation_name=relation_name,
            device=device,
        )

    for relation_name in [
        "task_person_type",
        "work_location",
    ]:
        # ロジット分析結果を保存
        save_logit_analysis_results(
            model=model,
            cache_base_dir=cache_base_dir,
            dataset_base_dir=dataset_base_dir,
            output_base_dir=output_base_dir,
            relation_type="commonsense",
            relation_name=relation_name,
            device=device,
        )


if __name__ == "__main__":
    main(
        model_name="gpt2-small",
        cache_base_dir="out/cache",
        dataset_base_dir="data/filtered_gpt2_small",
        output_base_dir="out/logits",
    )
