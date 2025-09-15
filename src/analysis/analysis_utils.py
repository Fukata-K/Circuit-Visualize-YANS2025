import torch
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import get_act_name


def get_attention_pattern(
    cache: dict | ActivationCache,
    data_idx: int,
    layer: int,
    head: int,
    seq_len: int | None = None,
) -> torch.Tensor:
    """
    指定した cache, data_idx, layer, head から対応する Attention Pattern を取得し，
    seq_len が指定されていればそのサイズにトリミングして返す関数.

    Args:
        cache (dict | ActivationCache): transformer_lens の run_with_cache で得られる cache
        data_idx (int): バッチ内のデータインデックス
        layer    (int): 層番号 (0始まり)
        head     (int): ヘッド番号 (0始まり)
        seq_len (int, optional): トリミング後の系列長 (None なら全体を返す, default: None)

    Returns:
        torch.Tensor: [seq_len, seq_len] 形式の Attention Pattern
    """
    key = get_act_name("attn", layer)
    attn = cache[key][data_idx, head]
    if seq_len is not None:
        return attn[:seq_len, :seq_len]
    return attn


def get_substring_token_span(
    model: HookedTransformer, prompt: str, substring: str
) -> tuple[int, int]:
    """
    Prompt を tokenize したときの substring 部分の最初と最後の token の index (inclusive) を返す.
    substring が 2つ以上見つかった場合は ValueError を投げる.

    Args:
        model (HookedTransformer): トークナイズ用モデル
        prompt    (str): プロンプト全文
        substring (str): 任意の部分文字列

    Returns:
        (int, int): substring の最初と最後の token の index (0始まり, 両端含む)

    Raises:
        ValueError: substring が prompt 内に見つからない場合, または 2つ以上見つかった場合
        TypeError : 引数の型が不正な場合
    """
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError("Model tokenizer is None")

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    substring_tokens_1 = tokenizer.encode(substring, add_special_tokens=False)
    substring_tokens_2 = tokenizer.encode(" " + substring, add_special_tokens=False)

    matches = []
    for substring_tokens in [substring_tokens_1, substring_tokens_2]:
        for i in range(len(prompt_tokens) - len(substring_tokens) + 1):
            if prompt_tokens[i : i + len(substring_tokens)] == substring_tokens:
                start_idx = i
                end_idx = i + len(substring_tokens) - 1
                matches.append((start_idx, end_idx))

    if len(matches) == 0:
        raise ValueError("substring が prompt 内に見つかりませんでした")
    if len(matches) > 1:
        raise ValueError(
            "substring が prompt 内に 2つ以上見つかりました: " + str(matches)
        )
    return matches[0]
