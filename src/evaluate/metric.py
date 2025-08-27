import torch


def get_logit_positions(
    logits: torch.Tensor, input_length: torch.Tensor
) -> torch.Tensor:
    """
    各サンプルの最終トークン位置の logit ベクトル (語彙サイズ分) を抽出する関数.

    モデルの出力 logits ([batch, seq_len, vocab_size]) と各サンプルの入力長 (input_length) を受け取り,
    それぞれのサンプルについて「input_length-1」の位置 (=入力の末尾トークン) の logit ベクトルのみを抽出して返す.

    Args:
        logits (torch.Tensor)      : モデルの出力 logits. shape: [batch_size, seq_len, vocab_size]
        input_length (torch.Tensor): 各サンプルの入力長. shape: [batch_size] (各要素はint)

    Returns:
        torch.Tensor: 各サンプルの末尾トークン位置の logit ベクトル. shape: [batch_size, vocab_size]
    """
    # バッチサイズを取得
    batch_size = logits.size(0)
    # バッチ内の各サンプルのインデックスを作成
    idx = torch.arange(batch_size, device=logits.device)
    # 各サンプルの「input_length-1」位置の logit ベクトルのみを抽出
    logits = logits[idx, input_length - 1]
    return logits


def logit_diff(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """
    モデルの出力 logits における「正解トークン」と「誤りトークン」の logit 値の差を評価値として返す関数.

    各サンプルについて, 末尾トークン位置 (input_length-1) の logit ベクトルから
    正解トークン (correct_idx) と誤りトークン (incorrect_idx) の logit 値を抽出し,
    その差 (正解 - 誤り) を計算する. 損失として使う場合は符号を反転する.
    バッチごとに平均を取るかどうかも選択できる.

    Args:
        logits (torch.Tensor)      : モデルの出力logits. shape: [batch_size, seq_len, vocab_size]
        corrupted_logits (torch.Tensor): (未使用だが他の評価関数との互換性のために受け取る)
        input_length (torch.Tensor): 各サンプルの入力長. shape: [batch_size]
        labels (torch.Tensor): 各サンプルの [correct_idx, incorrect_idx]. shape: [batch_size, 2]
        mean (bool): True の場合はバッチ平均を返す. False の場合は各サンプルごとの値を返す.
        loss (bool): True の場合は損失として符号を反転する (低いほど良い値になる).

    Returns:
        torch.Tensor: 評価値 (バッチ平均または各サンプルごとの値)
    """
    # 各サンプルの末尾トークン位置の logit ベクトルを抽出
    logits = get_logit_positions(logits, input_length)
    # labels ([correct_idx, incorrect_idx]) の位置にある logit 値を抽出
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    # 正解トークンと誤りトークンの logit 値の差を計算
    results = good_bad[:, 0] - good_bad[:, 1]
    # 損失として使う場合は符号を反転
    if loss:
        results = -results
    # バッチ平均を取る場合
    if mean:
        results = results.mean()
    return results


def correct_logit(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """
    各サンプルについて, 末尾トークン位置 (input_length-1) の正解トークンの logit 値のみを抽出して返す関数.

    Args:
        logits (torch.Tensor): モデルの出力 logits. shape: [batch_size, seq_len, vocab_size]
        corrupted_logits (torch.Tensor): (未使用だが他の評価関数との互換性のために受け取る)
        input_length (torch.Tensor): 各サンプルの入力長. shape: [batch_size]
        labels (torch.Tensor): 各サンプルの [correct_idx, incorrect_idx]. shape: [batch_size, 2]
        mean (bool): True の場合はバッチ平均を返す. False の場合は各サンプルごとの値を返す.
        loss (bool): True の場合は損失として符号を反転する (低いほど良い値になる).

    Returns:
        torch.Tensor: 正解トークンの logit 値 (バッチ平均または各サンプルごとの値)
    """
    # 各サンプルの末尾トークン位置の logit ベクトルを抽出
    logits = get_logit_positions(logits, input_length)
    # labels ([correct_idx, incorrect_idx]) の位置にある logit 値を抽出
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    # 正解トークンのみの logit 値を抽出
    results = good_bad[:, 0]
    # 損失として使う場合は符号を反転
    if loss:
        results = -results
    # バッチ平均を取る場合
    if mean:
        results = results.mean()
    return results


def correct_prob(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """
    各サンプルについて, 末尾トークン位置 (input_length-1) の正解トークンの確率のみを抽出して返す関数.

    Args:
        logits (torch.Tensor): モデルの出力 logits. shape: [batch_size, seq_len, vocab_size]
        corrupted_logits (torch.Tensor): (未使用だが他の評価関数との互換性のために受け取る)
        input_length (torch.Tensor): 各サンプルの入力長. shape: [batch_size]
        labels (torch.Tensor): 各サンプルの [correct_idx, incorrect_idx]. shape: [batch_size, 2]
        mean (bool): True の場合はバッチ平均を返す. False の場合は各サンプルごとの値を返す.
        loss (bool): True の場合は損失として符号を反転する (低いほど良い値になる).

    Returns:
        torch.Tensor: 正解トークンの確率 (バッチ平均または各サンプルごとの値)
    """
    # 各サンプルの末尾トークン位置の logit ベクトルを抽出
    logits = get_logit_positions(logits, input_length)
    # softmax で確率に変換
    probs = torch.softmax(logits, dim=-1)
    # labels ([correct_idx, incorrect_idx]) の位置にある確率を抽出
    good_bad = torch.gather(probs, -1, labels.to(logits.device))
    # 正解トークンのみの確率を抽出
    results = good_bad[:, 0]
    # 損失として使う場合は符号を反転
    if loss:
        results = -results
    # バッチ平均を取る場合
    if mean:
        results = results.mean()
    return results


def correct_rank(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """
    各サンプルについて, 末尾トークン位置 (input_length-1) の正解トークンの出力確率順位 (rank) を抽出して返す関数.
    注意: 順位は 1 から始まり, 1 が最高順位 (最も高い確率) となる.

    Args:
        logits (torch.Tensor): モデルの出力 logits. shape: [batch_size, seq_len, vocab_size]
        corrupted_logits (torch.Tensor): (未使用だが他の評価関数との互換性のために受け取る)
        input_length (torch.Tensor): 各サンプルの入力長. shape: [batch_size]
        labels (torch.Tensor): 各サンプルの [correct_idx, incorrect_idx]. shape: [batch_size, 2]
        mean (bool): True の場合はバッチ平均を返す. False の場合は各サンプルごとの値を返す.
        loss (bool): True の場合は損失として符号を反転する (低いほど良い値になる).

    Returns:
        torch.Tensor: 正解トークンの順位 (バッチ平均または各サンプルごとの値, 1が最高順位)
    """
    # 各サンプルの末尾トークン位置の logit ベクトルを抽出
    logits = get_logit_positions(logits, input_length)
    # softmax で確率に変換
    probs = torch.softmax(logits, dim=-1)
    # 正解トークンのインデックスを取得
    correct_indices = labels[:, 0].to(logits.device)
    # 各サンプルごとに降順ソートしたときの順位を計算
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    # 各サンプルについて正解トークンが何番目かを求める
    ranks = (sorted_indices == correct_indices.unsqueeze(1)).nonzero(as_tuple=False)[
        :, 1
    ] + 1  # 1始まり
    # 損失として使う場合は符号を反転
    if loss:
        ranks = -ranks
    # バッチ平均を取る場合
    if mean:
        ranks = ranks.float().mean()
    return ranks
