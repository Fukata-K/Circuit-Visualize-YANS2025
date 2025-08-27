from typing import Callable, List, Literal, Optional, Union

import torch
from einops import einsum
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import AttentionNode, Graph
from eap.utils import compute_mean_activations, make_hooks_and_matrices, tokenize_plus


def evaluate_graph(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metrics: Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]],
    quiet=False,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    skip_clean: bool = True,
    device="cuda",
    prepend_bos: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Circuit を評価する (つまり, 一部のノードのみが false になっているグラフで, 通常は graph.apply_threshold() を呼び出すことで作成される).
    Circuit が有効であることを確認するために事前に prune を実行することを推奨.

    Args:
        model (HookedTransformer): Circuit を実行するモデル
        graph (Graph): 評価する Circuit
        dataloader (DataLoader): 評価に使用するデータセット
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): 評価に使用するメトリック
        quiet (bool, optional): tqdm プログレスバーを非表示にするかどうか (default: False).
        intervention (Literal['patching', 'zero', 'mean','mean-positional'], optional): 評価に使用するアブレーション手法.
            'patching' は交換介入, 'mean-positional' は指定されたデータセット上で位置別平均を取る (default: 'patching').
        intervention_dataloader (Optional[DataLoader], optional):平均を計算するデータセット (default: None).
            intervention が mean または mean-positional の場合は必須.
        prepend_bos (bool, optional): 入力に BOS トークンを前置するかどうか (default: True).

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]:
            忠実度スコアのテンソル (またはそのリスト). リストの場合, 各要素は入力メトリックリストのメトリックに対応
    """
    assert model.cfg.use_attn_result, (
        "Model must be configured to use attention result (model.cfg.use_attn_result)"
    )
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, (
            "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
        )

    assert intervention in ["patching", "zero", "mean", "mean-positional"], (
        f"Invalid intervention: {intervention}"
    )

    if "mean" in intervention:
        assert intervention_dataloader is not None, (
            "Intervention dataloader must be provided for mean interventions"
        )
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model,
            graph,
            intervention_dataloader,
            per_position=per_position,
            device=device,
            prepend_bos=prepend_bos,
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    # このステップでグラフをクリーンアップし, 完全に接続されるまでコンポーネントを削除する
    graph.prune()

    # グラフ内のエッジを示す行列を構築
    in_graph_matrix = graph.in_graph.to(device=device, dtype=model.cfg.dtype)

    # ニューロン用の同様の行列
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=device, dtype=model.cfg.dtype)

        # エッジがグラフ内にあるが, そのすべてのニューロンが含まれていない場合, そのエッジも更新する必要がある
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(
            model.cfg.dtype
        )
        in_graph_matrix = einsum(
            in_graph_matrix,
            node_fully_in_graph,
            "forward backward, forward -> forward backward",
        )
    else:
        neuron_matrix = None

    # 逆の行列を取る. これを破損させたいエッジを指定するマスクとして使用するため
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        # モデルが Attention Head の出力も正規化する場合, グラフを評価する際にそれを考慮する必要がある
        attention_head_mask = torch.zeros(
            (graph.n_forward, model.cfg.n_layers), device=device, dtype=model.cfg.dtype
        )
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(
            dtype=model.cfg.dtype
        )
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # グラフ内の各ノードについて, 対応するエッジがグラフ内にない場合, その入力を破損する
    # 活性化差分 (クリーンと破損した活性化の間) を加えることで破損させる
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # layernorm が Attention 後に適用される場合 (gemmaのみ)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                # 前に来た Attention Head のクリーンな出力を取得
                clean_attention_results = einsum(
                    activation_matrix[1, :, :, : len(in_graph_vector)],
                    attention_head_mask[: len(in_graph_vector)],
                    "batch pos previous hidden, previous layer -> batch pos layer hidden",
                )

                # 非 Attention Head に対応する更新と, クリーンと破損した Attention Head の差分を取得
                if neuron_matrix is not None:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )
                else:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )

                # Attention 結果にバイアスを追加し, 差分を使用して破損した Attention 結果を計算
                # すべての Attention Head を一度に処理する. これが実行されているかどうかを判断する方法
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = (
                        clean_attention_results.unsqueeze(2)
                        + corrupted_attention_difference
                    )
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(
                        0
                    )
                    corrupted_attention_results += (
                        attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    )
                else:
                    corrupted_attention_results = (
                        clean_attention_results + corrupted_attention_difference
                    )
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(
                        0
                    )
                    corrupted_attention_results += attention_biases.unsqueeze(
                        0
                    ).unsqueeze(0)

                # クリーンと破損した Attention 結果の両方をレイヤーノームに通し, 更新に差分を追加する
                update = non_attention_update
                valid_layers = attention_head_mask[: len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(
                            clean_attention_results[:, :, None, i]
                        )
                        update += model.blocks[i].ln1_post(
                            corrupted_attention_results[:, :, :, i]
                        )
                    else:
                        update -= model.blocks[i].ln1_post(
                            clean_attention_results[:, :, i]
                        )
                        update += model.blocks[i].ln1_post(
                            corrupted_attention_results[:, :, i]
                        )

            else:
                # 非 gemma の場合は簡単
                activation_differences = activation_matrix
                # ここの ... は, Attention レイヤー全体の入力を構築する際の潜在的なヘッド次元を考慮するため
                if neuron_matrix is not None:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden",
                    )
                else:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous ... -> batch pos ... hidden",
                    )
            activations += update
            return activations

        return input_construction_hook

    def make_input_construction_hooks(
        activation_differences, in_graph_matrix, neuron_matrix
    ):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # レイヤー内のいずれかの Attention ノードがグラフ内にある場合, レイヤー全体の入力を構築するだけ
            if any(
                graph.nodes[f"a{layer}.h{head}"].in_graph
                for head in range(model.cfg.n_heads)
            ) and not (
                neuron_matrix is None
                and all(
                    parent_edge.in_graph
                    for head in range(model.cfg.n_heads)
                    for parent_edge in graph.nodes[f"a{layer}.h{head}"].parent_edges
                )
            ):
                for i, letter in enumerate("qkv"):
                    node = graph.nodes[f"a{layer}.h0"]
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(
                        activation_differences,
                        in_graph_matrix[:prev_index, bwd_index],
                        neuron_matrix,
                    )
                    input_construction_hooks.append(
                        (node.qkv_inputs[i], input_cons_hook)
                    )

            # MLP がグラフ内にある場合は MLP フックを追加
            if graph.nodes[f"m{layer}"].in_graph and not (
                neuron_matrix is None
                and all(
                    parent_edge.in_graph
                    for parent_edge in graph.nodes[f"m{layer}"].parent_edges
                )
            ):
                node = graph.nodes[f"m{layer}"]
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(
                    activation_differences,
                    in_graph_matrix[:prev_index, bwd_index],
                    neuron_matrix,
                )
                input_construction_hooks.append((node.in_hook, input_cons_hook))

        # 常に logits フックを追加
        if not (
            neuron_matrix is None
            and all(
                parent_edge.in_graph
                for parent_edge in graph.nodes["logits"].parent_edges
            )
        ):
            node = graph.nodes["logits"]
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(
                activation_differences,
                in_graph_matrix[:fwd_index, bwd_index],
                neuron_matrix,
            )
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks

    # メトリックがリストでない場合はリストに変換
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # ここで実際にモデルを実行/評価する
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(
            model, clean, device=device, prepend_bos=prepend_bos
        )
        corrupted_tokens, _, _, _ = tokenize_plus(
            model, corrupted, device=device, prepend_bos=prepend_bos
        )

        # fwd_hooks_corrupted は破損した活性化を activation_difference に追加
        # fwd_hooks_clean はクリーンな活性化を activation_difference から減算
        # activation_difference のサイズは (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = (
            make_hooks_and_matrices(
                model, graph, len(clean), n_pos, None, device=device
            )
        )

        input_construction_hooks = make_input_construction_hooks(
            activation_difference, in_graph_matrix, neuron_matrix
        )
        with torch.inference_mode():
            if intervention == "patching":
                # クリーンな活性化を減算し, 破損した活性化を追加することで介入する
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            else:
                # ゼロまたは平均アブレーションの場合, 破損した活性化の追加をスキップ
                # ただし平均アブレーションでは, 平均を追加する必要がある
                if "mean" in intervention:
                    activation_difference += means

            # 一部のメトリック (精度や KL など) にはクリーン logits が必要
            clean_logits = (
                None
                if skip_clean
                else model(clean_tokens, attention_mask=attention_mask)
            )

            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)

        for i, metric in enumerate(metrics):
            r = metric(logits, clean_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # メトリックが 1つしかない場合は結果を unwrap
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline(
    model: HookedTransformer,
    dataloader: DataLoader,
    metrics: List[Callable[[Tensor], Tensor]],
    run_corrupted=False,
    quiet=False,
    device="cuda",
    prepend_bos: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    任意の介入なしで指定されたデータローダーでモデルを評価する. これはモデルのベースライン性能を計算するのに有用.

    Args:
        model (HookedTransformer): 評価するモデル
        dataloader (DataLoader): 評価に使用するデータセット
        metrics (List[Callable[[Tensor], Tensor]]): 評価に使用するメトリック
        run_corrupted (bool, optional): 代わりに破損した例で評価するかどうか (default: False).
        prepend_bos (bool, optional): 入力に BOS トークンを前置するかどうか (default: True).

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]:
            性能スコアのテンソル (またはそのリスト). リストの場合, 各要素は入力メトリックリストのメトリックに対応
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    results = [[] for _ in metrics]
    if not quiet:
        dataloader = tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, _ = tokenize_plus(
            model, clean, device=device, prepend_bos=prepend_bos
        )
        corrupted_tokens, _, _, _ = tokenize_plus(
            model, corrupted, device=device, prepend_bos=prepend_bos
        )
        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            logits = model(clean_tokens, attention_mask=attention_mask)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results
