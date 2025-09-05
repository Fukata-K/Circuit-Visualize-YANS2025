from functools import partial
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from transformer_lens import HookedTransformer

from color import BGCOLOR, BLACK, BORDER, EDGE, GRAY, GREEN, NODE, RED, WHITE, Color
from dataset.dataset_utils import EAPDataset
from eap.attribute import attribute
from eap.graph import Graph
from evaluate.metric import logit_diff


class Circuit(Graph):
    @classmethod
    def from_graph(cls, graph: Graph) -> "Circuit":
        """
        Graph オブジェクトから Circuit オブジェクトを作成する.

        Args:
            graph (Graph): 変換元の Graph オブジェクト.

        Returns:
            Circuit: 新しい Circuit オブジェクト.
        """
        # 新しい Circuit インスタンスを作成
        circuit = cls()

        # 基底クラスの属性をすべてコピー
        for attr_name in dir(graph):
            if not attr_name.startswith("_") and hasattr(graph, attr_name):
                attr_value = getattr(graph, attr_name)
                if not callable(attr_value):
                    setattr(circuit, attr_name, attr_value)

        return circuit

    @classmethod
    def from_pt(cls, pt_path: Union[str, Path]) -> "Circuit":
        # 基底クラスの from_pt を呼び出して Graph オブジェクトを取得
        graph = super().from_pt(str(pt_path))

        # Graph から Circuit に変換
        return cls.from_graph(graph)

    def to_svg_with_node_styles(
        self,
        filename: Union[str, Path],
        border_colors: Optional[dict[str, str]] = None,
        fillcolors: Optional[dict[str, str]] = None,
        alphas: Optional[dict[str, str]] = None,
        size_scales: Optional[dict[str, float]] = None,
        shapes: Optional[dict[str, str]] = None,
        urls: Optional[dict[str, str]] = None,
        base_width: float = 0.75,
        base_height: float = 0.5,
        base_fontsize: float = 14,
        node_border_width: float = 3.0,
        edge_width: float = 1.0,
        display_not_in_graph: bool = False,
    ) -> None:
        """
        ノードごとに塗りつぶし色・透明度・大きさ (倍率) を指定してグラフを画像として保存する.

        Args:
            filename (str): 画像ファイルの保存先パス
            border_colors (dict[str, str], optional): ノード名 -> ボーダーカラーコード ("#RRGGBB") の辞書.
            fillcolors (dict[str, str], optional): ノード名 -> 色コード ("#RRGGBB") の辞書.
            alphas (dict[str, str], optional): ノード名 -> アルファ値 ("80"など) の辞書. 未指定時は全て"FF" (不透明).
            size_scales (dict[str, float], optional): ノード名 -> 大きさ倍率の辞書. 未指定時は全て1.0.
            shapes (dict[str, str], optional): ノード名 -> 形状 ("box" など) の辞書. 未指定時は全て "box".
            urls (dict[str, str], optional): ノード名 -> URL の辞書. 未指定時は空文字列.
            base_width (float): ノードの基準幅 (インチ単位).
            base_height (float): ノードの基準高さ (インチ単位).
            base_fontsize (float): ノードの基準フォントサイズ.
            node_border_width (float): ノードの境界線の太さ.
            edge_width (float): エッジの太さ.
            display_not_in_graph (bool): True の場合はグラフに含まれないノードも表示する.

        Returns:
            None
        """
        import pygraphviz as pgv

        g = pgv.AGraph(
            directed=True,
            layout="neato",
            bgcolor=BGCOLOR.to_hex(),
            overlap="true",
            splines="true",
        )

        # スタイル設定をデフォルト値で初期化
        style_config = self._initialize_style_config(
            border_colors, fillcolors, alphas, size_scales, shapes, urls
        )

        # ノードの追加
        self._add_nodes_to_graph(
            g,
            style_config,
            base_width,
            base_height,
            base_fontsize,
            node_border_width,
            display_not_in_graph,
        )

        # エッジの追加
        self._add_edges_to_graph(g, edge_width)

        # ダミーノード追加 (レイアウト調整用)
        self._add_corner_dummy_nodes(
            g, base_width, base_height, node_border_width, base_fontsize
        )

        g.layout(prog="neato")
        g.draw(filename, format="svg")

    def _initialize_style_config(
        self, border_colors, fillcolors, alphas, size_scales, shapes, urls
    ) -> dict:
        """スタイル設定のデフォルト値を初期化する"""
        if border_colors is None:
            border_colors = {node.name: BORDER.to_hex() for node in self.nodes.values()}

        if fillcolors is None:
            color_map = {
                "i": GRAY.to_hex(),
                "a": RED.to_hex(),
                "m": NODE.to_hex(),
                "l": GRAY.to_hex(),
            }
            fillcolors = {
                node.name: (
                    color_map.get(node.name[0], NODE.to_hex())
                    if node.in_graph
                    else NODE.to_hex()
                )
                for node in self.nodes.values()
            }

        if alphas is None:
            alphas = {node.name: "FF" for node in self.nodes.values()}
        if size_scales is None:
            size_scales = {node.name: 1.0 for node in self.nodes.values()}
        if shapes is None:
            shapes = {node.name: "box" for node in self.nodes.values()}
        if urls is None:
            urls = {node.name: "" for node in self.nodes.values()}

        return {
            "border_colors": border_colors,
            "fillcolors": fillcolors,
            "alphas": alphas,
            "size_scales": size_scales,
            "shapes": shapes,
            "urls": urls,
        }

    def _get_node_position(
        self, node_name: str, base_width: float, base_height: float
    ) -> tuple:
        """ノード名から座標を計算する"""
        x_spacing = base_width * 1.5
        y_spacing = base_height * 1.5

        n_heads = self.cfg.get("n_heads", 12)
        n_layers = self.cfg.get("n_layers", 12)
        x_left = -((n_heads - 1) / 2) * x_spacing
        x_right = ((n_heads - 1) / 2) * x_spacing
        y_top = (n_layers * 2 + 2) * y_spacing
        y_bottom = 0

        corner = {
            "top_left": (x_left, y_top),
            "top_right": (x_right, y_top),
            "bottom_left": (x_left, y_bottom),
            "bottom_right": (x_right, y_bottom),
        }

        if node_name == "input":
            return (0, y_bottom)
        elif node_name.startswith("a"):
            layer = int(node_name[1:].split(".")[0])
            head = int(node_name.split(".h")[1])
            x = (head - (n_heads - 1) / 2) * x_spacing
            y = (layer * 2 + 1) * y_spacing
            return (x, y)
        elif node_name.startswith("m"):
            layer = int(node_name[1:])
            y = (layer * 2 + 2) * y_spacing
            return (0, y)
        elif node_name == "logits":
            return (0, y_top)
        else:
            return corner.get(node_name, (0, 0))

    def _add_nodes_to_graph(
        self,
        g,
        style_config,
        base_width,
        base_height,
        base_fontsize,
        node_border_width,
        display_not_in_graph,
    ):
        """グラフにノードを追加する"""
        for node in self.nodes.values():
            if not node.in_graph and not display_not_in_graph:
                continue

            alpha = style_config["alphas"].get(node.name, "FF")
            border_color = style_config["border_colors"].get(node.name, BORDER.to_hex())
            fillcolor = style_config["fillcolors"].get(node.name, NODE.to_hex())
            fontcolor = (
                Color.from_hex(fillcolor).pick_text_color_from(BLACK, WHITE).to_hex()
            )
            scale = style_config["size_scales"].get(node.name, 1.0)
            shape = style_config["shapes"].get(node.name, "box")
            style = "filled, rounded" if shape == "box" else "filled"
            url = style_config["urls"].get(node.name, "")

            pos_x, pos_y = self._get_node_position(node.name, base_width, base_height)

            g.add_node(
                node.name,
                color=border_color + alpha,
                fillcolor=fillcolor + alpha,
                fontcolor=fontcolor + alpha,
                fontname="Helvetica",
                fontsize=base_fontsize * scale,
                shape=shape,
                style=style,
                width=base_width * scale,
                height=base_height * scale,
                fixedsize=True,
                penwidth=node_border_width,
                URL=url,
                pos=f"{pos_x},{pos_y}!",
            )

    def _get_edge_color(self, edge, max_score=0.1):
        """
        エッジの色を取得するヘルパー関数.
        各エッジに割り当てられたスコアに基づいてエッジの色を決定する.
        スコアが最大値に近いほど緑色で, 遠いほどデフォルト色で表示される.
        """
        if edge.score is not None:
            score = min(abs(float(edge.score)) / max_score, 1)
            R = EDGE.r * (1 - score) + GREEN.r * score
            G = EDGE.g * (1 - score) + GREEN.g * score
            B = EDGE.b * (1 - score) + GREEN.b * score
            return Color(R, G, B).to_hex()
        return EDGE.to_hex()

    def _add_edges_to_graph(self, g, edge_width):
        """グラフにエッジを追加する"""
        valid_edges = [
            edge
            for edge in self.edges.values()
            if edge.score is not None and edge.in_graph
        ]

        if not valid_edges:
            return  # エッジがない場合は何もしない

        # 全エッジ (有効でないものも含む) から max_edge_score を計算
        all_scored_edges = [
            edge for edge in self.edges.values() if edge.score is not None
        ]
        max_edge_score = (
            max(abs(float(edge.score)) for edge in all_scored_edges)
            if all_scored_edges
            else 1.0
        )

        for edge in valid_edges:
            g.add_edge(
                edge.parent.name,
                edge.child.name,
                penwidth=str(edge_width),
                color=self._get_edge_color(edge, max_edge_score),
            )

    def _add_corner_dummy_nodes(
        self, g, base_width, base_height, node_border_width, base_fontsize
    ):
        """レイアウト調整用のダミーノードを四隅に追加する"""
        for corner_name in ["top_left", "top_right", "bottom_left", "bottom_right"]:
            x, y = self._get_node_position(corner_name, base_width, base_height)
            if corner_name == "top_right":
                node_name = f"E: {self.in_graph.count_nonzero().item()}"
                fontcolor = WHITE.to_hex() + "00"  # トップ右にエッジ数を表示 (解除中)
            elif corner_name == "top_left":
                node_name = f"N: {self.nodes_in_graph.count_nonzero().item()}"
                fontcolor = WHITE.to_hex() + "00"  # トップ左にノード数を表示 (解除中)
            else:
                node_name = corner_name.split("_")[0][0] + corner_name.split("_")[1][0]
                fontcolor = BLACK.to_hex() + "00"  # 表示しない
            g.add_node(
                node_name,
                color=BORDER.to_hex() + "00",
                fillcolor=NODE.to_hex() + "00",
                fontcolor=fontcolor,
                fontname="Helvetica",
                fontsize=base_fontsize * 1.5,
                shape="box",
                style="filled, rounded",
                width=base_width,
                height=base_height,
                fixedsize=True,
                penwidth=node_border_width,
                pos=f"{x},{y}!",
            )

    def union_circuits(self, circuits):
        """
        self を基準として, 与えられた Circuit オブジェクト (単体またはリスト) のノード・エッジの和集合を計算し, self を破壊的に更新する関数.

        Args:
            circuits (Circuit or list[Circuit]): 和集合を取る Circuit オブジェクトまたはそのリスト.

        Notes:
            - ノードやエッジが「存在するか否か」のみ和集合を計算する.
            - ノードやエッジに付随する属性値 (スコア・ラベル等) は self (呼び出し元) のものがそのまま使われる.
            - 他の circuit の属性値は反映されない. 属性値のマージや平均化は行わない.
        """
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        for circuit in circuits:
            self.in_graph |= circuit.in_graph
            self.nodes_in_graph |= circuit.nodes_in_graph

    def intersection_circuits(self, circuits):
        """
        self を基準として, 与えられた Circuit オブジェクト (単体またはリスト) のノード・エッジの積集合を計算し, self を破壊的に更新する関数.

        Args:
            circuits (Circuit or list[Circuit]): 積集合を取る Circuit オブジェクトまたはそのリスト.

        Notes:
            - ノードやエッジが「存在するか否か」のみ積集合を計算する.
            - ノードやエッジに付随する属性値 (スコア・ラベル等) は self (呼び出し元) のものがそのまま使われる.
            - 他の circuit の属性値は反映されない. 属性値のマージや平均化は行なわない.
        """
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        for circuit in circuits:
            self.in_graph *= circuit.in_graph
            self.nodes_in_graph *= circuit.nodes_in_graph

    def difference_circuits(self, circuits):
        """
        self を基準として, 与えられた Circuit オブジェクト (単体またはリスト) のノード・エッジを順に引いた差集合を計算し, self を破壊的に更新する関数.

        Args:
            circuits (Circuit or list[Circuit]): 差集合を取る Circuit オブジェクトまたはそのリスト.

        Notes:
            - ノードやエッジが「存在するか否か」のみ差集合を計算する.
            - 親または子が存在しないエッジは削除される.
            - ノードやエッジに付随する属性値 (スコア・ラベル等) は self (呼び出し元) のものがそのまま使われる.
            - 他の circuit の属性値は反映されない. 属性値のマージや平均化は行わない.
        """
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        for circuit in circuits:
            self.in_graph *= ~circuit.in_graph
            self.nodes_in_graph *= ~circuit.nodes_in_graph
        self.nodes_in_graph[0] = True  # Input ノードは常に存在する
        self._prune_edge()

    def weighted_difference_circuits(self, circuit):
        """
        self を基準として, 与えられた Circuit オブジェクトのエッジスコアを引いた重み付き差集合を計算し, self を破壊的に更新する関数.

        Args:
            circuit (Circuit): 重み付き差集合を取る Circuit オブジェクト.

        Notes:
            - エッジに割り当てられたスコアの絶対値を最大値で正規化してから差を取り, 0 未満になる場合はそのエッジを削除する.
            - 親または子が存在しないエッジは削除される.
        """
        # スコアの絶対値を取得
        self_abs_scores = self.scores.abs()
        circuit_abs_scores = circuit.scores.abs()

        # それぞれの最大値で正規化
        self_max = self_abs_scores.max()
        circuit_max = circuit_abs_scores.max()

        if self_max > 1e-10:  # 数値的安定性のため閾値を使用
            self_normalized = self_abs_scores / self_max
        else:
            self_normalized = self_abs_scores

        if circuit_max > 1e-10:  # 数値的安定性のため閾値を使用
            circuit_normalized = circuit_abs_scores / circuit_max
        else:
            circuit_normalized = circuit_abs_scores

        # 正規化されたスコアの差を計算
        self.scores = self_normalized - circuit_normalized
        self.in_graph &= self.scores >= 0  # スコアが 0 未満のエッジを削除

        # 孤立ノードを削除
        self.prune()

    def _prune_edge(self):
        """
        親または子ノードがグラフに含まれていないエッジを削除する.
        """
        for edge in self.edges.values():
            if not edge.parent.in_graph or not edge.child.in_graph:
                edge.in_graph = False


def score_graph_with_eap_ig(
    df: pd.DataFrame,
    model: HookedTransformer,
    batch_size: int = 128,
    device: str = "cuda",
    ig_steps: int = 5,
    prepend_bos: bool = True,
    quiet: bool = False,
) -> Circuit:
    """
    指定したデータフレームとモデルで EAP-IG を用いてグラフにスコア付けを行い, スコア付け済みの Circuit を返す.
    データフレームには 'clean', 'corrupted', 'correct_idx', 'incorrect_idx' の列が必要.

    Args:
        df (pd.DataFrame): 入力データセット (各行が 1サンプル)
        model (HookedTransformer): 評価に用いるモデル
        batch_size (int): DataLoader のバッチサイズ
        device (str): モデル・データの配置先デバイス (例: "cuda")
        ig_steps (int): EAP-IG の積分ステップ数
        prepend_bos (bool): BOS トークンを入力に追加するかどうか (default: True)
        quiet (bool): True の場合スコアリングの進捗を表示しない (default: False)

    Returns:
        Circuit: スコア付け済みの Circuit オブジェクト

    Notes:
        スコア付けはエッジに対して行われ, 各エッジのスコアはモデルの出力に対する寄与度を表す.
    """
    # データセットから DataLoader を作成
    ds = EAPDataset(df)
    dataloader = ds.to_dataloader(batch_size)

    # モデルからグラフを生成
    g = Circuit.from_model(model)

    # EAP-IG でスコア付け
    attribute(
        model=model,
        graph=g,
        dataloader=dataloader,
        metric=partial(logit_diff, loss=True, mean=True),  # type: ignore
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        device=device,
        prepend_bos=prepend_bos,
        quiet=quiet,
    )

    # スコア付け済みのグラフを Circuit に変換して返す
    return Circuit.from_graph(g)


def save_circuit_as_json_and_pt(
    circuit: Circuit,
    json_path: Optional[Union[str, Path]] = None,
    pt_path: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> None:
    """
    Circuit オブジェクト (circuit) を JSON, PT 形式で指定パスに保存する関数.

    Args:
        circuit (Circuit): Circuit オブジェクト
        json_path (str): JSON ファイルの保存パス (None の場合は保存しない)
        pt_path (str): PT ファイルの保存パス (None の場合は保存しない)
        quiet (bool): True の場合保存の進捗を表示しない (default: False)

    Returns:
        None
    """
    # JSON ファイルとして保存
    if json_path is not None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        circuit.to_json(str(json_path))
        if not quiet:
            print(f"Saved JSON file: {json_path}")

    # PT ファイルとして保存 (PyTorch 形式)
    if pt_path is not None:
        Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
        circuit.to_pt(str(pt_path))
        if not quiet:
            print(f"Saved PT file: {pt_path}")
