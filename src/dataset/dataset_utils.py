import json
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer


def load_json_and_prepare(json_file: str) -> tuple[int, str, str, str, pd.DataFrame]:
    """
    JSON ファイルを読み込み, 必要な情報とサンプルデータフレームを準備する.

    Args:
        json_file (str): 入力 JSON ファイルのパス

    Returns:
        tuple: (num_samples, relation_type, relation_name, prompt_template, df)
            num_samples (int)    : サンプル数
            relation_type (str)  : relation type (例: factual)
            relation_name (str)  : relation name (例: city_in_country, 空白は '_' に置換・小文字化済み)
            prompt_template (str): prompt_template (例: "{} is part of")
            df (pd.DataFrame)    : サンプルデータフレーム (subject, object 列を含む)

    Raises:
        ValueError: JSON 構造が想定と異なる場合や必要なキー・列が存在しない場合

    Note:
        relation_name は空白をアンダースコアに置換し全て小文字にする.
        df は JSON 内の 'samples' を DataFrame 化したもの.
    """
    # JSONファイルの読み込み
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 必要なキーの存在チェック
    required_keys = ["samples", "properties", "name", "prompt_templates"]
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"JSON ファイル {json_file} に '{key}' キーが存在しません."
            )

    # データ数を取得
    num_samples = len(data["samples"])

    # relation type の抽出 (出力パスの指定に使用)
    relation_type = data["properties"]["relation_type"]  # 例: factual

    # relation name の抽出 (csvファイル名に使用)
    relation_name = data["name"]
    relation_name = relation_name.replace(
        " ", "_"
    ).lower()  # 空白を '_' に置換して小文字化 (例: city_in_country)

    # prompt_templateの抽出
    prompt_template = data["prompt_templates"][0]  # 例: "{} is part of"

    # サンプル部分をデータフレームに変換
    samples = data["samples"]
    for sample in samples:
        # 必要な列の存在チェック
        for col in ["subject", "object"]:
            if col not in sample:
                raise ValueError(
                    f"JSON ファイル {json_file} の 'samples' に '{col}' 列が存在しません."
                )
        # subject が先頭に来る場合は先頭を大文字にする
        if (
            isinstance(sample["subject"], str)
            and prompt_template.startswith("{}")
            and len(sample["subject"]) > 0
        ):
            sample["subject"] = sample["subject"][0].upper() + sample["subject"][1:]
        # object は先頭に半角スペースを付加する
        if isinstance(sample["object"], str):
            sample["object"] = " " + sample["object"]
    df = pd.DataFrame(samples)

    return num_samples, relation_type, relation_name, prompt_template, df


def collate_EAP(xs: list) -> tuple[list, list, torch.Tensor]:
    """
    DataLoader が 1バッチ分のサンプルを取り出した際に, それらを分解・整理してモデルや評価関数で扱いやすい形式に変換する関数.

    各サンプルは (clean, corrupted, labels) のタプルであることを想定し, それぞれをリストまたは tensor としてまとめて返す.

    Args:
        xs (list): バッチ内のサンプルのリスト. 各要素は (clean, corrupted, labels) のタプル.

    Returns:
        tuple:
            clean (list)         : バッチ内の clean 入力 (文字列) のリスト
            corrupted (list)     : バッチ内の corrupted 入力 (文字列) のリスト
            labels (torch.Tensor): バッチ内のラベル (正解・誤りトークンIDペア) の tensor (torch.long 型)
    """
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels, dtype=torch.long)  # 明示的に torch.long 型に変換
    return clean, corrupted, labels


class EAPDataset(Dataset):
    """
    clean/corrupted 入力ペアとラベル情報を含む pandas DataFrame を PyTorch の Dataset として扱えるように整形するクラス.

    主な用途は EAP (Edge Attribution Patching) などの実験で,
    モデルへの入力 (clean, corrupted) とラベル (正解・誤りトークンIDペア) を
    DataLoader でバッチ処理できる形で提供すること.

    Main features:
    - DataFrame からデータを受け取り内部で保持する.
    - __getitem__ で (clean, corrupted, [correct_idx, incorrect_idx]) のタプルを返す.
    - __len__ でデータセットのサンプル数を返す.
    - shuffle() でデータの順序をランダムに入れ替えることができる.
    - head(n) で先頭 n 件だけにデータを絞れる.
    - to_dataloader() で DataLoader を簡単に生成できる.
    """

    def __init__(self, df: pd.DataFrame):
        """
        指定された DataFrame を内部に保持する.
        DataFrame には 'clean', 'corrupted', 'correct_idx', 'incorrect_idx' 列が必要.

        Args:
            df (pd.DataFrame): 入力データセット (各行が 1サンプル)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df は pandas.DataFrame 型である必要があります")
        required_columns = {"clean", "corrupted", "correct_idx", "incorrect_idx"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame には {required_columns} の全ての列が必要です")
        self.df = df.copy()

    def __len__(self) -> int:
        """
        データセットのサンプル数を返す.

        Returns:
            int: サンプル数
        """
        return len(self.df)

    def shuffle(self, seed: int = 42) -> None:
        """
        乱数シードを指定してデータの行順をランダムにシャッフルする.
        Args:
            seed (int): 乱数シード (default: 42)
        """
        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

    def head(self, n: int) -> None:
        """
        データセットを先頭 n件のみに絞り込む.

        Args:
            n (int): 残すサンプル数
        """
        self.df = self.df.head(n).reset_index(drop=True)

    def __getitem__(self, index: int) -> tuple[str, str, list]:
        """
        指定したインデックスのサンプルを取得する.

        Args:
            index (int): 取得するサンプルのインデックス

        Returns:
            tuple:
                clean (str)    : クリーンな入力文
                corrupted (str): 破損させた入力文
                [correct_idx, incorrect_idx] (list): 正解・誤りトークンIDのペア
        """
        row = self.df.iloc[index]
        # clean: クリーンな入力文
        # corrupted: 破損させた入力文
        # [correct_idx, incorrect_idx]: 正解・誤りトークンIDのペア
        return (
            row["clean"],
            row["corrupted"],
            [row["correct_idx"], row["incorrect_idx"]],
        )

    def to_dataloader(self, batch_size: int) -> DataLoader:
        """
        このデータセットから DataLoader を生成する.

        Args:
            batch_size (int): バッチサイズ

        Returns:
            DataLoader: PyTorch DataLoader インスタンス
        """
        # collate_EAP 関数でバッチを整形
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)


def collate_EM(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Exact Match をバッチ処理で行うための collate 関数.

    この関数はデータローダーから受け取ったバッチ (リスト形式の辞書オブジェクト)をモデルへの入力として扱いやすい形式の辞書に変換する.
    各要素ごとにリスト化し, バッチ全体をまとめて返す.

    Args:
        batch (List[Dict[str, Any]]): 各サンプルが辞書形式で格納されたバッチデータ.
            各辞書は以下のキーを持つ必要がある:
                - "task_id": タスクの識別子
                - "prompt": 入力プロンプト
                - "target": 期待される出力
                - "full_prompt": プロンプト+ターゲットの結合文字列
                - "full_corrupted_prompt": 破損プロンプト+破損ターゲットの結合文字列
                - "prompt_length": プロンプトの長さ
                - "target_length": ターゲットの長さ

    Returns:
        Dict[str, List[Any]]: バッチ内の各項目をリスト化した辞書.
            各キーは上記と同じで値はバッチ内の全サンプルのリストとなる.

    Example:
        >>> batch = [
        ...     {"task_id": 0, "prompt": "AB", "target": "C", "full_prompt": "ABC", "full_corrupted_prompt": "XYZ",
        ...      "prompt_length": 2, "target_length": 1},
        ...     {"task_id": 1, "prompt": "DE", "target": "G", "full_prompt": "DEG", "full_corrupted_prompt": "ZWV",
        ...      "prompt_length": 2, "target_length": 1}
        ... ]
        >>> collate_EM(batch)
        {
            "task_ids": [0, 1],
            "prompts": ["AB", "DE"],
            "targets": ["C", "G"],
            "full_prompts": ["ABC", "DEG"],
            "full_corrupted_prompts": ["XYZ", "ZWV"],
            "prompt_lengths": [2, 2],
            "target_lengths": [1, 1]
        }
    """
    return {
        "task_ids": [item["task_id"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "targets": [item["target"] for item in batch],
        "full_prompts": [item["full_prompt"] for item in batch],
        "full_corrupted_prompts": [item["full_corrupted_prompt"] for item in batch],
        "prompt_lengths": [item["prompt_length"] for item in batch],
        "target_lengths": [item["target_length"] for item in batch],
    }


class PromptTargetDataset(Dataset):
    """
    PromptTargetDataset(df: pd.DataFrame, model: HookedTransformer, prepend_bos: bool = False)

    プロンプトとターゲットのペアを扱う PyTorch Dataset クラス.

    このデータセットは言語モデルのプロンプト-ターゲット形式のタスク評価を容易にするために設計されている.
    各アイテムはプロンプトとその対応するターゲット, および破損プロンプトとその対応するターゲットから構成される.

    Args:
        df (pd.DataFrame):
            評価データを含む DataFrame.
            "clean" (プロンプト), "object" (ターゲット), "corrupted" (破損プロンプト), "corrupted_object" (破損ターゲット) のカラムが必要.
        model (HookedTransformer): トークン化に使用する言語モデル. `to_tokens`メソッドを持っている必要がある.
        prepend_bos (bool, オプション): トークン化時にプロンプトの先頭に BOS トークンを付与するかどうか (default: False)

    Attributes:
        df (pd.DataFrame): 評価データを保持する DataFrame.
        model (HookedTransformer): トークン化に使用するモデル.
        prepend_bos (bool): プロンプトの先頭に BOS トークンを付与するかどうか.

    Methods:
        __len__(): データセット内のサンプル数を返す.
        __getitem__(idx): インデックス `idx` のサンプルの辞書を返す.辞書のキーは以下の通り:
            - "task_id": サンプルのインデックス
            - "prompt": プロンプト文字列
            - "target": ターゲット文字列
            - "full_prompt": プロンプト+ターゲットの結合文字列
            - "full_corrupted_prompt": 破損プロンプト+破損ターゲットの結合文字列
            - "prompt_length": トークン化されたプロンプトの長さ
            - "target_length": トークン化されたターゲットの長さ

    Example:
        >>> dataset = PromptTargetDataset(df, model)
        >>> sample = dataset[0]
        >>> print(sample["prompt"], sample["target"])
        >>> print(sample["full_prompt"], sample["full_corrupted_prompt"])
    """

    def __init__(
        self, df: pd.DataFrame, model: HookedTransformer, prepend_bos: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.model = model
        self.prepend_bos = prepend_bos

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["clean"]
        target = row["object"]
        corrupted_prompt = row["corrupted"]
        # 破損ターゲットは使用しない (今後の拡張で必要な場合は修正)
        _ = row["corrupted_object"]

        # プロンプトとターゲットのトークン化
        prompt_tokens = self.model.to_tokens(prompt, prepend_bos=self.prepend_bos)[0]
        target_tokens = self.model.to_tokens(target, prepend_bos=False)[0]

        # 完全なプロンプト
        full_prompt = prompt + target

        # 破損プロンプトの作成
        # トークン長を揃えるため共通のターゲット (clean target) を使用
        # Circuit evaluation では corrupted_prompt 部分の差異が重要であり, この変更は評価結果に影響しない
        # ただし, 何らかの理由で corrupted_target を使用する必要がある場合は修正する必要がある
        full_corrupted_prompt = corrupted_prompt + target

        return {
            "task_id": idx,
            "prompt": prompt,
            "target": target,
            "full_prompt": full_prompt,
            "full_corrupted_prompt": full_corrupted_prompt,
            "prompt_length": len(prompt_tokens),
            "target_length": len(target_tokens),
        }
