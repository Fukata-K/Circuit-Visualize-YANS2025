import streamlit as st
from PIL import Image

st.title("ℹ️ サーキットとは？")

st.markdown("""
本デモでいう **サーキット** とは、Transformer 型言語モデルを有向グラフとして捉えたときの部分グラフを指します。  
ノードは入力, Attention Head, MLP, 出力で構成され、エッジは残差接続を表します。
""")

st.markdown("""
### 理論的な定義
- Transformer 全体は有向非巡回グラフ (DAG) とみなせます。
- サーキットはその**部分グラフ**であり、任意のノードが出力への経路を持つものとします。
""")

img = Image.open("demo/figures/explain_circuit.png")
st.image(img)

st.markdown("""
### 実装上の定義
1. モデルを有向グラフとして表現します。
2. 各エッジに「出力への影響度」スコアを割り当てます。
3. ある基準を決めてエッジをスコア順に選択します。
4. 基準を満たした時点での部分グラフをサーキットとします。

ここで、本デモでのエッジの選択基準は以下の2つから選べます。
- **Top-N Edges**：スコア上位 N 本のエッジを選択
- **Performance**：サーキットの性能がある閾値を超えるまでエッジを追加
""")

st.markdown("""
### 特定方法
まずは、直感的でわかりやすい [ACDC (Automatic Circuit DisCovery)](https://arxiv.org/abs/2304.14997) という手法を説明します。

1. モデルを有向グラフとして表現します。
2. 特定のエッジを除去してモデルの出力の変化を見ます。
3. 2で求めた変化量が事前に定めた閾値未満である場合、そのエッジが出力に与える影響は小さいとみなして、グラフから除去します。
4. 2と3を繰り返して、出力に大きな影響を与えるエッジのみを残し、最終的に残ったエッジとノードからなるグラフをサーキットとします。


上記の手法は直感的でわかりやすいものの、エッジを一つずつ除去して評価するため計算コストが高いという欠点があります。  
そこで、本デモでは **[EAP-IG (Edge Attribution Patching with Integrated Gradients)](https://arxiv.org/abs/2403.17806)**
という ACDC の近似手法を使用します。
""")

img_acdc = Image.open("demo/figures/acdc.png")
st.image(img_acdc, caption="ACDC のイメージ図", width=400)
