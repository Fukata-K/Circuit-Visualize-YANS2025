import streamlit as st
import streamlit.components.v1 as components

from demo.figure_utils import get_available_relations
from demo.html_utils import create_svg_html_content
from paths import get_svg_path

st.title("📝 分析事例")

# 分析事例1
st.markdown("""
### 分析事例 1：City In Country の 50% サーキット
「City In Country」のタスクに対して 50% の性能を満たすサーキットを特定すると、
第10層0番目の Head (a10.h0) と 第9層8番目の Head (a9.h8) に多くのエッジが集中していることがわかる。

これらの Head は 主語に強く注目すると同時に、目的語を高い順位で出力している。  
すなわち、「都市」に注目した上で「所在国」にマッピングする役割を持つ Head であると解釈できる。

これは[先行研究](https://arxiv.org/abs/2403.19521)で Mover Head として報告されている Head と一致しており、本研究の可視化手法が、既存知見を直感的に再現できることを示している。
""")

max_height1 = st.slider(
    "画像の高さ1 (px):",
    400,
    1200,
    600,
    50,
    help="表示するサーキット画像の高さを調整します。",
)
svg_path = get_svg_path(
    base_relation="city_in_country",
    perf_percent=0.5,
)
html_content = create_svg_html_content(svg_path, max_height=max_height1)
components.html(html_content, height=max_height1 + 20, scrolling=False)

st.divider()

# 分析事例2
st.markdown("""
### 分析事例 2：全関係の 40% サーキットの共通部分
9種類の関係タスクについて、それぞれ 40% の性能を満たすサーキットを求め、それらの 積集合を取ると、3つの Head のみが共通して残る結果となった。

このことは、これらの Head が今回の関係推論タスクにおいて特に重要な役割を担っていることを示唆している。  
可視化を通じて、このような「共通要素の抽出」も容易に確認できる点が、本手法の利点のひとつである。
""")

max_height2 = st.slider(
    "画像の高さ2 (px):",
    400,
    1200,
    600,
    50,
    help="表示するサーキット画像の高さを調整します。",
)
svg_path = get_svg_path(
    base_relation="city_in_country",
    other_relation=[
        rel for rel in get_available_relations() if rel != "city_in_country"
    ],
    set_operation_mode="intersection",
    perf_percent=0.4,
)
html_content = create_svg_html_content(svg_path, max_height=max_height2)
components.html(html_content, height=max_height2 + 20, scrolling=False)
