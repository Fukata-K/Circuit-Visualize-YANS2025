import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from demo.core import load_model
from demo.figure_utils import (
    format_text,
    generate_circuit_svg,
    get_available_relations,
    get_svg_path_from_config,
)
from demo.html_utils import create_svg_html_content

st.title("🎯 タスクの説明")

st.markdown("""
本研究で扱うタスクは **知識推論タスク** です。

具体的には、**主語 (subject)**、**関係 (relation)**、**目的語 (object)** の三つ組からなる知識を対象とし、
与えられた主語と関係のペアから、正しい目的語を推論できるかを調べます。
""")

img = Image.open("demo/figures/task.png")
st.image(img)

st.markdown("""
本デモでは、このような知識の中から以下の表に示す **9種類の関係** を選び、
それぞれに対してサーキットを特定することで、言語モデルがどのように内部処理を行なっているかを分析します。

サーキット特定方法については「[ℹ️ サーキットとは？](./explain_circuit)」のページをご覧ください。
""")

# 9種類の関係を表で表示
data = [
    [
        "City In Country",
        "{} is located in the country of",
        "Rio de Janeiro, New Delhi",
        "Brazil, India",
    ],
    [
        "Company HQ",
        "{} is headquartered in the city of",
        "BBC Radio, Nippon Cultural Broadcasting",
        "London, Tokyo",
    ],
    [
        "Landmark In Country",
        "{} is in the country of",
        "Yamato Museum, Gobind Sagar",
        "Japan, India",
    ],
    [
        "Landmark On Continent",
        "{} is on the continent of",
        "Egypt, Scotland",
        "Africa, Europe",
    ],
    [
        "Plays Pro Sport",
        "{} plays professionally in the sport of",
        "Michael Jordan, R.A. Dickey",
        "Basketball, Baseball",
    ],
    [
        "Product By Company",
        "{} is a product of",
        "Game Boy Advance, Windows Vista",
        "Nintendo, Microsoft",
    ],
    [
        "Star Constellation Name",
        "{} is part of the constellation named",
        "Alpha Aquarii, Gamma Hydrae",
        "Aquarius, Hydra",
    ],
    [
        "Task Person Type",
        "{} is best suited for someone with the role of a",
        "Flying airplanes, Reporting news",
        "Pilot, Journalist",
    ],
    [
        "Work Location",
        "A {} typically works at a",
        "Doctor, Teacher",
        "Hospital, School",
    ],
]

df = pd.DataFrame(
    data, columns=["関係", "プロンプトテンプレート", "主語の例", "目的語の例"]
)
st.table(df)

st.markdown(
    "データセットは[こちら](https://github.com/evandez/relations/tree/main/data)を使用しています。"
)

st.divider()

# 評価方法の説明
st.markdown("""
### 📊 評価方法

タスクの性能評価には **Exact Match** を用います。

これは、推論された **目的語のトークン列** が、正解の目的語と **完全に一致した場合のみ** を正解とみなす指標です。

各関係ごとに用意した全ての主語に対して Exact Match を行ない、その正答率をもって関係ごとの性能とします。
""")

st.divider()

# タスクとサーキットの関係のデモ
st.markdown("""
### タスクとサーキットの関係

以下では、各関係に対応するサーキットと性能の関係を簡易的に確認できます。

性能の閾値を変更することで、サーキットのサイズがどのように変化するかを観察してみてください。

ここでの性能は、上記で説明した Exact Match の正答率を反映しており、
指定した閾値を達成できる (今回の[サーキット特定手法の手順](./explain_circuit)上) 最小のサーキットを表示します。
""")

# サーキット表示の簡易デモ
model = load_model()
relations = get_available_relations()

selected_relation = st.selectbox(
    "関係を選択：",
    relations,
    format_func=format_text,
    help="表示したいサーキットの関係を選択してください。",
)

score_threshold = (
    st.slider(
        "性能の閾値 (%):",
        0,
        100,
        50,
        10,
        help="サーキットに求める Exact Match の正答率を設定します。",
    )
    / 100.0
)

max_height = st.slider(
    "画像の高さ (px):",
    400,
    1200,
    600,
    50,
    help="表示するサーキット画像の高さを調整します。",
)

generate_button = st.button("サーキットを表示", use_container_width=True)

if generate_button:
    with st.spinner("サーキット生成中..."):
        # 固定設定でサーキット生成
        config = {
            "edge_selection_mode": "Performance",
            "topn": 200,  # Performance モードでは使用されない
            "score_threshold": score_threshold,
        }

        svg_path = get_svg_path_from_config(
            selected_relation,
            config["edge_selection_mode"],
            config["topn"],
            config["score_threshold"],
        )

        if not svg_path.exists():
            generate_circuit_svg(
                model=model,
                relation_name=selected_relation,
                svg_path=svg_path,
                topn=config["topn"],
                score_threshold=config["score_threshold"],
            )

# サーキット表示
if selected_relation:
    config = {
        "edge_selection_mode": "Performance",
        "topn": 200,
        "score_threshold": score_threshold,
    }

    svg_path = get_svg_path_from_config(
        selected_relation,
        config["edge_selection_mode"],
        config["topn"],
        config["score_threshold"],
    )

    if svg_path.exists():
        st.markdown(f"### {format_text(selected_relation)} のサーキット")
        st.info(f"**性能の閾値**: {int(score_threshold * 100)}%")
        st.warning(
            "注意：エッジ数が多すぎる場合は自動的にトリムされます。(上限 3000 本)"
        )

        html_content = create_svg_html_content(svg_path, max_height=max_height)
        components.html(html_content, height=max_height + 20, scrolling=False)
    else:
        st.warning(
            "「サーキットを表示」ボタンをクリックしてサーキットを生成してください。"
        )

st.divider()

# サーキットの見方
st.markdown("""
### サーキット図の見方

##### ノードの種類
中央に縦1列に並んでいるノードは、Input、各層の MLP、Output です。  
横に広がっているノードが Attention Head です。**Head をクリック**すると Attention Pattern が表示されます。

##### ノードの色 / 形状
Head の色は注目トークンの性質を表しています。  
オレンジ色に近いほど主語トークンに、青色に近いほど関係トークンに注目する Head であることを表しています。  
- 主語トークン：プロンプトテンプレートの `{}` に入る部分 (の last token)
- 関係トークン：プロンプトテンプレートの残りの部分 (の last token)

ひし形のノードは自己参照的な Head です。

**Head をクリック**して表示される Attention Pattern で確認してみてください。

##### ノードの枠線の色
枠線の色は目的語の出力順位を表しています。  
各ノードの出力を logit lens に通して語彙空間に射影したときに、目的語が上位に来るほどノードの枠線が緑色に近くなります。
""")
