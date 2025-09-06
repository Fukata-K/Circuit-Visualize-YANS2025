import streamlit as st


def set_page(page_name):
    st.session_state.page = page_name


def display_sidebar_navigation():
    """サイドバーにページナビゲーションを表示"""
    current_page = st.session_state.get("page", "home")

    # サイドバーにナビゲーションを追加
    with st.sidebar:
        # ホームページ以外の場合は区切り線を追加
        if current_page != "home":
            st.markdown("---")

        st.subheader("ページナビゲーション")

        # 全ページのリスト
        pages = {
            "home": ("Home", "home"),
            "single": ("Single Circuit", "single"),
            "multi": ("Multi Circuits", "multi"),
            "pairwise_union": ("Pairwise Union", "pairwise_union"),
            "pairwise_intersection": ("Pairwise Intersection", "pairwise_intersection"),
            "pairwise_difference": ("Pairwise Difference", "pairwise_difference"),
            "multi_union": ("Multi Union", "multi_union"),
            "multi_intersection": ("Multi Intersection", "multi_intersection"),
            "multi_difference": ("Multi Difference", "multi_difference"),
        }

        # 現在のページ以外のページを表示
        for page_key, (page_display, page_value) in pages.items():
            if page_key != current_page:
                if st.button(page_display, key=f"nav_{page_key}", use_container_width=True):
                    set_page(page_value)
                    st.rerun()


def display_home():
    """ホームページの表示"""
    st.markdown("""
    ## Circuit Visualize Demo へようこそ！

    このアプリケーションでは、Transformerモデルのサーキットを可視化・分析できます。
    下記のボタンから各機能にアクセスできます。
    """)

    # ボタンを2列に配置
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("基本機能")
        if st.button("Single Circuit", use_container_width=True):
            set_page("single")
            st.rerun()

        if st.button("Multi Circuits", use_container_width=True):
            set_page("multi")
            st.rerun()

    with col2:
        st.subheader("ペアワイズ演算")
        if st.button("Pairwise Union", use_container_width=True):
            set_page("pairwise_union")
            st.rerun()

        if st.button("Pairwise Intersection", use_container_width=True):
            set_page("pairwise_intersection")
            st.rerun()

        if st.button("Pairwise Difference", use_container_width=True):
            set_page("pairwise_difference")
            st.rerun()

    # マルチ演算は横幅いっぱいに表示
    st.subheader("マルチ演算")
    col3, col4, col5 = st.columns(3)

    with col3:
        if st.button("Multi Union", use_container_width=True):
            set_page("multi_union")
            st.rerun()

    with col4:
        if st.button("Multi Intersection", use_container_width=True):
            set_page("multi_intersection")
            st.rerun()

    with col5:
        if st.button("Multi Difference", use_container_width=True):
            set_page("multi_difference")
            st.rerun()

    # 説明セクション
    st.markdown("---")
    st.markdown("""
    ### 機能説明

    **基本機能**
    - **Single Circuit**: 単一のサーキットを表示
    - **Multi Circuits**: 複数のサーキットを同時表示

    **ペアワイズ演算**
    - **Pairwise Union**: 2つのサーキットの和集合
    - **Pairwise Intersection**: 2つのサーキットの積集合
    - **Pairwise Difference**: 2つのサーキットの差集合

    **マルチ演算**
    - **Multi Union**: 複数サーキットの和集合
    - **Multi Intersection**: 複数サーキットの積集合
    - **Multi Difference**: 複数サーキットの差集合
    """)
