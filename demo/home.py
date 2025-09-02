import streamlit as st


def set_page(page_name):
    st.session_state.page = page_name


def display_home():
    """ホームページの表示"""
    if st.button("Single Graph"):
        set_page("single")
        st.rerun()

    if st.button("Multi Graphs"):
        set_page("multi")
        st.rerun()

    if st.button("Pairwise Union"):
        set_page("pairwise_union")
        st.rerun()

    if st.button("Pairwise Intersection"):
        set_page("pairwise_intersection")
        st.rerun()

    if st.button("Pairwise Difference"):
        set_page("pairwise_difference")
        st.rerun()

    # 開発中
    # if st.button("Weighted Difference"):
    #     set_page("weighted_difference")
    #     st.rerun()

    if st.button("Multi Union"):
        set_page("multi_union")
        st.rerun()

    if st.button("Multi Intersection"):
        set_page("multi_intersection")
        st.rerun()

    if st.button("Multi Difference"):
        set_page("multi_difference")
        st.rerun()
