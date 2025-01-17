import streamlit as st
from PIL import Image
from agentic_rag import AgenticRAG  
import pandas as pd
import plotly.express as px

class buildApp:
    def __init__(self):
        self.agentic_rag = AgenticRAG()  # Initialize the AgenticRAG instance
        self.header_image = Image.open("header_image.webp")  # Header image
        self.sidebar_image = Image.open("sidebar_image.webp")  # Sidebar image

    def set_global_styles(self):
        st.markdown(
            """
            <style>
            .css-18e3th9 { padding: 0px; }
            .css-1d391kg { padding: 0px; }
            .css-1u6zpdt { padding-left: 0px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        st.sidebar.image(self.sidebar_image, caption="ANA: Your Health Query Assistant", use_container_width=True)
        st.sidebar.header("Instructions")
        st.sidebar.write(
            """1. Enter your query in the input box below.
            2. Click on the **Submit** button."""
        )
        st.sidebar.markdown(
            "### About\nANA is powered by Agentic RAG and LangGraph, combining LLM with document retrieval and web search to provide accurate answers."
        )

    def render_tab1(self):
        st.image(self.header_image, use_container_width=True)
        query = st.text_input("📝 Type your question below:", "")
        if st.button("Submit"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    response = self.agentic_rag.invoke(query)
                    st.success("Query processed successfully!")
                    st.markdown("### 💡 ANA Says")
                    st.write(response.get("generation", "No response generated."))
            else:
                st.warning("Please enter a valid question.")

    def render_tab2(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                <h4 style="color: #4caf50;">Most Asked Categories</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            df_queries = self.agentic_rag.agent.db.read_sql()
            fig_query = px.bar(
                df_queries,
                x="category",
                y="Count",
                color="category",
                text="Count",
            )
            st.plotly_chart(fig_query, use_container_width=True)

        with col2:
            st.markdown(
                """
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px;">
                <h4 style="color: #ff5722;">Engagement Over Time</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            df_engagement = self.agentic_rag.agent.db.read_qns()
            df_engagement["year_month"] = pd.Categorical(df_engagement["year_month"], ordered=True)
            fig_engagement = px.line(
                df_engagement,
                x="year_month",
                y="Count",
                markers=True,
                labels={"year_month": "Month-Year", "Count": "Number of Questions"},
            )
            st.plotly_chart(fig_engagement, use_container_width=True)

    def run(self):
        self.set_global_styles()
        self.render_sidebar()

        tab1, tab2 = st.tabs(["📝 Ask ANA", "📊 View Analytics"])

        with tab1:
            self.render_tab1()

        with tab2:
            self.render_tab2()

        st.markdown(
            "---\nDeveloped by [Safni Usman](https://www.linkedin.com/in/safniusman/)"
        )

# Run the application
if __name__ == "__main__":
    app = buildApp()
    app.run()

        