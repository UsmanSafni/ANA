import streamlit as st
from PIL import Image
from agentic_rag import AgenticRAG  # Import the class from your OOP implementation
import pandas as pd
import plotly.express as px




# Initialize the AgenticRAG instance
agentic_rag = AgenticRAG()

# Load images
header_image = Image.open("header_image.webp")  # Replace with a relevant image path
sidebar_image = Image.open("sidebar_image.webp")  # Replace with a relevant image path


# Header with image
#st.title("🌟 Agentic RAG: Q & A System")

# Tabs with custom styling (visually appealing)
tab1, tab2 = st.tabs(["📝 Ask ANA",  "📊 View Analytics"])

st.markdown(
    """
    <style>
    /* Remove padding from the body */
    .css-18e3th9 {
        padding-top: 0px;
        padding-bottom: 0px;
        padding-left: 0px;
        padding-right: 0px;
    }

    /* Remove default padding from sidebar */
    .css-1d391kg {
        padding-top: 0px;
        padding-bottom: 0px;
    }

    /* Adjust space between sidebar and main content */
    .css-1u6zpdt {
        padding-left: 0px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Tab 1: Ask ANA
with tab1:
    st.image("header_image.webp", use_container_width=True)
    #st.markdown("## 📝 Ask ANA: Your Health Query Assistant")
    
    # Sidebar with instructions and image
    st.sidebar.image(sidebar_image, caption="ANA: Your Health Query Assistant", use_container_width=True)
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Enter your query in the input box below.
    2. Click on the **Submit** button.
    """)
    st.sidebar.markdown(
        "### About\nANA is powered by Agentic RAG and LangGraph that combines LLM with document retrieval and web search to provide accurate answers based on context."
    )

    # Main Q&A Section
    query = st.text_input("📝Type your question below:", "")
    if st.button("Submit"):
        if query.strip():
            with st.spinner("Processing your query..."):
                # Invoke the AgenticRAG pipeline
                response = agentic_rag.invoke(query)
                # Display the results
                st.success("Query processed successfully!")
                st.markdown("### 💡 ANA Says")
                st.write(response.get("generation", "No response generated."))
        else:
            st.warning("Please enter a valid question.")

# Tab 2: View Analytics with Columns
with tab2:
    #st.markdown("## 📊 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:  # Query Insights Card
        #st.markdown("### Query Insights")
        st.markdown(
            """
            <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #4caf50;">Most Asked Categories</h4>
            """,
            unsafe_allow_html=True,
        )
        #query_data = {
        #   "category": ["General Health"],
        #    "Count": [1],
        #}
        #df_queries = pd.DataFrame(query_data)
        
        #df_queries = pd.read_sql(query, con=engine)
        df_queries = agentic_rag.agent.db.read_sql()

        fig_query = px.bar(
            df_queries, 
            x="category", 
            y="Count", 
            color="category", 
            text="Count"
        )
        st.plotly_chart(fig_query, use_container_width=True)

    with col2:  # User Engagement Card
        #st.markdown("### User Engagement")
        st.markdown(
            """
            <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #ff5722;">Engagement Over Time</h4>
            """,
            unsafe_allow_html=True,
        )
        df_engagement = agentic_rag.agent.db.read_qns()
        df_engagement["year_month"] =pd.Categorical(df_engagement["year_month"], ordered=True)

        # Create the plot
        fig_engagement = px.line(
            df_engagement, 
            x="year_month", 
            y="Count", 
            markers=True,
            labels={"year_month": "Month-Year", "Count": "Number of Questions"}
            )

        # Display the plot
        st.plotly_chart(fig_engagement, use_container_width=True)

# Footer
st.markdown(
    """
    ---
    Developed by [Safni Usman](https://www.linkedin.com/in/safniusman/) 
    """
)
