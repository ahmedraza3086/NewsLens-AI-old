
import streamlit as st
import pandas as pd

from backend2 import (
    initialize_project,
    predict_headline,
    investigate_news,
    chat_about_investigation
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NewsLens AI",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #10141d;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #f1f5f9;
}

div[data-testid="stMetric"] {
    background: #1a2230;
    padding: 15px;
    border-radius: 12px;
}

.news-card {
    padding: 22px;
    border-radius: 14px;
    background: #1a2230;
    margin-bottom: 15px;
}

.fake-result {
    background: rgba(239,68,68,0.12);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 20px;
}

.real-result {
    background: rgba(34,197,94,0.12);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD PROJECT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_project():
    return initialize_project()


st.title("📰 NewsLens AI")
st.caption(
    "Machine Learning News Detection · AI Investigation · Web Evidence"
)

with st.spinner("Loading dataset and training ML models..."):

    try:
        project = load_project()

    except Exception as error:

        st.error(f"Initialization error: {error}")

        st.info(
            "Verify your dataset path and installed packages."
        )

        st.stop()


df = project["df"]
eda = project["eda"]
trained_data = project["trained_data"]
clustered_df = project["clustered_df"]
cluster_topics = project["cluster_topics"]


# ============================================================
# SESSION STATE
# ============================================================

if "investigation" not in st.session_state:
    st.session_state.investigation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_headline" not in st.session_state:
    st.session_state.last_headline = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("NewsLens AI")

    st.markdown("""
    **AI News Intelligence**

    Detect suspicious news, investigate claims,
    search web evidence, and chat with your investigator.
    """)

    st.divider()

    st.metric("Dataset Articles", f"{len(df):,}")

    st.metric("Fake Articles", f"{eda['fake_count']:,}")

    st.metric("Real Articles", f"{eda['real_count']:,}")

    st.divider()

    if st.button("🗑️ Reset Investigation", use_container_width=True):

        st.session_state.investigation = None
        st.session_state.chat_history = []
        st.session_state.last_headline = ""

        st.rerun()

    st.caption(
        "ML predictions are not independent fact checks."
    )


# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs([
    "📊 Dashboard",
    "🤖 News Investigator",
    "💬 AI Chat",
    "📈 Model Evaluation",
    "🔎 Topic Discovery"
])


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tabs[0]:

    st.header("Dataset Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Headlines", f"{eda['total_records']:,}")

    col2.metric("Fake News", f"{eda['fake_count']:,}")

    col3.metric("Real News", f"{eda['real_count']:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Class Distribution")

        st.bar_chart(eda["class_distribution"])

    with col2:

        st.subheader("Headline Length")

        lengths = eda["headline_lengths"].value_counts().sort_index()

        st.bar_chart(lengths)

    st.divider()

    st.subheader("Most Common Words")

    common_words_df = pd.DataFrame(
        eda["common_words"],
        columns=["Word", "Frequency"]
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.bar_chart(
            common_words_df.set_index("Word")
        )

    with col2:

        st.dataframe(
            common_words_df,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sources")

        if not eda["top_sources"].empty:
            st.bar_chart(eda["top_sources"])
        else:
            st.info("Source information unavailable.")

    with col2:

        st.subheader("Categories")

        if not eda["categories"].empty:
            st.bar_chart(eda["categories"])
        else:
            st.info("Category information unavailable.")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# TAB 2 — NEWS INVESTIGATOR
# ============================================================

with tabs[1]:

    st.header("AI News Investigator")

    st.write(
        "Predict the headline with machine learning, then investigate "
        "it using web search and AI-powered evidence analysis."
    )

    model_name = st.selectbox(
        "ML Model",
        [
            "Logistic Regression",
            "Naive Bayes"
        ],
        key="investigation_model"
    )

    headline = st.text_area(
        "News Headline or Claim",
        height=120,
        placeholder="Paste a headline or news claim here...",
        key="investigation_headline"
    )

    article_text = st.text_area(
        "Article Content (Optional)",
        height=200,
        placeholder="Paste the full article here if you have it..."
    )

    run_investigation = st.button(
        "🔍 Analyze & Investigate News",
        type="primary",
        use_container_width=True
    )

    if run_investigation:

        if not headline.strip():

            st.warning("Please enter a headline.")

        else:

            # Step 1 — ML prediction
            with st.spinner("Running ML prediction..."):

                ml_result = predict_headline(
                    headline,
                    model_name,
                    trained_data
                )

            # Step 2 — Web investigation
            with st.spinner(
                "AI is searching the web and analyzing evidence..."
            ):

                try:

                    report = investigate_news(
                        headline,
                        ml_result,
                        article_text
                    )

                    st.session_state.investigation = {
                        "headline": headline,
                        "article_text": article_text,
                        "model_name": model_name,
                        "ml_result": ml_result,
                        "report": report
                    }

                    st.session_state.last_headline = headline
                    st.session_state.chat_history = []

                except Exception as error:

                    st.error(
                        f"Investigation failed: {str(error)}"
                    )

    # Display saved investigation
    investigation = st.session_state.investigation

    if investigation:

        st.divider()

        st.subheader("Machine Learning Prediction")

        ml_result = investigation["ml_result"]

        prediction = ml_result["prediction"]

        if prediction == "FAKE":

            st.markdown(
                f"""
                <div class="fake-result">
                    <h2>🚨 LIKELY FAKE</h2>
                    <h3>Confidence: {ml_result['confidence']:.2%}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="real-result">
                    <h2>✅ LIKELY REAL</h2>
                    <h3>Confidence: {ml_result['confidence']:.2%}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2 = st.columns(2)

        col1.metric(
            "Fake Probability",
            f"{ml_result['probabilities']['Fake']:.2%}"
        )

        col2.metric(
            "Real Probability",
            f"{ml_result['probabilities']['Real']:.2%}"
        )

        st.divider()

        st.subheader("Prediction Explanation")

        explanations = pd.DataFrame(
            ml_result["explanations"]
        )

        if not explanations.empty:

            st.dataframe(
                explanations,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("🌐 AI Investigation Report")

        st.markdown(investigation["report"])

        st.divider()

        st.caption(
            "The ML result is a text-pattern prediction. "
            "The investigation report evaluates evidence retrieved "
            "from the web. Neither should be treated as infallible."
        )


# ============================================================
# TAB 3 — CHATBOT
# ============================================================

with tabs[2]:

    st.header("News Investigation Chat")

    investigation = st.session_state.investigation

    if not investigation:

        st.info(
            "First run a news investigation from the News Investigator tab."
        )

    else:

        st.markdown(
            f"**Current news:** {investigation['headline']}"
        )

        st.caption(
            f"ML prediction: {investigation['ml_result']['prediction']}"
        )

        st.divider()

        # Display chat history
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # Chat input
        user_message = st.chat_input(
            "Ask about the news, evidence, or sources..."
        )

        if user_message:

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_message
            })

            with st.chat_message("user"):
                st.markdown(user_message)

            with st.chat_message("assistant"):

                with st.spinner(
                    "NewsLens is thinking and searching if needed..."
                ):

                    try:

                        response = chat_about_investigation(
                            user_message,
                            investigation["headline"],
                            investigation["ml_result"],
                            investigation["report"],
                            st.session_state.chat_history[:-1]
                        )

                        st.markdown(response)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })

                    except Exception as error:

                        st.error(f"Chat error: {str(error)}")


# ============================================================
# TAB 4 — MODEL EVALUATION
# ============================================================

with tabs[3]:

    st.header("Model Evaluation")

    results = trained_data["results"]

    comparison = pd.DataFrame(results).T[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]

    st.subheader("Model Comparison")

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.bar_chart(comparison)

    st.divider()

    selected_model = st.selectbox(
        "Inspect Model",
        list(results.keys()),
        key="evaluation_model"
    )

    model_result = results[selected_model]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{model_result['Accuracy']:.2%}")

    col2.metric("Precision", f"{model_result['Precision']:.2%}")

    col3.metric("Recall", f"{model_result['Recall']:.2%}")

    col4.metric("F1 Score", f"{model_result['F1 Score']:.2%}")

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        model_result["Confusion Matrix"],
        index=["Actual Fake", "Actual Real"],
        columns=["Predicted Fake", "Predicted Real"]
    )

    st.dataframe(cm, use_container_width=True)

    st.subheader("Classification Report")

    report = pd.DataFrame(
        model_result["Classification Report"]
    ).T

    st.dataframe(
        report.style.format("{:.3f}"),
        use_container_width=True
    )


# ============================================================
# TAB 5 — TOPIC DISCOVERY
# ============================================================

with tabs[4]:

    st.header("Unsupervised Topic Discovery")

    st.write(
        "Explore similar headlines grouped by K-Means clustering."
    )

    cluster_counts = (
        clustered_df["cluster"]
        .value_counts()
        .sort_index()
    )

    st.subheader("Cluster Distribution")

    st.bar_chart(cluster_counts)

    st.divider()

    for cluster_number, words in cluster_topics.items():

        cluster_data = clustered_df[
            clustered_df["cluster"] == cluster_number
        ]

        with st.expander(
            f"Cluster {cluster_number + 1}: "
            f"{', '.join(words[:4])}"
        ):

            col1, col2, col3 = st.columns(3)

            col1.metric("Headlines", len(cluster_data))

            col2.metric(
                "Fake",
                int((cluster_data["label"] == 0).sum())
            )

            col3.metric(
                "Real",
                int((cluster_data["label"] == 1).sum())
            )

            st.write("**Top Keywords**")

            st.write(", ".join(words))

            st.write("**Sample News**")

            sample = cluster_data[["title", "label"]].head(10).copy()

            sample["label"] = sample["label"].map({
                0: "Fake",
                1: "Real"
            })

            st.dataframe(
                sample,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NewsLens AI | Machine Learning + LLM + Tool Calling + Web Evidence"
)

st.caption(
    "For educational purposes. AI-generated analysis may be incorrect."
)
