import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ---------------------------------------------------------
# BUILD PRODUCT SIMILARITY AT RUNTIME 
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def build_product_similarity():
    df = pd.read_csv("https://drive.google.com/uc?id=1xVV5c_X4ZGZd3QEXQt-u_5JPwzmfizT6")

    # Clean data
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]

    # Create product-customer matrix
    product_matrix = (
        df.pivot_table(
            index="Description",
            columns="CustomerID",
            values="Quantity",
            aggfunc="sum",
            fill_value=0
        )
    )

    # Compute cosine similarity
    similarity = cosine_similarity(product_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=product_matrix.index,
        columns=product_matrix.index
    )

    return similarity_df


similarity_df = build_product_similarity()

# ---------------------------------------------------------
# LOAD CLUSTERING MODELS
# ---------------------------------------------------------
try:
    kmeans_model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    kmeans_model, scaler = None, None

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page):
    st.session_state.page = page

# ---------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------
col1, col2 = st.columns([1, 4])

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with col1:
    st.markdown("""
        <style>
         {
            background-color: #1e1e1e;
            padding: 25px;
            border-radius: 15px;
            height: 85vh;
            border-right: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 4px 0 15px rgba(0, 0, 0, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)

    # SIDEBAR HEADER
    st.caption("#### E-Commerce Solutions")

    # NAVIGATION BUTTONS
    st.button(
        "Home", 
        on_click=set_page, 
        args=("Home",), 
        use_container_width=True, 
        type="primary" if st.session_state.page == "Home" else "secondary"
    )

    st.button(
        "Clustering", 
        on_click=set_page, 
        args=("Clustering",), 
        use_container_width=True, 
        type="primary" if st.session_state.page == "Clustering" else "secondary"
    )

    st.button(
        "Recommendation", 
        on_click=set_page, 
        args=("Recommendation",), 
        use_container_width=True, 
        type="primary" if st.session_state.page == "Recommendation" else "secondary"
    )
# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------
with col2:
    page = st.session_state.page

    # ---------------------- CONTENT AREA ----------------------
# Logic to display headers based on page selection
with col2:
    if st.session_state.page == "Home":
        st.markdown("<h1 style='text-align:center;'>Welcome to Shopper Spectrum Dashboard</h1>", unsafe_allow_html=True)

    # ---------------- HOME ----------------
    if page == "Home":
        st.write("##### Shopper Spectrum is your all-in-one solution for enhancing e-commerce experiences through advanced Product Recommendation and Customer Segmentation techniques.")
        st.write("###### This application provides two main functionalities:")
        
        # ----------------------------------------------------------------
        # PRODUCT RECOMMENDATION SECTION
        # ----------------------------------------------------------------
        st.subheader("Product Recommendation")
        st.write("- **Personalized Recommendations:** Suggests products similar to those viewed or purchased by the user.")
        st.write("- **Fast Processing:** Utilizes efficient algorithms to deliver recommendations in real-time.")
        
        # ----------- Professional Metric Cards Row 1 --------------------
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Total Products</h4>
                <h2 style="color:white;">{}</h2>
            </div>
            """.format(len(similarity_df) if 'similarity_df' in locals() else 0), unsafe_allow_html=True)

        with m2:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Similarity Features</h4>
                <h2 style="color:white;">Cosine Similarity</h2>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Avg Recommendation Speed</h4>
                <h2 style="color:white;">Under 200 ms</h2>
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------------------
        # CUSTOMER SEGMENTATION SECTION
        # ----------------------------------------------------------------
        st.subheader("Customer Segmentation")
        st.write("- **Behavioral Insights:** Segments customers based on purchasing behavior for targeted marketing.")
        st.write("- **Data-Driven Decisions:** Helps businesses make informed decisions based on customer segments.")

        # ----------- Professional Metric Cards Row 2 --------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Total Segments</h4>
                <h2 style="color:white;">4</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Model Type</h4>
                <h2 style="color:white;">KMeans Clustering</h2>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">Normalization Used</h4>
                <h2 style="color:white;">StandardScaler</h2>
            </div>
            """, unsafe_allow_html=True)
        
        #------Additional Metrics----------------------------------------
        st.markdown("### Additional Insights")

        # ---------------- Additional Metrics Row ------------------------
        i1, i2 = st.columns(2)

        with i1:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">💰 Avg Customer Value (Sample)</h4>
                <h2 style="color:white;">₹ 4,500</h2>
            </div>
            """, unsafe_allow_html=True)

        with i2:
            st.markdown("""
            <div style="background:#222626; padding:20px; border-radius:10px; text-align:center;
                        box-shadow:0px 3px 8px rgba(0,0,0,0.4);">
                <h4 style="color:#ccc;">👑 At-Risk Customer Ratio</h4>
                <h2 style="color:white;">22%</h2>
            </div>
            """, unsafe_allow_html=True)


    # ---------------- RECOMMENDATION ----------------
    elif page == "Recommendation":
        st.title("Product Recommendation System")

        product_list = ["Select a Product"] + sorted(similarity_df.index.tolist())
        product_name = st.selectbox("Choose a product", product_list)

        if st.button("Get Recommendations"):
            if product_name == "Select a Product":
                st.warning("Please select a product.")
            else:
                st.subheader("Top 5 Similar Products")
                similar_items = similarity_df[product_name].sort_values(ascending=False)[1:6]
                for i, item in enumerate(similar_items.index, 1):
                    st.write(f"{i}. {item}")

    # ---------------- CLUSTERING ----------------
    elif page == "Clustering":
        st.title("Customer Segmentation (RFM Analysis)")

        recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
        frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1000.0, format="%.2f")

        if st.button("Predict Segment"):

            # Validation checks
            if recency == 0 and frequency == 0 and monetary == 0:
                st.warning("⚠  Enter valid customer values before predicting.")
            elif recency == 0:
                st.error("❌ Recency cannot be 0. Enter days since last purchase.")
            elif frequency == 0:
                st.error("❌ Frequency must be at least 1.")
            elif monetary == 0:
                st.error("❌ Monetary value cannot be 0.")
            else:
                if 'scaler' in locals() and 'kmeans_model' in locals():
                    # Prediction
                    new_data = pd.DataFrame([[recency, frequency, monetary]],
                                            columns=["Recency", "Frequency", "Monetary"])
                    scaled = scaler.transform(new_data)
                    cluster = kmeans_model.predict(scaled)[0]

                    segment_map = {
                        0: "High-Value Customer",
                        1: "Regular Buyer",
                        2: "Occasional Shopper",
                        3: "At-Risk Customer"
                    }

                    st.success(f"This customer belongs to: **{segment_map.get(cluster)}**")
                else:
                    st.error("Models not loaded.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption("© 2026 Shopper Spectrum | Built with Streamlit")
st.caption("Developed by Vadla Shiva Kumar")
