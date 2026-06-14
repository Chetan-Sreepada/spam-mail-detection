import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Page config
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="centered"
)

# Title and description
st.title("📧 Spam Mail Detector")
st.markdown("Enter an email or message below to check if it's **Spam** or **Ham** (legitimate).")

# Load model and vectorizer
@st.cache_resource
def load_model():
    # Note: You need to train and save these files first
    # For now, we'll retrain from the notebook or load if exists
    try:
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Training function (run once when app starts)
def train_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # Load data - update path to your CSV location
    raw_mail_data = pd.read_csv('mail_data.csv')
    
    # Replace null values
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
    
    # Label encoding
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    
    # Separate features and labels
    X = mail_data['Message']
    Y = mail_data['Category'].astype(int)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    
    return model, feature_extraction

# Main app logic
def main():
    # Try to load existing model, otherwise train
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        with st.spinner("Training model for first time..."):
            model, vectorizer = train_model()
            # Save for future runs
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
        st.success("Model trained successfully!")
    
    # Text input
    user_input = st.text_area(
        "Message content:",
        height=200,
        placeholder="Paste your email or message here..."
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Check Message", use_container_width=True)
    
    # Results
    if predict_button:
        if not user_input.strip():
            st.warning("Please enter a message to check.")
            return
        
        # Transform input
        input_features = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_features)[0]
        
        # Display result with styling
        if prediction == 1:
            st.success("### ✅ HAM (Legitimate Message)")
            st.markdown("This appears to be a legitimate message.")
        else:
            st.error("### ⚠️ SPAM (Suspicious Message)")
            st.markdown("This appears to be spam or promotional content.")
        
        # Optional: Show prediction probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_features)[0]
            st.markdown("---")
            st.markdown("**Confidence:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Spam Probability", f"{proba[0]*100:.1f}%")
            with col2:
                st.metric("Ham Probability", f"{proba[1]*100:.1f}%")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        This detector uses:
        - **Logistic Regression** algorithm
        - **TF-IDF** feature extraction
        - Trained on SMS spam dataset
        
        **Label meanings:**
        - 🟢 HAM = Legitimate message
        - 🔴 SPAM = Junk/suspicious message
        """)
        
        st.markdown("## Example messages")
        with st.expander("Show examples"):
            st.markdown("**Ham example:**")
            st.code("Hey, want to grab coffee tomorrow?")
            st.markdown("**Spam example:**")
            st.code("WINNER! You've won a free iPhone. Click here to claim.")

if __name__ == "__main__":
    main()