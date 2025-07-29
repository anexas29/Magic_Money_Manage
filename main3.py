import streamlit as st
import pandas as pd
import os
st.write("Installed packages:", os.popen("pip list").read())

import matplotlib.pyplot as plt
import joblib
import google.generativeai as genai
from io import StringIO

#  Load local model & vectorizer
model = joblib.load("expense_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#  Configure Gemini
api_key = st.secrets["api_keys"]["google_api_key"]
genai.configure(api_key=api_key)
genai_model = genai.GenerativeModel("gemini-2.5-flash")

#  Streamlit UI
st.title("💸 Magic_Money_Manage — Smart Expense Analyzer")
st.write("Upload CSV → Gemini standardizes it → Saved as data.csv → Local ML → Analyze Debits & Credits!")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write("### Raw CSV Preview")
    st.dataframe(df_raw)

    if st.button("✨ Standardize & Save with Gemini"):
        st.info("Standardizing with Gemini...")

        # Take sample rows for prompt
        lines = []
        for idx, row in df_raw.iterrows():
            line = ", ".join([f"{col}: {row[col]}" for col in df_raw.columns])
            lines.append(line)
        joined_rows = "\n".join(lines)

        prompt = f"""
        You are a smart finance NLP assistant.

        Here are raw bank transactions:

        {joined_rows}

        ➡️ Task:
        For each row, extract:
        Date, Description, Amount, Transaction (Debit/Credit), Category (Food & Dining, Transportation, Entertainment, Utilities, Shopping, Subscriptions, Healthcare, Other)

        Return as plain CSV with header:
        Date,Description,Amount,Transaction,Category

        Only output CSV rows — no explanations.
        """

        response = genai_model.generate_content(prompt)
        csv_text = response.text.strip()

        df_clean = pd.read_csv(StringIO(csv_text))
        df_clean.to_csv("data.csv", index=False)

        st.success("✅ Standardized data saved as data.csv")
        st.dataframe(df_clean)

        # Reload
        df = pd.read_csv("data.csv")

        # Local ML prediction
        X_new = df['Description']
        X_new_tfidf = vectorizer.transform(X_new)
        local_pred = model.predict(X_new_tfidf)
        df['ML_Predicted_Category'] = local_pred  # just for testing

        # Split Debit & Credit
        df['Transaction'] = df['Transaction'].str.lower()
        debit_df = df[df['Transaction'] == 'debit'].copy()
        credit_df = df[df['Transaction'] == 'credit'].copy()

        debit_df['Date'] = pd.to_datetime(debit_df['Date'], errors='coerce')
        debit_df = debit_df.dropna(subset=['Date'])
        debit_df = debit_df.sort_values('Date')

        total_debit = debit_df['Amount'].sum()
        total_credit = credit_df['Amount'].sum()
        difference = total_credit - total_debit

        st.write(f"💰 **Total Debit:** ₹{total_debit:,.2f}")
        st.write(f"💵 **Total Credit:** ₹{total_credit:,.2f}")
        st.write(f"🟢 **Net Difference (Credit - Debit):** ₹{difference:,.2f}")

        if not debit_df.empty:
            daily = debit_df.groupby('Date')['Amount'].sum().reset_index()
            fig1, ax1 = plt.subplots()
            ax1.bar(daily['Date'], daily['Amount'])
            ax1.set_title("Daily Debit Amounts")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Total Debit")
            st.pyplot(fig1)

            cat = debit_df.groupby('Category')['Amount'].sum()
            fig2, ax2 = plt.subplots()
            ax2.pie(cat, labels=cat.index, autopct='%1.1f%%')
            ax2.set_title("Spending by Gemini Category (Debit)")
            st.pyplot(fig2)

        else:
            st.warning("No debit transactions found!")
