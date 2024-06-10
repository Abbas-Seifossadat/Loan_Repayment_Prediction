import streamlit as st
import numpy as np
import pandas as pd
import json
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns

# read data
data = pd.read_csv('loan_data.csv') 
data['credit.policy'] = data['credit.policy'].map({1: 'Yes', 0: 'No'})
data['not.fully.paid'] = data['not.fully.paid'].map({1: 'Yes', 0: 'No'})
categorical_cols = data.select_dtypes(include=['object']).columns

# Set page configuration
st.set_page_config(
    page_title="Loan Repayment Prediction",
    page_icon="💰",
    initial_sidebar_state='expanded'
)

# Cache the model loading function
@st.cache_resource()
def load_model_cached():
    model = load_model('model_option_1.h5', compile=False)
    return model

with st.spinner('Model is loading...'):
    model = load_model_cached()

# Function to load bounds from a JSON file
def load_bounds(file_path):
    with open(file_path, 'r') as f:
        bounds = json.load(f)
    return bounds

# Function to handle outliers
def handle_outliers(X, bounds):
    for col, b in bounds.items():
        lower_bound = b["lower_bound"]
        upper_bound = b["upper_bound"]
        X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
        X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
    return X

# Function to get input data from user
def get_input_data():
    credit_policy = st.selectbox('Credit Policy', ['Yes', 'No'], help="Select if the customer meets the credit underwriting criteria.")
    purpose = st.selectbox('Purpose', 
                           ['all_other', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business'],
                           help="Select the purpose of the loan.")
    int_rate = st.slider('Interest Rate', min_value=0.06, max_value=0.22, step=0.01, help="Enter the interest rate for the loan.")
    installment = st.number_input('Installment', min_value=0.0, max_value=950.0, step=50.0, help="Enter the monthly installment amount.")
    annual_inc = st.number_input('Annual Income', min_value=1896, max_value=2100000, step=1000, help="Enter the annual income of the borrower.")
    dti = st.slider('DTI', min_value=0.0, max_value=30.0, step=0.5, help="Enter the debt-to-income ratio.")
    fico = st.number_input('FICO', min_value=612.0, max_value=850.0, step=30.0, help="Enter the FICO credit score.")
    days_with_cr_line = st.number_input('Days with Credit Line', min_value=178, max_value=17639, step=365, help="Enter the number of days with an open credit line.")
    revol_bal = st.number_input('Revolving Balance', min_value=0.0, max_value=1207359.0, step=1000.0, help="Enter the revolving balance.")
    revol_util = st.slider('Revolving Utilization Rate', min_value=0.0, max_value=119.0, step=5.0, help="Enter the revolving line utilization rate.")
    inq_last_6mths = st.number_input('Inquiries in Last 6 Months', min_value=0, max_value=33, step=1, help="Enter the number of inquiries in the last 6 months.")
    delinq_2yrs = st.number_input('Delinquencies in Last 2 Years', min_value=0, max_value=13, step=1, help="Enter the number of delinquencies in the last 2 years.")
    pub_rec = st.number_input('Public Records', min_value=0, max_value=5, step=1, help="Enter the number of derogatory public records.")

    input_data = pd.DataFrame({
        'credit.policy': [1 if credit_policy == 'Yes' else 0],
        'int.rate': [int_rate],
        'installment': [installment],
        'log.annual.inc': [np.log(annual_inc)],
        'dti': [dti],
        'fico': [fico],
        'revol.util': [revol_util],
        'days.with.cr.line_sqrt': [np.sqrt(days_with_cr_line)],
        'revol.bal_sqrt': [np.sqrt(revol_bal)],
        'inq.last.6mths_sqrt': [np.sqrt(inq_last_6mths)],
        'delinq.2yrs_sqrt': [np.sqrt(delinq_2yrs)],
        'pub.rec_sqrt': [np.sqrt(pub_rec)]
    })
    
    # Ensure all columns are present
    required_columns = [
        'credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'revol.util',
        'days.with.cr.line_sqrt', 'revol.bal_sqrt', 'inq.last.6mths_sqrt', 'delinq.2yrs_sqrt', 'pub.rec_sqrt',
        'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement',
        'purpose_major_purchase', 'purpose_small_business'
    ]
    
    for col in required_columns:
        if col not in input_data.columns:
            if col == 'purpose_' + purpose:
                input_data[col] = 1
            else:
                input_data[col] = 0

    return input_data

def plot_cat_dist(categorical_cols, data):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f'Distribution of {col}' for col in categorical_cols])

    for i, col in enumerate(categorical_cols):
        counts = data[col].value_counts()
        fig.add_trace(go.Bar(x=counts.index, y=counts.values, marker_color=counts.values, marker_colorscale='viridis'), row=1, col=i+1)
        fig.update_xaxes(title_text=col, row=1, col=i+1)
        fig.update_yaxes(title_text='Count', row=1, col=i+1)

    # Update layout
    fig.update_layout(height=500, width=1000, showlegend=False)

    return fig

    # Update layout
    fig.update_layout(height=500, width=1000, showlegend=False)

    # Display the plots in Streamlit
    st.plotly_chart(fig)


def plot_num_dist(data):
    # Create subplots with 4 rows and 3 columns
    fig = make_subplots(rows=4, cols=3, subplot_titles=data.select_dtypes(include='number').columns)

    # Iterate through numerical columns and add a histogram for each
    for index, col in enumerate(data.select_dtypes(include='number').columns):
        row = index // 3 + 1
        col_pos = index % 3 + 1
        fig.add_trace(go.Histogram(x=data[col]), row=row, col=col_pos)
        fig.update_xaxes(title_text=col, row=row, col=col_pos)

    # Update layout
    fig.update_layout(height=1000, width=1200)

    # Display plot in Streamlit
    st.plotly_chart(fig)


def show_introduction():
    st.title("Loan Repayment Prediction 💰")
    st.image('Loan.jpg')
    st.header("Project Objective 🎯")
    st.markdown("""
    This project leverages deep neural networks to predict the probability of borrowers successfully repaying their loans. Accurate loan repayment predictions are crucial for financial institutions to make informed lending decisions, reduce risks, and promote responsible lending practices.
    
    In this project, we try to utilize the best preprocessing techniques, we clean and prepare the data to enhance model performance.
    """)
    st.header("Source Dataset 📝")
    if st.checkbox("Show Data"):
        st.dataframe(data)
    st.markdown("[Loan Data](https://www.kaggle.com/datasets/itssuru/loan-data)")
    st.header("Data Overview 📊")
    st.markdown("""
    The dataset used in this project was obtained from Lending Club and comprises the following attributes from loan_data.csv:
    1. **credit.policy**: A binary variable indicating whether the customer meets LendingClub.com's credit underwriting criteria (1 for meeting the criteria, 0 otherwise).
    2. **purpose**: The purpose of the loan, with values such as "credit_card," "debt_consolidation," "educational," "major_purchase," "small_business," and "all_other."
    3. **int.rate**: The interest rate of the loan, represented as a proportion (e.g., 0.11 for 11%).
    4. **installment**: The monthly installment amount owed by the borrower if the loan is funded.
    5. **log.annual.inc**: The natural log of the self-reported annual income of the borrower.
    6. **dti**: The debt-to-income ratio of the borrower, calculated as the amount of debt divided by annual income.
    7. **fico**: The FICO credit score of the borrower.
    8. **days.with.cr.line**: The number of days the borrower has had a credit line.
    9. **revol.bal**: The borrower's revolving balance, representing the amount unpaid at the end of the credit card billing cycle.
    10. **revol.util**: The borrower's revolving line utilization rate, indicating the amount of the credit line used relative to the total credit available.
    11. **inq.last.6mths**: The number of inquiries by creditors made on the borrower's credit history in the last 6 months.
    12. **delinq.2yrs**: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
    13. **pub.rec**: The number of derogatory public records, including bankruptcy filings, tax liens, or judgments.
    14. **not.fully.paid**: The dependent variable, where "0" indicates that the loan was fully paid by borrowers, and "1" indicates that it was not fully paid.
    """)
    st.write("**Summary statistics for numerical features:**")
    st.write(data.describe())
    st.write("**Summary statistics for categorical features:**")
    st.write(data.describe(include="O"))
    st.write("**Unique values in each column:**")
    st.write(data.nunique())

    st.write("**Distribution of categorical columns:**")
    # Call the function to get the figure
    fig = plot_cat_dist(categorical_cols, data)
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write("**Distribution of numerical columns:**")
    plot_num_dist(data)

def show_prediction_phase():
    st.title("Loan Repayment Prediction 💰")
    st.write("Enter the loan details below to predict whether the loan will be fully paid or not.")
    
    input_data = get_input_data()
    
    bounds = load_bounds('bounds.json')
    input_data = handle_outliers(input_data, bounds)
    
    scaler = joblib.load('scaler.pkl')
    
    numerical_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'revol.util',
                      'days.with.cr.line_sqrt', 'revol.bal_sqrt', 'inq.last.6mths_sqrt', 'delinq.2yrs_sqrt', 'pub.rec_sqrt']
    
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            time.sleep(2)  # Simulate a delay for the spinner
            pred_prob = model.predict(input_data)
            pred_label = (pred_prob > 0.5).astype(int)
            
            if pred_label[0] == 1:
                st.error("The Loan not fully Paid", icon="❌")
                st.toast('Oh no!', icon='😩')
                # st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg", start_time=0)
                st.markdown(
                """
                <audio autoplay>
                    <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                </audio>
                """,
                unsafe_allow_html=True,
            )
            else:
                st.success("The Loan fully Paid", icon="✅")
                st.toast('Hooray!', icon='🎉')
                time.sleep(1)
                st.balloons()
                st.markdown(
                """
                <audio autoplay>
                    <source src="https://actions.google.com/sounds/v1/cartoon/cartoon_boing.ogg" type="audio/ogg">
                </audio>
                """,
                unsafe_allow_html=True,
            )
                # st.audio("https://actions.google.com/sounds/v1/cartoon/cartoon_boing.ogg", start_time=0)
                

def main():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select Phase", ["Introduction", "Prediction Phase"])
    
    st.sidebar.markdown("""
    Please refer to Kaggle profiles for each team member:
    
    Enigma Team Members:
    1. Mahdie Rahmati [Kaggle Profile](https://www.kaggle.com/mahdierahmati)
    2. Ali Mehrjou [Kaggle Profile](https://www.kaggle.com/alimehrjou)
    3. Ali Shokouhy [Kaggle Profile](https://www.kaggle.com/alishokouhy)
    4. Rasul Noshad [Kaggle Profile](https://www.kaggle.com/rasulnoshad)
    5. Abbas Seifossadat [Kaggle Profile](https://www.kaggle.com/abbasseifossadat)
    """)
    
    if option == "Introduction":
        show_introduction()
    elif option == "Prediction Phase":
        show_prediction_phase()

    st.sidebar.markdown(
        """
        <div style="text-align:center;">
            <a href="https://github.com/Abbas-Seifossadat/Loan_Repayment_Prediction/tree/main">
                <img src="https://img.icons8.com/ios-filled/25/000000/github.png" alt="GitHub"/>
            </a>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
