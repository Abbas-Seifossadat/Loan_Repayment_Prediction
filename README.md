# Loan Repayment Prediction

This project leverages deep neural networks to predict the probability of borrowers successfully repaying their loans. Accurate loan repayment predictions are crucial for financial institutions to make informed lending decisions, reduce risks, and promote responsible lending practices.

![Loan Repayment](Loan.jpg)


## Project Overview

- **Objective**: Predict whether a loan will be fully paid or not using various borrower attributes.
- **Dataset**: The dataset used is from Lending Club, which includes features like credit policy, interest rate, annual income, and more.

## Features

- **User Interface**: Developed using Streamlit for easy interaction.
- **Model**: A pre-trained deep neural network model.
- **Data Processing**: Handles outliers and scales numerical features.
- **Prediction**: Provides a probability of loan repayment along with a success/failure indication.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/loan-repayment-prediction.git
    cd loan-repayment-prediction
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

4. **Interact with the Application**:
    - Navigate to the `Prediction Phase` in the sidebar.
    - Enter loan details and click 'Predict' to see the results.

## Files

- `app.py`: Main application file.
- `model_option_1.h5`: Pre-trained model.
- `bounds.json`: JSON file containing bounds for outlier handling.
- `scaler.pkl`: Pre-trained scaler for numerical features.
- `requirements.txt`: List of required Python packages.

## Team Members

- Mahdie Rahmati [Kaggle Profile](https://www.kaggle.com/mahdierahmati)
- Ali Mehrjou [Kaggle Profile](https://www.kaggle.com/alimehrjou)
- Ali Shokouhy [Kaggle Profile](https://www.kaggle.com/alishokouhy)
- Rasul Noshad [Kaggle Profile](https://www.kaggle.com/rasulnoshad)
- Abbas Seifossadat [Kaggle Profile](https://www.kaggle.com/abbasseifossadat)

