# LLM Prompt for S&P 500 Trading Strategy Code

Please act as an expert Python developer and data scientist. I need you to generate a foundational Python script to build a machine learning trading strategy for the S&P 500 index. The code should be well-structured, easy to understand for a beginner, and follow best practices.

**Project Requirements:**

1.  **Project Setup:** Provide the necessary shell commands to set up a new Python virtual environment and initialize a Git repository.
2.  **Required Libraries:** The script should use `pandas` for data manipulation, `scikit-learn` for machine learning, `yfinance` for price data, and **`fredapi` for macroeconomic data**.
3.  **Data Acquisition:**
    * Fetch daily historical S&P 500 price data using the `yfinance` library.
    * Use the `fredapi` library to fetch the following specific macroeconomic factors:
        * **Federal Funds Rate (`FEDFUNDS`)**
        * **Unemployment Rate (`UNRATE`)**
        * **Consumer Price Index (`CPIAUCSL`)**
        * **10-Year Treasury Yield (`DGS10`)**
    * Include a placeholder for the user's FRED API key and a note on how to obtain one.
4.  **Feature Engineering:**
    * Combine the S&P 500 data and the macroeconomic data into a single `pandas` DataFrame, ensuring they are aligned by date.
    * Create at least two common technical indicators from the price history (e.g., a simple moving average and another relevant indicator).
    * The code must handle the interpolation of monthly and weekly macroeconomic data to a daily frequency (e.g., using a forward-fill method).
5.  **Target Variable:**
    * Create a binary target variable `y` that represents a future price movement. The simplest approach is to calculate the percentage change of the S&P 500's close price five days in the future. The label should be '1' if the future change is positive (suggesting "Buy"), and '0' if it's negative or zero (suggesting "Sell" or "Hold"). Crucially, the code must handle shifting this target variable to prevent look-ahead bias.
6.  **Model Training:**
    * Split the dataset into features (`X`) and the target variable (`y`). The features should include both the technical indicators and the macroeconomic factors.
    * Perform a time-based split for the training and test sets (e.g., 80% for training, 20% for testing), as a random split is inappropriate for time-series data.
    * Train a simple scikit-learn classification model, such as `LogisticRegression` or `RandomForestClassifier`.
7.  **Signal Generation:**
    * Use the trained model's `predict_proba()` method to get a probability score for the "Buy" signal (class 1).
    * Implement a function that applies a user-defined threshold to this probability score, converting it into a final binary signal (`1` for Buy, `0` for Sell).
8.  **Backtesting:**
    * Include a basic backtesting function. This function should iterate through the test data, apply the generated buy/sell signals, and calculate a few simple performance metrics like the total return.
9.  **Code Quality:**
    * The final script must be heavily commented to explain each step, especially the data acquisition, feature engineering, and target variable creation.
    * Use clear and descriptive variable names.
    * Structure the code cleanly, perhaps using a main function or a `if __name__ == "__main__":` block to run the entire script.

Please provide a single, complete Python script that meets all of these requirements.
