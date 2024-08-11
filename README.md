# LSTM Time Series Forecasting Flask App

## Overview

This project is a Flask web app called `futureSight` that uses Long Short-Term Memory (LSTM) neural network to forecast time series data. Users can upload CSV or Excel files containing historical data, and the application will return predictions along with a visual representation of both the predicted and actual values.

## Features

- Upload CSV or Excel files containing time series data.
- Use of an LSTM model for forecasting.
- Visualize predictions against actual historical data.
- Simple web interface for easy interaction.

## Requirements

- Python 3.7 or later
- Flask
- NumPy
- Pandas
- Torch (PyTorch)
- Matplotlib
- Scikit-learn (if you're using it for scaling)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Elprinz1/LSTM-Time-Series-Forecasting-Flask-App.git
    cd futureSight
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download or prepare your trained LSTM model and scaler:**
   - Ensure your model is saved as `model.pth`.
   - Ensure your scaler is saved as `scaler.pkl`.

5. **Create a static directory:**
   Ensure you have a directory named `static` in the project root to store plot images.

## Running the Application

To start the Flask application, run:

```bash
python app.py

```

### Accessing the Application

1. Open a web browser and go to `http://127.0.0.1:5000/`.
2. Upload a CSV or Excel file that contains the historical time series data.
3. Submit the form to receive predictions and view the results.

## File Structure

```
/futureSight
│
├── app.py               # Main Flask application file
├── model.pth            # Trained LSTM model weights
├── model.py             # Model function to define the LSTM Architecture
├── scaler.pkl           # Scaler object for preprocessing
├── templates/           # Directory for HTML templates
│   ├── upload.html      # File upload interface
├── static/              # Directory for storing static files like plots
│   └── plot.png         # Placeholder for output plots
├── utils.py             # Utility functions for preprocessing
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

## Test Data Format

For the application to work correctly, your uploaded CSV or Excel files should contain a time series dataset with at least one column for the values you want to predict (e.g., stock prices). Ensure your dataset is structured properly to facilitate the necessary preprocessing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

For questions or feedback regarding this project, please reach out via [coinhub.info@gmail.com](mailto:coinhub.info@gmail.com).
