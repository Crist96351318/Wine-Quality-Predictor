# Wine Quality Predictor 🍷

## Overview
This project is an end-to-end Machine Learning pipeline developed for the IT325 Performance Innovative Task (PIT) at the University of Science and Technology of Southern Philippines (USTP). It features a clean, web-based UI that allows non-technical stakeholders to input the physicochemical properties of a wine and instantly receive a quality prediction rating.

## Author
**Fabio Joseph M. Tugonon**

## Tech Stack
* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
* **Frontend:** HTML5, CSS3 (CSS Grid Layout)

## Project Structure
```text
Wine_Project/
├── app.py              # Flask server and routing
├── train_model.py      # ML Model Training Script
├── WineQT.csv          # Dataset containing 1,143 rows of wine features
├── model.pkl           # Saved Random Forest model (generated after training)
└── templates/
    └── index.html      # Frontend web interface

------Installation & Setup (Windows)--------
1. Open the project folder in VS Code.
2. Create and activate a virtual environment:
  python -m venv venv
  .\venv\Scripts\activate
3. Install the required dependencies:
  pip install flask pandas scikit-learn numpy

------Usage---------
1. Train the Model:
  python train_model.py
2. Start the Web Application:z
  python app.py
3. Access the Interface:
Open your web browser and navigate to http://127.0.0.1:5000.
