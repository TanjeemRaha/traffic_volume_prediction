# I-94 Hourly Traffic-Volume Prediction

This project predicts westbound hourly traffic volume at Interstate 94 ATR
station 301 between Minneapolis and St. Paul. It downloads and cleans the UCI
Metro Interstate Traffic Volume dataset, explores traffic patterns, compares
regression models, evaluates the selected model chronologically, and saves the
results.

The prediction target is `traffic_volume`, measured in vehicles per hour.

## Skills useful for this codebase

![Python](https://img.shields.io/badge/Python-Basics-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Cleaning-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-4DABCF?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-E34F26)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Neural Network](https://img.shields.io/badge/MLP-Neural%20Network-8E44AD)

- Basic Python: functions, modules, command-line arguments, and virtual
  environments
- pandas and NumPy: tabular data cleaning and numerical operations
- Data visualization: reading and creating Matplotlib charts
- Machine learning: regression, preprocessing, chronological validation, and
  metrics such as MAE, RMSE, R-squared, and WAPE
- Optional neural-network knowledge: understanding a small multilayer
  perceptron (MLP)

Deep-learning experience is not required. The main workflow starts with simple
baselines and classical machine-learning models.

## Project workflow

1. Download and validate the public UCI dataset.
2. Clean the observations and combine duplicate hourly weather descriptions.
3. Explore traffic patterns by hour, weekday, month, and weather.
4. Split the data chronologically into training, validation, and test periods.
5. Compare baseline, linear, support-vector, random-forest, and optional neural
   network models.
6. Select the model with the lowest validation MAE and evaluate it on the final
   test period.
7. Save the model, metrics, plots, reports, and example prediction.

## Models

| Model | Purpose |
|---|---|
| Global median | Minimum reference baseline |
| Hour-of-week median | Interpretable weekly traffic-pattern baseline |
| Linear Regression | Simple linear regression benchmark |
| Linear SVR | Support-vector regression benchmark |
| Random Forest | Nonlinear tree-based benchmark |
| Small MLP | Optional neural-network comparison |

Logistic Regression is not included because traffic volume is a continuous
value rather than a class label.

## Installation

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the project

Run the complete workflow:

```bash
python run_project.py
```

Run a quicker version with fewer model iterations:

```bash
python run_project.py --fast
```

Include the optional neural network:

```bash
python run_project.py --include-neural-network
```

Run each stage separately:

```bash
python 01_download_and_clean.py
python 02_explore_data.py
python 03_train_models.py
python 04_make_prediction.py
```

The numbered scripts must be run in order.

## Main outputs

| Location | Contents |
|---|---|
| `data/processed/` | Cleaned hourly dataset and quality report |
| `models/` | Saved selected model |
| `outputs/figures/` | Traffic-pattern and evaluation plots |
| `outputs/metrics/` | Model comparison and grouped error tables |
| `outputs/predictions/` | Test predictions and example prediction |
| `outputs/reports/` | EDA summary, model results, and experiment metadata |

See [METHODOLOGY.md](docs/METHODOLOGY.md) for the experiment design and
[LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) for a beginner-friendly walkthrough.

## Data source

Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning
Repository. [Dataset page](https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume) |
[DOI](https://doi.org/10.24432/C5X60B)

The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
