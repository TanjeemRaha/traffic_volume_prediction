# Learning guide for a non-CS reader

You do not need to understand every Python statement before running this
project. Begin with the visible outputs, then trace one stage at a time. The
scripts at the project root are short entry points; detailed reusable work is
kept in the `traffic_prediction` folder so it can be found by topic.

## A useful learning order

### 1. Confirm that the workflow runs

The existing `.venv` is verified with Python 3.12.10. Activate it as shown in the
[README](../README.md), then use:

```powershell
python run_project.py --fast
```

This quick run checks installation, data access, cleaning, plotting, training,
and saving. A successful fast run is more useful than reading all modules before
Python works.

### 2. Inspect the data audit

Open `data/processed/data_quality_report.json`. Focus first on:

- `rows_received`;
- `exact_duplicate_rows_removed`;
- `duplicate_timestamps_aggregated`;
- `conflicting_target_timestamps`;
- `clean_hourly_rows`;
- `unobserved_hours_between_start_and_end`;
- `remaining_missing_values_by_column`.

Then open the top rows of `i94_traffic_cleaned.csv`. Each row should represent
one recorded hour. The target column is `traffic_volume`.

### 3. View the transportation patterns

Read `outputs/reports/eda_summary.md` and open every PNG in
`outputs/figures/`. Ask engineering questions before model questions:

- At what hours do volumes peak?
- How do weekday and weekend profiles differ?
- Are all months represented equally?
- Are there gaps in the time series?
- Does an apparent weather pattern remain plausible after considering hour and
  season?

An exploratory plot can suggest a relationship. It does not prove causation.
The default EDA contains only the pre-2017 training period. In particular,
`traffic_pattern_by_weather.csv` and
`05b_traffic_by_weather_condition.png` compare descriptive weather groups in
that training period. Some hours have multiple combined conditions and appear
in more than one group.

Only after the code and evaluation rules have been frozen, a retrospective
full-record description can be generated with:

```powershell
python 02_explore_data.py --include-held-out-periods
```

This overwrites the standard EDA outputs. Rerun step 2 without the option to
restore training-only figures.

### 4. Compare models

Open `outputs/metrics/model_comparison.csv` in a spreadsheet. Filter or read the
`partition` column:

- candidate rows labeled `validation` were used to select a model;
- exactly one row labeled `final_test` reports the chosen model on the later
  2018 chronological holdout.

Start with `MAE_vehicles_per_hour`, then inspect RMSE, R-squared, runtime, and
grouped errors. A good project does not simply announce the smallest number; it
checks whether rush hours and other important operating periods are acceptable.

For the verified normal run, the hour-of-week baseline had the lowest validation
MAE, 255.6 vehicles/hour, and was selected. Its 2018 holdout results were MAE
252.2 vehicles/hour, RMSE 481.3 vehicles/hour, R-squared 0.941, and WAPE 7.6%.
Linear Regression had lower validation RMSE (426.5) but higher validation MAE
(288.9), so it did not win the predeclared MAE rule. The weather-using candidates
did not improve the winning MAE; this does not mean weather is physically
irrelevant.

Treat those numbers as a verified reference snapshot. The generated CSV, report,
and metadata files are authoritative for the latest run.

### 5. Inspect individual errors

Open `outputs/predictions/final_test_predictions.csv`. Important columns are:

- `actual_traffic_volume`: observed count;
- `predicted_traffic_volume`: model estimate;
- `residual_actual_minus_predicted`: positive means underprediction, negative
  means overprediction;
- `absolute_error`: magnitude of the miss;
- hour, day, travel period, holiday, and weather context.

Sort by `absolute_error` from largest to smallest. These failures often reveal
more about practical limitations than the overall metric.

Also open `outputs/predictions/example_prediction.csv`. The columns
`inputs_used`, `uses_weather`, and `ignored_inputs` prevent a common
misinterpretation: the current schedule-only winner uses date/time converted to
hour of week and ignores the supplied weather and holiday fields. A different
winner after retraining could use those fields.

## Code tour in reading order

Do not begin in the longest file. Follow the execution path:

1. `run_project.py` shows the complete sequence.
2. `01_download_and_clean.py` calls the public functions in
   `traffic_prediction/data.py`.
3. `traffic_prediction/paths.py` defines every input and output location.
4. `traffic_prediction/data.py` validates and aggregates raw observations.
5. `02_explore_data.py` calls `traffic_prediction/visualization.py` to make plots
   and summaries.
6. `traffic_prediction/features.py` turns time/weather into model inputs and
   defines the chronological partitions.
7. `traffic_prediction/modeling.py` defines preprocessing and the model ladder.
8. `traffic_prediction/evaluation.py` defines the four metrics and grouped
   summaries.
9. `traffic_prediction/training.py` compares on validation data, refits the
   winner, evaluates the later chronological holdout, and saves results.
10. `traffic_prediction/prediction.py` validates a scenario and applies the
    saved artifact; `04_make_prediction.py` provides the beginner-facing command
    for it.

Search for a function name to connect a short script with its implementation.
For example, `prepare_data` leads from step 1 to `data.py`, and
`train_and_evaluate` leads from step 3 to `training.py`.

## The same workflow in engineering language

| Machine-learning term | Familiar engineering analogy |
|---|---|
| Observation/row | One hourly field record |
| Feature/input | Explanatory variable used in an estimating equation |
| Target | Quantity to estimate: vehicles/hour |
| Fit/train | Calibrate coefficients or rules using historical records |
| Validation | Compare candidate methods on a later calibration-check period |
| Chronological holdout/test | Later-period performance check after within-run selection; confirmatory only if it was not used during development |
| Prediction | Estimated hourly volume |
| Residual | Observed minus estimated volume |
| Pipeline | Fixed sequence for data preparation and estimation |
| Hyperparameter | Analyst-selected model setting, such as number of trees |
| Overfitting | Calibrating historical detail that does not transfer to later data |
| Leakage | Letting future or answer information enter calibration improperly |
| One-hot encoding | Converting a category into separate yes/no columns |
| Imputation | Replacing a missing input using a rule learned from training data |
| Random seed | Fixed starting state that makes stochastic calculations repeatable |

## Understanding the most important feature choices

### Why use hour of week?

There are 24 hours x 7 days = 168 weekly hour positions. Treating these as
categories lets the model distinguish Monday at 08:00 from Sunday at 08:00. It
captures commuting structure more directly than using only "hour = 8."

### Why transform rain and snow?

Most hours have little or no precipitation, while a few have much larger values.
`log(1 + value)` compresses that long range without deleting storm observations.
Adding 1 makes the transformation defined when precipitation is zero.

### Why sine and cosine for season?

Calendar day is circular: December 31 and January 1 are neighbors. A raw day
number makes them look far apart. A sine/cosine pair represents their location
on an annual circle.

### Why no previous traffic volume?

Lagged volume can improve short-term forecasts, but it changes the operational
question and requires a reliable live traffic feed. Missing recorded hours also
make a simple row shift unsafe. This first implementation deliberately asks how
far calendar and same-hour weather can go on their own.

## Making a report from the outputs

A defensible short report can use this order:

1. State the station, direction, period, target, and data citation.
2. Report duplicate handling, missing-hour count, and chronological split.
3. Present training-period traffic patterns; if full-record retrospective EDA
   was requested, label it clearly and state that held-out targets are included.
4. Compare validation MAE against both engineering baselines.
5. Name the model selected before within-run holdout evaluation.
6. Report 2018 chronological-holdout MAE, RMSE, R-squared, and WAPE with units.
7. Discuss peak/travel-period errors and several largest residuals.
8. Describe weather relationships as associations.
9. State spatial, temporal, and omitted-variable limitations.

Record whether `--fast` or `--include-neural-network` was used. The generated
`experiment_metadata.json` preserves this information.

## Safe experiments after the default run

Change one item at a time, rerun, and copy the original output folder if you
need to preserve a comparison.

- Include the optional MLP and check whether validation MAE improves by at least
  the required 2% over the best classical model. Its candidate early-stopping
  subset comes only from pre-2017 training data.
- Change one Random Forest setting in `modeling.py` and record both validation
  error and runtime.
- Add a grouped error summary for a civil-engineering category that matters to
  your study.
- Add a new plot while leaving the model selection rule unchanged.

Do not select a model after looking for the best 2018 result. Repeated use of a
holdout for development makes it validation data and can make accuracy
optimistic. The current 2018 result is explicitly a chronological
holdout/reproducibility demonstration, not an untouched or preregistered
confirmation, because it was rerun after a runtime-motivated `LinearSVR` solver
change. For future confirmation, freeze the code and rules first and use new
later or local data.

## Common beginner mistakes

- Running a numbered step before its required earlier output exists.
- Installing packages outside `.venv`, then running Python inside `.venv`.
- Randomly splitting a time series because many introductory examples do so.
- Treating rows with repeated timestamps as independent traffic counts.
- Including `traffic_volume` or a value derived from the same target among the
  model inputs.
- Saying weather "caused" a change based on feature importance.
- Reporting MAE without the unit vehicles/hour.
- Assuming the most complex model must be the best.
- Generalizing one westbound station to an entire road network.

## If Python still does not run

The project environment on this computer is verified with Python 3.12.10. If an
old terminal still opens the Microsoft Store alias, open a new PowerShell window
or bypass PATH with:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe run_project.py --fast
```

For another computer, return to the README's [fresh Windows
setup](../README.md#setup-on-a-fresh-windows-computer). Do not debug project code
until either the activated `python --version` or the direct `.venv` command
prints a real version.
