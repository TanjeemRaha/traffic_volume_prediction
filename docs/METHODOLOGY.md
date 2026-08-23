# Methodology and engineering assumptions

This document explains what the experiment measures, how information leakage is
prevented, and what can and cannot be concluded. It is written so that a civil
engineer can audit the workflow without first studying computer science.

## 1. Study question

For a recorded hour at one westbound I-94 count station, compare estimates of
the reported traffic volume from:

- simple historical/schedule references, including weekly hour; and
- machine-learning candidates that can use calendar position plus same-hour
  temperature, rain, snow, cloud cover, and broad weather condition.

The response variable is a continuous vehicle count, so this is **regression**.
It is not classification, and therefore Logistic Regression is not a suitable
model for the main question. Logistic Regression would become relevant only
after defining a separate categorical outcome such as "volume above a
training-derived threshold."

The project tests whether the calendar-and-weather candidates improve the
predeclared validation metric over simpler schedule baselines. It does not force
the selected model to use every available field. If a weather-using candidate
wins, its task is best described as **same-hour calendar-and-weather
estimation**, and a real future estimate requires a weather forecast for that
hour. The current verified winner is schedule-only and uses local date/time
converted to hour of week.

## 2. Source data and scope

The UCI dataset contains 48,204 source records dated from 2012 through 2018. It
reports hourly westbound traffic at Minnesota DOT ATR station 301, approximately
midway between Minneapolis and St Paul, together with weather and holiday
fields.

| Source field | Engineering meaning | Treatment |
|---|---|---|
| `date_time` | Local hour of observation | Parse, sort, and derive calendar inputs |
| `holiday` | US national/regional holiday label | Expand the source label across observed hours on that date |
| `temp` | Average temperature, kelvin | Plausibility check; convert to degrees Celsius |
| `rain_1h` | Rain during the hour, mm | Plausibility check; use `log(1 + rain)` |
| `snow_1h` | Snow during the hour, mm | Plausibility check; use `log(1 + snow)` |
| `clouds_all` | Cloud cover, percent | Require the physical range 0-100% |
| `weather_main` | Broad weather description | Convert conditions into yes/no model inputs |
| `weather_description` | Detailed weather description | Preserve in the cleaned audit data; exclude from the model set |
| `traffic_volume` | Hourly westbound count | Prediction target; never an input |

UCI calls the timestamps local CST. The file does not supply an unambiguous UTC
offset, so this project retains timezone-naive local timestamps. Daylight-saving
interpretation is therefore a source limitation.

## 3. Cleaning decisions

The cleaning stage is conservative: it removes information only when it cannot
be used safely.

1. Required columns are checked before any analysis.
2. Exact repeated rows are removed.
3. Dates and numeric values are parsed; rows without a valid timestamp or a
   nonnegative target are removed and counted in the quality report.
4. Clearly impossible weather readings are changed to missing values. Broad
   limits are 180-340 K for temperature, 0-500 mm for hourly rain, 0-200 mm for
   hourly snow, and 0-100% for cloud cover. Unusual but possible traffic and
   storm observations remain.
5. Multiple records can represent the same timestamp because the source lists
   more than one weather description. Those records become one hourly row:
   numeric weather uses the median, text conditions are combined, and the one
   traffic count is preserved.
6. If duplicate records for a timestamp contain conflicting traffic targets,
   execution stops rather than choosing a target silently.
7. Missing hours between observations are reported but are not invented or
   interpolated.

All counts and decisions are written to
`data/processed/data_quality_report.json`. Duplicate timestamps are collapsed
before splitting, which prevents copies of one observation from appearing in
both training and evaluation data.

## 4. Model inputs

The raw timestamp is not passed directly to a machine-learning candidate. The
available candidate feature set contains:

- `hour_of_week`: 0-167, allowing Monday 08:00 to differ from Sunday 08:00;
- month and holiday name;
- sine and cosine of day of year, representing the annual cycle without an
  artificial jump between December 31 and January 1;
- days since the first observation, representing a gradual long-term trend;
- temperature in degrees Celsius, cloud percentage, and log-transformed rain
  and snow;
- yes/no flags for broad conditions such as rain, snow, fog, clouds, and clear.

The code intentionally creates **no traffic target lags**: no volume from the
current hour, previous hour, previous day, or previous week is an input. This
choice makes a clean calendar-and-weather study, avoids subtle leakage through
missing hours, and allows estimation without a live traffic feed. The tradeoff
is weaker response to incidents, work zones, events, and sudden demand changes.

Missing numeric inputs are replaced with a training-period median. Categorical
inputs are filled with their most common training value and converted to
indicator columns. Scaling and encoding are kept inside each model pipeline, so
they cannot learn from the validation or test future.

The two historical baselines deliberately use subsets of these inputs. The
global median uses no scenario input; the hour-of-week baseline uses only local
date/time converted to `hour_of_week`. The saved artifact and each one-hour
prediction report `inputs_used`, `uses_weather`, and `ignored_inputs`, so a
schedule-only winner is not misrepresented as a weather model.

By default, descriptive EDA uses only pre-2017 training observations. This keeps
2017 validation and 2018 holdout targets out of exploratory model development.
After the code and evaluation rules are frozen, the explicit command below may
be used for retrospective description of the full record:

```powershell
python 02_explore_data.py --include-held-out-periods
```

It overwrites the same EDA outputs. The default weather chart and table are
descriptive training-period summaries; hours with multiple combined weather
conditions appear in more than one weather group. Neither those group means nor
model importance establish causal weather effects.

## 5. Chronological experiment

Randomly mixing hours would let a model learn from the future and then be tested
on the past. That is unrealistically easy. This project keeps the order of time.

| Partition for the UCI data | Role | Permitted use |
|---|---|---|
| 2012-2016 | Training | Fit every candidate model and its preprocessing |
| 2017 | Validation | Compare candidates and choose the lowest MAE |
| Available part of 2018 | Final chronological holdout | Evaluate the method chosen from 2017 within that run |

After selection, a fresh copy of the chosen model is fitted on training plus
validation data. It is then applied to 2018. Within a run, 2018 results do not
choose the model. The final 2018 coverage ends on September 30 rather than
covering a complete calendar year, which limits seasonal interpretation.

The 2018 result in this repository must not be described as an untouched or
preregistered confirmatory test. During implementation verification, the
holdout was rerun after a runtime-motivated `LinearSVR` solver change. The
within-run chronology remains valid, so 2018 is useful as a final chronological
holdout and reproducibility demonstration. A future confirmatory study should
freeze code, features, solvers, and selection rules before evaluating new later
or local observations.

For a replacement dataset with at least three usable calendar years, the last
year is test, the preceding year is validation, and all earlier years train. If
those year partitions are too small, the ordered fallback is 70% training, 15%
validation, and 15% test. No partition is shuffled.

## 6. Candidate models

### Global median baseline

Every hour receives the median training count. This answers: "Can the proposed
method beat one typical number?" Median is less affected by extreme counts than
mean.

### Historical hour-of-week baseline

Each of the 168 positions in a week receives its training-period median. It is a
strong engineering reference because recurrent commuting demand often explains
much of hourly traffic variation. An unseen weekly hour falls back to the global
median.

### Ordinary Linear Regression

Linear Regression estimates a weighted additive relationship after encoding
the calendar categories. It is transparent and fast. It cannot naturally learn
every nonlinear threshold or interaction unless that relationship has been
specified as a feature.

### Linear Support Vector Regression

`LinearSVR` finds a linear relationship using the support-vector-regression
loss concept. Numeric inputs and the target are standardized inside the fitted
pipeline. It demonstrates an SVM-family approach without the high computation
of a full nonlinear kernel SVM on roughly 48,000 source rows.

### Random Forest

A Random Forest averages many decision trees. It can learn nonlinear effects
and interactions, such as a different temperature association in different
months. Its flexibility can improve predictions, but it is less directly
interpretable and may not extrapolate a time trend reliably.

### Optional small neural network

The optional MLP has two small hidden layers and early stopping. It is excluded
by default because tabular data of this scale do not automatically benefit from
a neural network. Run it only as an additional candidate:

```powershell
python 03_train_models.py --include-neural-network
```

It participates in the same validation comparison, but it is retained only if
its validation MAE is at least 2% below the best classical model's MAE. During
candidate comparison, the MLP's 15% internal early-stopping subset is sampled
only from pre-2017 training data; 2017 remains the external model-selection
period. If the MLP wins, its final refit uses all observations available through
2017 and performs early stopping inside that refit data. Complexity alone is not
evidence of quality.

Every candidate prediction is clipped at zero to enforce the physical lower
bound of an hourly count.

## 7. Metrics and selection rule

For observations \(y_i\), estimates \(p_i\), and \(n\) hours:

- **MAE** = average of \(|y_i-p_i|\). Units are vehicles/hour, and lower is
  better. It is the primary model-selection metric. The only added rule is that
  the optional MLP must improve on the best classical MAE by at least 2%.
- **RMSE** = square root of the average squared error. Units are vehicles/hour,
  and large failures receive additional weight.
- **R-squared** compares squared error with a constant-mean reference. One is a
  perfect fit; zero gives no improvement over that reference; negative is
  worse.
- **WAPE** = 100 times total absolute error divided by total observed volume.
  It summarizes error relative to the overall traffic scale.

MAPE is omitted as a headline result because dividing by an hour with zero or
very low volume makes a percentage unstable. WAPE should also not be read as a
guaranteed per-hour percentage.

The final chronological holdout is summarized overall and by hour, travel
period, weekend status, and month. `permutation_importance.csv` records the
increase in MAE when an input is disrupted in a sample of 2018 holdout hours. It
is generated only after within-run model selection and does not select the
model. A large importance means predictive usefulness within this dataset, not
causal impact.

### Verified reference run

The completed normal run used Python 3.12.10, the full 250-tree Random Forest,
and no optional MLP. Its generated outputs report:

| Partition | Selected model | MAE | RMSE | R-squared | WAPE |
|---|---|---:|---:|---:|---:|
| 2017 validation | Historical hour-of-week baseline | 255.6 vehicles/hour | 460.3 vehicles/hour | 0.946 | 7.6% |
| 2018 chronological holdout | Historical hour-of-week baseline | 252.2 vehicles/hour | 481.3 vehicles/hour | 0.941 | 7.6% |

Linear Regression achieved a lower 2017 RMSE (426.5 vehicles/hour) but a higher
MAE (288.9 vehicles/hour). The project therefore kept the hour-of-week baseline
under its predeclared MAE rule. The strongest tested calendar-and-weather
candidate was Linear Regression, and its validation MAE was 33.3 vehicles/hour
worse than the winner. That result rejects neither a physical weather effect nor
all possible weather models; it only describes the tested representations and
selection metric.

These rounded values are a reference snapshot. Generated outputs are
authoritative for the latest execution because options, software, or later code
can change the results.

## 8. Reproducibility

- A fixed random seed of 42 controls stochastic models.
- The verified project environment uses Python 3.12.10. A normal full run may
  need roughly 1-2 GB of available RAM.
- Split dates, selected model, options, package versions, and seed are saved in
  `outputs/reports/experiment_metadata.json`.
- The fitted preprocessing and model are saved together in
  `models/best_traffic_model.joblib`.
- Raw data, cleaned data, metrics, individual holdout predictions, and grouped
  errors provide an audit trail.
- The normal run uses 250 random-forest trees; `--fast` uses 60 as a quicker
  educational check and should be reported as a different experiment.

## 9. Interpretation boundaries

This is a predictive case study, not a causal traffic-demand model.

- The station represents one direction at one permanent count location, not the
  I-94 corridor, a city network, or another country.
- Weather, time, season, commuting, holidays, and unrecorded conditions are
  associated. Feature importance does not isolate a weather treatment effect.
- No incidents, construction, lane closures, special events, speeds, occupancy,
  upstream counts, population, or land-use variables are present.
- Source timestamps and missing-hour patterns may affect daily and seasonal
  summaries.
- Future climate, travel behavior, roadway capacity, and sensor calibration can
  differ from 2012-2018.
- The built-in 2019 one-hour scenario demonstrates saved-model use beyond the
  observed period; the 2018 holdout does not establish its 2019 accuracy. The
  current saved baseline ignores its supplied weather and holiday values, as
  recorded in the prediction output.
- A new location requires local training data and a new chronological test.
- If a future run selects a weather-using model, operational use would require
  reliable weather forecasts. Every deployed winner requires monitoring for
  drift.

## 10. Source, citation, and license

Official page: [Metro Interstate Traffic Volume, UCI dataset
492](https://archive.ics.uci.edu/dataset/492/metro%2Binterstate%2Btraffic%2Bvolume)

> Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5X60B

UCI publishes the dataset under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Sharing and adaptation are permitted when appropriate credit is given. The data
license and citation should accompany derived datasets, figures, reports, or
applications that redistribute or build on the source data.
