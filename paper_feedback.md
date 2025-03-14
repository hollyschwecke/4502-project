# Project Proposal Feedback and Solutions

## Overview of Feedback Items

1. **Spatial-Temporal Data Integration**: "Providing more detail on how you will handle spatial-temporal data integration. Specifically, when creating circular zones around wildfire sites to collect weather data from nearby stations, how will you handle situations where multiple weather stations fall within the same zone? Will you take an average of the readings, or will you use more advanced spatial interpolation techniques?"

2. **Local vs. Regional Weather Patterns**: "Specify how you plan to assess the impact of local versus regional weather patterns on wildfire risk. Since extreme weather events (like heatwaves) may have different regional effects."

3. **Granular Temporal Factors**: "Your feature engineering process could consider including more granular temporal factors, such as daily, weekly, and seasonal trends."

4. **Model Evaluation Metrics**: "Clarify how you'll compare the performance of the different models you plan to use, beyond using AUC and RMSE. Are you also considering precision, recall, or F1 score, particularly for predicting rare but high-impact events like large wildfires?"

5. **Timeline and Risk Mitigation**: "The proposed timeline is well-structured, but you may want to highlight any potential challenges related to data preprocessing or model tuning, and how you plan to mitigate these issues."

## Item #1: Spatial-Temporal Data Integration

**Feedback:**
More detail needed on handling situations where multiple weather stations fall within the same circular zone around wildfire sites.

**Investigation:**
When creating buffer zones around wildfire locations, it's common to encounter multiple weather stations within a single zone. This creates a methodological challenge in determining how to aggregate or prioritize these multiple data sources.

**Solution:**
We will implement **Weighted Distance Averaging** as our primary approach:
* For each wildfire point, weather stations within the defined buffer zone will be weighted by their inverse distance to the fire location
* Closer stations will have more influence on the aggregated weather metrics
* We will test both linear and quadratic distance weighting functions
* We will include a mathematical formulation and visual representation of this methodology in our updated proposal
* For validation, we'll conduct sensitivity analysis using different weighting schemes and buffer distances

## Item #2: Local vs. Regional Weather Pattern Assessment

**Feedback:**
Need to specify how we'll assess the impact of local versus regional weather patterns on wildfire risk.

**Investigation:**
Our current approach does not clearly distinguish between local weather effects (immediately around fire locations) and broader regional patterns that might influence fire behavior differently.

**Solution:**
We will implement a **Two-Tier Weather Feature Framework**:

* **Local Weather Features (0-10km radius)**:
  - Direct measurements from the closest weather station to each wildfire point
  - Variables: daily max/min temperature, relative humidity, wind speed/direction, precipitation
  - Implementation: Each wildfire point will be paired with data from its single closest weather station

* **Regional Weather Features (10-50km radius)**:
  - Aggregated data from all weather stations within the broader region
  - Variables: regional temperature anomalies, drought indices (PDSI), atmospheric pressure patterns
  - Implementation: Simple averaging of values across all stations in the region

* **Comparative Analysis Method**:
  1. Create two separate feature sets (local-only, regional-only)
  2. Train identical model types on each feature set independently
  3. Calculate feature importance scores to identify which variables at each scale are most predictive
  4. Compare prediction accuracy between local-only and regional-only models
  5. Possibly develop a combined model and quantify the improvement over single-scale models

This approach would allow us to directly measure and report the relative contribution of local versus regional weather patterns to wildfire prediction accuracy.

## Item #3: Granular Temporal Factors in Feature Engineering

**Feedback:**
Consider including more granular temporal factors, such as daily, weekly, and seasonal trends in the feature engineering process.

**Investigation:**
Temporal patterns at different scales may have distinct influences on wildfire behavior, and extreme events like heatwaves require special consideration.

**Solution:**
We could enhance our feature engineering with **Hierarchical Temporal Features**:
* **Daily Factors**: Diurnal temperature range, daily precipitation events, wind pattern shifts
* **Weekly Trends**: 7-day rolling averages and variances, week-over-week changes
* **Seasonal Components**: Seasonal dummies, days since wet season, seasonal anomalies
* **Extreme Event Windows**: Custom features to capture heatwave duration, consecutive dry days
* Implementation of lag features at multiple time scales (3, 7, 14, 30, 60, 90 days)
* Incorporation of cumulative metrics (e.g., growing season precipitation deficit)

## Item #4: Model Evaluation Metrics Beyond AUC and RMSE

**Feedback:**
Clarify how you'll compare the performance of different models beyond AUC and RMSE, particularly for rare but high-impact events like large wildfires.

**Investigation:**
Standard metrics may not adequately capture model performance for imbalanced datasets where wildfire events are rare but consequential.

**Solution:**
We will implement a **Comprehensive Evaluation Framework**:
* **Class-Specific Metrics**: Precision, recall, and F1 scores calculated specifically for high-risk wildfire events
* **Threshold Optimization**: Finding optimal decision thresholds that balance false positives and false negatives
* **Cost-Sensitive Evaluation**: Implementing custom loss functions that penalize missed wildfire predictions more heavily
* **Temporal Performance**: Evaluating model stability across different seasons and years
* **Spatial Performance**: Assessing model accuracy across different geographic regions and terrain types
* **Visualization Suite**: ROC curves, Precision-Recall curves, and confusion matrices for intuitive performance comparison

## Item #5: Timeline Challenges and Risk Mitigation

**Feedback:**
Highlight potential challenges related to data preprocessing or model tuning, and how you plan to mitigate these issues.

**Investigation:**
Our project involves complex spatial-temporal data that poses several technical challenges.

**Solution:**
We will implement a **Targeted Risk Management Plan**:
* **Data Challenges**:
  - Missing weather data &rarr; Interpolation with uncertainty measures
  - Satellite detection errors &rarr; Validation with ground-truth records
* **Computational Concerns**:
  - Large dataset processing &rarr; Parallel computing techniques
  - Extended model training &rarr; Progressive complexity approach
* **Timeline Strategy**:
  - Two-week buffer periods for each project phase
  - Weekly checkpoints with prioritized feature development
