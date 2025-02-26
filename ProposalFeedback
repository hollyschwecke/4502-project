# Project Proposal Feedback and Solutions

## 1. Geographical Data Integration

### Feedback:
More detail needed on matching coordinates between datasets with different spatial resolutions.

### Investigation:
NOAA weather station data is point-based, while wildfire data may cover polygonal areas or be represented as points. This spatial mismatch will require careful integration.

### Solution:
We will implement **Buffer Analysis** as our primary approach. For each wildfire point, we will create radius buffers (e.g., 10km, 25km, 50km) and extract weather data from stations within these buffers. 
We will include a diagram illustrating this methodology and validate by testing different radius parameters to find optimal spatial relationships.

## 2. Cross-Validation with Time-Series Data

### Feedback:
Need more information on approach to temporal validation for time-dependent wildfire and weather data.

### Investigation:
With some further research, its evident that standard k-fold cross-validation can lead to data leakage with time series, as future information might be used to predict past events.

### Solution:
We will implement:
- **Forward Chaining**: Training on years 2000-2010, validate on 2011; train on 2000-2011, validate on 2012, etc.
- **Seasonal Considerations**: Accounting for yearly fire seasons with complete seasonal coverage

We could include a diagram showing our temporal cross-validation strategy that preserves time-dependency while maximizing training data usage.

## 3. Specify Model Types and Explain Why Appropriate

### Feedback:
Need to specify model types and explain why they're appropriate for this project.

### Investigation:
We evaluated various model types for their suitability to wildfire prediction based on weather data.

### Solution:
We will employ:
- **Gradient Boosting Models**: (XGBoost) For capturing non-linear relationships between weather patterns and fire risk
- **Random Forests**: For interpretability through feature importance, common for weather data
- **LSTM Networks**: Capturing long-term temporal dependencies
- **Ensemble Approach**: Combining models to improve robustness for rare wildfire events

This ensemble approach handles imbalanced data, provides feature importance rankings, and captures complex temporal patterns.

## Additional Temporal Resolution Considerations

### Investigation:
We recognized that weather conditions preceding a wildfire may be more important than conditions on the exact day of detection.

### Solution:
We could enhance our approach with:
- Lagged features (1-day to 30-day averages/extremes)
- Cumulative metrics (precipitation deficits, heat indices)
- Trend indicators capturing weather patterns
- Seasonal context features
