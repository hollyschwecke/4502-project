# Weather Patterns and Wildfire Analysis 
### CSPB4502 Data Mining Project
### The 8-Team: Holly Schwecke, Haoye Tong & Matthew Presti 


## Description
This data mining project analyzes the relationship between weather patterns and wildfire activity in California from 2011-2020. Using NOAA weather station data and NASA MODIS satellite wildfire detections, we developed a custom spatial integration methodology and applied both predictive modeling (XGBoost) and unsupervised pattern discovery (UMAP and HDBSCAN clustering) to identify weather patterns associated with wildfire events.

## Research Questions
1. Can we identify causative factors in years of elevated wildfire activity based on weather data trends?
2. Can we predict periods of extreme winds that pose elevated fire risks?
3. What trends could be valuable for wildfire prediction or forecasting fire season severity?


## Findings
1. Higher temperatures and longer periods without rain show slight positive relationships with fire intensity, suggesting they contribute modestly to elevated wildfire activity. Precipitation and dewpoint did not show strong links, indicating they are less reliable predictors on their own.
2. Wind speed alone does not atrongly correlate with fire intensity, meaning extreme winds are not a strong standalone predictor of elevated fire risk without other contributing factors. Instead, a combination of extended drought periods, elevated temperatures, and low humidity emerged as the strongest predictors.
3. High temperatures and extended dry periods appear most useful for forecasting severe fire seasons. Dewpoint, precipitation, and wind speed were less predictive individually but could add value when combined with other factors.

## Application
Understanding the relationship between weather variables and fire intensity can greatly improve wildfire prediction, management, and prevention strategies. By recognizing that higher temperatures and longer dry periods slightly increase fire intensity, agencies can enhance early warning systems, better allocate firefighting resources, and plan public safety campaigns more effectively. This knowledge also supports smarter urban planning, guiding the development of fire-resistant infrastructure in high-risk areas. In agriculture and forestry, it can inform practices like controlled burns and vegetation management to reduce fuel loads. Additionally, as climate change intensifies heat and drought conditions, these insights are critical for modeling future wildfire risks and shaping environmental policies. Overall, focusing on temperature and dryness, rather than wind alone, allows for more targeted and proactive wildfire preparedness. 

## Video Presentation
https://www.youtube.com/watch?v=tzcaMViA1gc

## Final Paper
Group8_WildfirePrediction_Part4.pdf
---
*Note: This project is part of CSPB-4502 Data Mining course.*
