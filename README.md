# ShippingEmissions
Improving emissions estimates from reported values and ML

Contents
========
1. [Todo](#todo)
2. [Calculations](#calculations)
3. [ML Training](#ml-training)
4. [Extrapolation (Prediction)](#extrapolation-prediction)
5. [Robustness](#robustness-checks)
6. [Analyses](#analyses)
7. [Extensions](#extensions)
8. [Meetings](#meetings)


Todo
====
1. check emissions calc code to figure out offset and create required plots
2. compare with IMO4 procedure
3. Write skeleton of data part
4. check ML code

[contents](#contents)

Calculations
============
# WFR Tech Specs for fixed power component
    - first drop of all cols with NA
    - compare speed we are using to imo4
        - This study assumes that the service speed reported in the IHS database corresponds to a power output of 100% of the main engine’s MCR for all vessel types, with the exception of the two largest container sizes and cruise ships. In contrast, the Third IMO GHG Study 2014 assumed that the reported values corresponded to 90% MCR. As a result, a factor of 0.9 was applied to the Admiralty equation when estimating the main engine power. In theory this means that the estimated main engine load in this report is around 10% higher than that of the Third IMO GHG Study 2014, however the speed reported in IHS dataset used in this study contains either service speed or max speed corresponding with 100% MCR making the load factors comparison rather difficult. Further explanation on the reasoning for changing the MCR correction factor method is addressed in Sections 2.2.5 and 2.7.1.
        -  Me from earlier: "Service speed is likely at 80% operating power (most efficient for engine) which is likely equivalent to 92% of max speed"

    - imputing RPM 
    - imputing a few missing engine powers? where do we use these?
    - where is Engine_Category used?
    - impute missing design draughts?
        - IMO Section 2.2.1: infilled with the vessel type and size median design draught.

    - What used for load? speed or power?
        - Should be power:
        https://www.marineinsight.com/main-engine/how-to-use-main-engine-performance-curve-for-economical-fuel-consumption-on-ships/


# AIS pre-processing
AIS_Calculations.py
    - We don't join first, so could miss vessels that change mmsi mid-year? p53
    - confirm no lats or longs outside of -90/90 and -180/180
    -[waiting on response from Oliver] Check why new version of ais_bulkers_calcs has a few fewer observations for each ship-year than the archived version I have
    - Check and report comparable values (p54):
        - A vessel is not extrapolated into a full year when a) there are less than 10 AIS observations detected, b) the number of AIS observations with an SOG greater than 3 knots are less than 20, and c) when the entire set of SOG and GPS observations are missing or incorrect. These filtered vessels were most likely inactive during the year or had their AIS receivers switched off. By applying these filters, approximately 8-9% per year of the originally matched Type 1 and Type 2 vessels were excluded.
    - haversine for missing data may cause to pass through land

    - TODO: try to use IMO number as per IMO4 as opposed to dropping every second path

AIS_Calculations_Interp.py
    -x fix draught infill - 
    - we are using instantaneous speed, unless the point is interpolated
        - compare to p55/56 complex procedure involving phase assignment
    - compare p52 onward for pre-processing
    - create simplified table 13
    -x fixed sjoin and speed for manoeuvring
    - document creation of buffered_coastline:
        - I used natural_earth_vector.gpkg 10m land
        - split using ogr2ogr -wrapdateline (in terminal) to avoid weird distortion at dateline
        - buffer 0.83333 degrees (equivalent to 5nm at equator)
        - probably need code for buffered_reprojected_coastline for publishing

    -* Compare to IMO4 AIS data quality (missing observations, distance travelled, total emissions)
        -***** How are large distances getting through??? Related to separate paths? NO *** Come back to this!!
    -* Check previous filling of draught!
    -* check for ships with no draught information (IMO imputes based on aggregated mean of ship type and size, p56)
    

# Check merge.py after updates
# Confirm with maps that working
# run on all


## Trip Detection
AIS_Calculations_PotentialPortcalls.py
    - using instantaneous speed for observed, implied for interpolated rows

AIS_Calculations_PP_Coast_EU.py



- Compare to IMO4 figure 109a both speed and proportion of year on EU routes

# Aggregation
AIS_Calculations_EU.py
    - use relative path to buffered_reprojected_ ...
    


EU_yearly_agg.py
    - do some spot checks on hourly power calculation
    - check missings and validity for power calcs


    - rename
    - which years have we been using?
    - update with brexit indicators
    - check upon merging for (p54):
        - SOG greater than 1.5 times the design speed are replaced with an interpolated speed by applying the AIS SOG infilling methodology described below.
        - draughts greater than the design draught are replaced with the design draught values.
        - impute draught with all missing can get list from stats explore file
        - trip level draught could then be imputed as median of trip values (p56)
        - drop if less than 10 observations detected (as per IMO4)
    - we make no distinction for emission control zones (p58) because LSFO has same carbon content
    - similarly, no low-load adjusment because factor is 1 for CO2
    - second version with load as speed ratio rather than power
    - check fuel type assignment for emission factor (p74)


- How aggregating?
- Using instantaneous or implied speed?
    - Are speeds above service speed getting truncated?
- check how using draught in calc

# AIS-WFR matching
    - reproduce Table 13 with matching stats

# Compare engineering emissions estimate to IMO
    - get number on our calculated emissions
    - compare our estimated cargo (tonnes) to MRV reported

# Explore Residuals (and bias)
    - figure 104 both emissions and distance distribution
    - IMO should have plotted distribution of discrepancies, not comparisons of distributions
    - try using different definition of reference speed
    - try using different definition of load (speed vs. power)
    - try checking how many vessels change mmsi mid-year or mid-sample

[contents](#contents)

ML Training
===========
- Final.jpynb
- df_ml.csv

- filter outliers (with +/-3stds of mean)
- are we not applying scaler? (Jasper said he recalls it performed worse with)
- what's ME_W_ref_first?
- which one is best? using feature selection? which features?
- Stochastic GB?

# Imputing missing tech specs (WFR)
    - typically using median imputation for missing observations right now, (categorical is mode)
            - all vars have better than 90\% non-missing?

# Checks
    - which years? train on 2019,2020, test on 2021 (without extrapolation, just for EU reporting ships)
    - check sample balance, scaling
    - check hyperparameter tuning

# Variable selection
    - compare included RHS variables to Yan et al (2023) table 2
    - include monitoring method from MRV (A,B,C) as RHS variable
    - include number of interpolated observations, number of filled draught values, ...
    - do we have fuel type?

# Cross-validation

# Performance Statistics
    - R2, RMSE, MAE, MAPE

# OECD Replication

[contents](#contents)

Extrapolation (Prediction)
==========================


[contents](#contents)

Robustness Checks
=================
# Subsetting to ships with most trips in the EU
# Clustering for draught
# for trip detection algorithm: vary threshold of distance discrepancy for retention (other?)
# Try 3kts rather than 1kt for trip detection in-port speed threshold


[contents](#contents)

Analyses
========
# Methodology
## engineering only and ML only model
## comparison to fuel efficiency prediction from OECD
    - emphasize improvement over random forest
## compare variable importance to Jebsen et al (2020) figure 6.6

## discuss feature importance as an insight to how we are improving upon engineering model

# Economics
## country pair emissions reports (relies on trip detection) and variation over COVID
## rank top routes
## then time series - can compare to OECD (we have speed) (decomposition speed vs volume using speed change counterfactual - which countries are important, EU ETS)





Extensions
==========
- weather?


Meetings
========

# Hiro 04Mar2024
1. are we discarding linear then tree approach?
    - no real gain, so not much point to present
2. weighting observations based on residual for ml step, as per adaboost?
    - proably already being by boosting algorithm internally
3. universal approximator? 'fixing' speed exponent?
    - maybe if very deep
    - Wagner causal forest, in the limit something like 
    - not splines because many variables but not too many obs
4. thoughts on claiming it mitigates data errors?
5. can we get a sense of model improvement vs measurement error by changing prediction features?
    - prediction for ships with missing covariates
    - e.g. number of missing obs probably doesn't help for not missing
    - treatment of NAs is maybe package specific
- 2 ways to extrapolate to missing variables
    - imputation
    - using NAs in ml
- mostly push on prediction for missing covariates
- 2 types of missing data
    - tracking points (don't want to call this missing data)
    - WFR



[contents](#contents)