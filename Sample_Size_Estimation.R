# pmsampsize can be used to calculate the minimum sample size for the development of models with continuous, binary or survival (time-to-event) outcomes. Riley et al. lay out a series of criteria the sample size should meet. These aim to minimise the overfitting and to ensure precise estimation of key parameters in the prediction model.
# For continuous outcomes, there are four criteria:
#   i) small overfitting defined by an expected shrinkage of predictor effects by 10% or less, ii) small absolute difference of 0.05 in the model’s apparent and adjusted R-squared value, iii) precise estimation of the residual standard deviation, and
# iv) precise estimation of the average outcome value.
# The sample size calculation requires the user to pre-specify (e.g. based on previous evidence) the anticipated R-squared of the model, and the average outcome value and standard deviation of outcome values in the population of interest.
# For binary or survival (time-to-event) outcomes, there are three criteria:
#   i) small overfitting defined by an expected shrinkage of predictor effects by 10% or less,
# ii) small absolute difference of 0.05 in the model’s apparent and adjusted Nagelkerke’s R-squared value, and
# iii) precise estimation (within +/- 0.05) of the average outcome risk in the population for a key timepoint of interest for prediction.
# With thanks to Richard D. Riley, Emma C Martin, Gary Collins, Glen Martin & Kym Snell for helpful input & feedback
library(pmsampsize)
# we used 208 GM and 1794 WM features = 2002 max. features
#
pmsampsize(type = "c", parameters = 2002, rsquared = .7, intercept = 64.63, sd = 7.7)
# required N = 14826, assuming 30% shrinkage
pmsampsize(type = "c", parameters = 2002, rsquared = .6, intercept = 64.63, sd = 7.7)
# required N = 19522, assuming 40% shrinkage
pmsampsize(type = "c", parameters = 2002, rsquared = .5, intercept = 64.63, sd = 7.7)
# required N = 25848, assuming 50% shrinkage
pmsampsize(type = "c", parameters = 2002, rsquared = .4, intercept = 64.63, sd = 7.7)
# required N = 35117, assuming 60% shrinkage

# With other words, even at an overestimated 60% shrinkage, 
# we still have enough participants (N = 36,930) to build a valid prediction model

