# Time-point isolated brain age predictions

## I set the working dir to the place where the data are stored.
setwd("/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/data/unscaled")
## The save_path is where the output/results (figures and tables) are stored
save_path = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/unscaled/"
#
# load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lme4, nlme, ggplot2, tidyverse, lm.beta, remotes, ggpubr, 
               grid, lmtest, car, lmtest,lmeInfo,lmerTest,sjstats,effsize,Rmpfr,
               ggrepel,PASWR2, reshape2, xgboost, confintr, factoextra, mgcv, 
               itsadug, Metrics, ggpointdensity, viridis, MuMIn,hrbrthemes,
               ggridges, egg, pheatmap, ggtext, RColorBrewer,Bioconductor,caret,
               glmnet,marginaleffects, update = F)
# load data
T1w_T1 = read.csv("T1w_test1.csv") 
T1w_T2 = read.csv("T1w_test2.csv") 
# make list of all data
data = list(T1w_T1, T1w_T2)
# train-test sampling
sampling = T1w_T1[sample(ncol(multi_T1)/2, replace = F),]$eid # random drawing of 50% of eids
train = list()
test = list()
for (i in 1:length(data)){
  train[[i]] = data[[i]] %>% filter(eid %in% sampling)
  test[[i]] = data[[i]]  %>% filter(!eid %in% sampling)
}
# train & predict (linear models)
models = list()
predictions = list()
for (i in 1:length(data)){
  train_df = train[[i]] %>% select(-c(eid,sex,site))
  test_df = test[[i]] %>% select(-c(eid,sex,site))
  models[[i]] =  lm(age~.,data = train_df)
  predictions[[i]] = predict(models[[i]], test_df)
}
# descriptive stats
for (i in 1:length(data)){
  print(mean(predictions[[4]] - predictions[[3]]))
}

# check the mean differences between ...
## age predictions
mean(predictions[[2]] - predictions[[1]])
## age
mean(test[[2]]$age - test[[1]]$age)
## brain age gap
mean(predictions[[1]] - test[[1]]$age)
mean(predictions[[2]] - test[[2]]$age)
## difference in brain age gaps assessed by t-test
pval = t.test(predictions[[1]] - test[[1]]$age, predictions[[2]] - test[[2]]$age, paired = T)$p.value
## same test (repeated t-test), but this time controlling for sex, site, and age
t1 = test[[1]] %>% select(eid,sex,site, age)
t1$pred = predictions[[1]]
t2 = test[[2]] %>% select(eid,sex,site, age)
t2$pred = predictions[[2]]
df = rbind(t1,t2)
df$TP = c(replicate(nrow(df)/2, 0),replicate(nrow(df)/2, 1))
df$BAG = df$pred - df$age
mod1 = lmer(pred ~ TP + age*sex + site + (1|eid), data = df)
lmerpval = spredlmerpval = summary(mod1)$coefficients[2,5]
beta = summary(mod1)$coefficients[2]
predictions(mod1, by = "TP") # print marginal effects
summary(mod1)
#
#
# corrected BAG formula
corrected = function(training_frame, predicted_age_train, predicted_age_test, age_test){
  corlm = lm(predicted_age_train~age,data = training_frame)
  intercept = summary(corlm)$coefficients[1]
  slope = summary(corlm)$coefficients[2]
  predicted_age_test + (age_test - (slope*predicted_age_test+intercept))
}
res.tab = data.frame(Variable = c("predicted age difference","corrected predicted age difference", "age difference", "brain age gap time point 1", "brain age gap time point 2","corrected brain age gap time point 1", "corrected brain age gap time point 2", "t-test pval", 
                        "LMER pval", "LMER beta", "Marginal means"),
           Value = c(mean(predictions[[2]] - predictions[[1]]), 
                     mean(corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age)-corrected(train[[2]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age)),
                     mean(test[[2]]$age - test[[1]]$age), 
                     mean(predictions[[1]] - test[[1]]$age), 
                     mean(predictions[[2]] - test[[2]]$age), 
                     mean(corrected(train[[1]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age) - test[[1]]$age), 
                     mean(corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age) - test[[2]]$age), 
                     pval,
                     lmerpval, 
                     beta, 
                     "Decreasing, non-sign."))
write.csv(res.tab, paste(save_path,"TP_isolated_BAG.csv",sep=""), row.names = T)
#
#
# one can also attempt to predict from TP 1 to TP 2
res.tab2 = data.frame(Prediction = c("TP1 to TP1", "TP2 to TP2", "TP1 to TP2", "TP2 to TP1"),
           PredictedAge = c(mean(predictions[[1]]),mean(predictions[[2]]), mean(predict(models[[1]], test[[2]] %>% select(-c(eid,sex,site)))), mean(predict(models[[2]], test[[1]] %>% select(-c(eid,sex,site))))),
           ChronologicalAge = c(mean(T1w_T1$age), mean(T1w_T2$age), mean(T1w_T1$age), mean(T1w_T2$age)),
           BAG = c(mean(predictions[[1]] - test[[1]]$age), 
                                mean(predictions[[2]] - test[[2]]$age),
                                mean(predict(models[[1]], test[[2]] %>% select(-c(eid,sex,site))) - test[[1]]$age), 
                                mean(predict(models[[2]], test[[1]] %>% select(-c(eid,sex,site))) - test[[2]]$age)))
write.csv(res.tab2, paste(save_path,"TP_isolated_BAG_cross_model_pred.csv",sep=""), row.names = T)
# just as the previous table, but for corrected BAG
res.tab3 = data.frame(Prediction = c("TP1 to TP1", "TP2 to TP2", "TP1 to TP2", "TP2 to TP1"),
                      CorrectedPredictedAge = c(mean(corrected(train[[1]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age)), 
                                                mean(corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age)),
                                                mean(corrected(train[[2]], predict(models[[1]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age)),
                                                mean(corrected(train[[1]], predict(models[[2]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age))),
                      ChronologicalAge = c(mean(T1w_T1$age), mean(T1w_T2$age), mean(T1w_T1$age), mean(T1w_T2$age)),
                      CorrectedBAG = c(mean(corrected(train[[1]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age) - test[[1]]$age), 
                              mean(corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age) - test[[2]]$age),
                              mean(corrected(train[[2]], predict(models[[1]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age) - test[[2]]$age),
                              mean(corrected(train[[1]], predict(models[[2]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age) - test[[1]]$age)))
write.csv(res.tab3, paste(save_path,"TP_isolated_CORRECTED_BAG_cross_model_pred.csv",sep=""), row.names = T)
#
#
#
#
# estimate rate of change in brain age
AROC = data.frame(eid = test[[1]]$eid, sex = test[[1]]$sex, site = test[[1]]$site, 
                  AROCc = ((corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age)-(test[[2]]$age) - corrected(train[[1]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age))-test[[1]]$age)/(test[[2]]$age -test[[1]]$age),
                  AROCu = ((predictions[[2]]-test[[2]]$age) - (predictions[[1]]-test[[1]]$age))/(test[[2]]$age - test[[1]]$age),
                  CCc = (corrected(train[[2]], predict(models[[2]], train[[2]] %>% select(-c(eid,sex,site,age))), predictions[[2]], test[[2]]$age) - corrected(train[[1]], predict(models[[1]], train[[1]] %>% select(-c(eid,sex,site,age))), predictions[[1]], test[[1]]$age))/2,
                  CCu = (predictions[[2]] - predictions[[1]])/2) 
###### PCA
# first check which variables are significantly changing over time
df = rbind(test[[1]],test[[2]])
df$TP = c(replicate(nrow(df)/2, 0),replicate(nrow(df)/2, 1))
t.val = c()
outcome_vars = names(train[[2]] %>% select(-c(eid,sex,site,age)))
for (i in outcome_vars){
  f = formula(paste(i,"~TP + (1|eid)", sep = ""))
  # some of the p values will be rounded to 0. We need to use a more accurate approach (see below)
  t.val[i] = summary(lmer(f, data = df))$coefficients[2,4]
}
p.val = .N(2 * pnorm(-abs(t.val)))
adj.p.val = p.adjust(p.val)
keep = data.frame(outcome_vars, adj.p.val) %>% filter(adj.p.val < .05)
FeatureAROC = (subset(test[[2]], select = names(test[[2]]) %in% keep$outcome_vars) - (subset(test[[1]], select = names(test[[2]]) %in% keep$outcome_vars)))/(test[[2]]$age-test[[1]]$age)
FeatureCC = ((subset(test[[2]], select = names(test[[2]]) %in% keep$outcome_vars) - (subset(test[[1]], select = names(test[[2]]) %in% keep$outcome_vars))))/2
PCAROC = prcomp(FeatureAROC, scale. = T, center = T)
PCCC = prcomp(FeatureCC, scale. = T, center = T)
a = fviz_eig(PCAROC, main = "Rate of Change", addlabels=TRUE, hjust = -0.3,
         linecolor ="red") + theme_minimal() + ylim(0,30)
b = fviz_eig(PCAROC, main = "Centercept", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,30)
plot = ggarrange(a,b,nrow = 1)
ggsave(file = paste(save_path,"Scree_Plots_ind_TP.pdf",sep=""), plot, width = 16, height = 7)
rm(a,b, plot)
AROC$AROC_PCA = as.numeric(PCAROC$x[,1])
AROC$CC_PCA = as.numeric(PCCC$x[,1])
AROC$ISI = test[[2]]$age - test[[1]]$age
AROC$age = test[[1]]$age
#
###### hypothesis tests
# HYPOTHESIS 1a: longitudinal and cross sectional measures show a relationship
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," + ISI + age*sex + site", sep = ""))
  H1a01 = lm(f, data = BAG2)
  Beta = (summary(H1a01)$coefficients[2])
  Beta_standardized = summary(lm.beta(H1a01))$coefficients[2,2]
  SE = (summary(H1a01)$coefficients[2,2])
  t = summary(H1a01)$coefficients[2,3]
  p = summary(H1a01)$coefficients[2,4]
  return(data.frame(Beta,Beta_standardized, SE, t, p))
} 
# the function returns the beta coefficient and SE in years
# we need to scale the data first
BAG2 = AROC
BAG2[4:ncol(BAG2)] = data.frame(scale(BAG2[4:ncol(BAG2)]))
# make a list of all the combinations to look at
BAG_pairs = data.frame(x = c("CCc", "CCu"),
                       y = c("AROCc", "AROCu"))
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 5))
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])
}
names(res1) = c("Beta","Std.Beta", "SE", "t", "p")
rownames(res1) = c("corrected BAG", "uncorrected BAG")
# now, we look at the association of longitudinal and centercept PCs
PC_pairs = data.frame(x = c("CC_PCA"),
                      y = c("AROC_PCA"))
res2 = data.frame(matrix(nrow = nrow(PC_pairs), ncol = 5))
for (i in 1:nrow(PC_pairs)){
  res2[i,] = H_test(PC_pairs$x[i], PC_pairs$y[i])
}
names(res2) = names(res1)
rownames(res2) = c("PC")
res = rbind(res1, res2)
write.csv(x=res, paste(save_path,"H1a_long_cross_associations_no_interaction_TPindi.csv",sep=""))
# HYPOTHESIS 1b: longitudinal and cross sectional measures show a relationship, controlling ISI
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," * ISI + age*sex + site", sep = ""))
  H1a01 = lm(f, data = BAG2)
  Beta = (summary(H1a01)$coefficients[2])
  Beta_standardized = summary(lm.beta(H1a01))$coefficients[2,2]
  SE = (summary(H1a01)$coefficients[2,2])
  t = summary(H1a01)$coefficients[2,3]
  p = summary(H1a01)$coefficients[2,4]
  return(data.frame(Beta,Beta_standardized, SE, t, p))
}
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 5))
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])
}
names(res1) = c("Beta","Std.Beta", "SE", "t", "p")
rownames(res1) = c("corrected BAG", "uncorrected BAG")
# now, we look at the association of longitudinal and centercept PCs
PC_pairs = data.frame(x = c("CC_PCA"),
                      y = c("AROC_PCA"))
res2 = data.frame(matrix(nrow = nrow(PC_pairs), ncol = 5))
for (i in 1:nrow(PC_pairs)){
  res2[i,] = H_test(PC_pairs$x[i], PC_pairs$y[i])
}
names(res2) = names(res1)
rownames(res2) = c("PC")
res = rbind(res1, res2)
write.csv(x=res, paste(save_path,"H1a_long_cross_associations_interaction_TPindi.csv",sep=""))
#
######## H2
BAG2$age = as.numeric(BAG2$age)
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," + ISI + ti(",x,",ISI, k = 4) + age:sex + age + sex + site", sep = ""))
  H1a01 = gam(f, data = BAG2)
  Beta = (summary(H1a01)$p.coeff[2])
  SE = (summary(H1a01)$se[2])
  t = summary(H1a01)$p.t[2]
  p = summary(H1a01)$p.pv[2]
  F_smooth = summary(H1a01)$s.table[3]
  p_smooth = summary(H1a01)$s.table[4]
  return(list(data.frame(Beta, SE, t, p, F_smooth, p_smooth),H1a01))
}
BAG_pairs = data.frame(x = c("CCc", "CCu", "CC_PCA"),
                       y = c("AROCc", "AROCu", "AROC_PCA"))
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 6))
plot.mod = list()
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[1]]
  plot.mod[[i]] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[2]]
}
names(res1) = c("Std.Beta", "SE", "t", "p", "F_smooth.Interact", "p.Interact")
rownames(res1) = paste(BAG_pairs$y, BAG_pairs$x, sep = "_")
write.csv(res1, paste(save_path,"H1b_ISI_BAG_smooth_interactions_TPindi.csv",sep=""))

BAG_pairs = data.frame(x = c("AROC_PCA", "AROC_PCA"),
                       y = c("CCu", "CCc"))
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 6))
plot.mod = list()
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[1]]
  plot.mod[[i]] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[2]]
}
names(res1) = c("Std.Beta", "SE", "t", "p", "F_smooth.Interact", "p.Interact")
rownames(res1) = paste(BAG_pairs$y, BAG_pairs$x, sep = "_")
write.csv(res1, paste(save_path,"long_cross_associations_TPindi.csv",sep=""))
print("The end.")
