# Annual change in white matter brain age, physiology, and associated polygenic disorder risk
# July 2024, Bergen, Norway
# Max Korbmacher (max.korbmacher@gmail.com)
# Functional R Version: 4.2.0
#
############################################################################## #
##### OVERVIEW ############################################################### #
############################################################################## #
#
# 0) Preparations & demographics
#
# 1) Brain age estimation
# 1.1) Use trained model to predict from time point 1 and 2 data
# 1.2) Estimate age-correlations in training and test data (3 scatter plots)
# 1.2.1) Uncorrected age predictions
# 1.2.2) Corrected age predictions
# 1.3) Assess model metrics
# 1.4) Estimate annual rate of brain age change
#
# 2. Relationship between cross-sectional brain ages
#
# 3. Relationship between brain age deltas and change in brain features
# 3.1 select significant features from paired samples t-test of brain features
# 3.2 compute annual rate of change in brain features & estimate principal components
# 3.3 TEST HYPOTHESES
# 3.3.1 H1: Associations between cross-sectional and longitudinal measures
# 3.3.2 H2: Interaction effects between inter-scan interval and changes in WMBAG on WMBAG, AND between the ISI and the WM PC on the PC of change
# 3.3.3 H3: Phenotype and genotype associations with ...
#           - annual change in brain age delta
#           - brain age delta (time point 1 and 2)
#           - principal components of white matter
# 4. Supplement
#
############################################################################## #
############################################################################## #
#
# 0) Preparations ####
#
# set working and saving directories
## I set the working dir to the place where the data are stored.
## Change this location!
setwd("/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/data/unscaled")
## The save_path is where the output/results (figures and tables) are stored
## Change this location!
save_path = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/unscaled/"
## previous predictions from different models will also have a location (not all code is contained in this single file)
# These predictions are named uniformely and placed in individual folders (per algorithm)
prev_preds = "/Users/max/Library/CloudStorage/OneDrive-HøgskulenpåVestlandet/Documents/Projects/LongBAG/results/prediction/"
#
# load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lme4, nlme, ggplot2, tidyverse, lm.beta, remotes, ggpubr, 
               grid, lmtest, car, lmtest,lmeInfo,lmerTest,sjstats,effsize,Rmpfr,
               ggrepel,PASWR2, reshape2, xgboost, confintr, factoextra, mgcv, 
               itsadug, Metrics, ggpointdensity, viridis, MuMIn,hrbrthemes,
               ggridges, egg, pheatmap, ggtext, RColorBrewer,Bioconductor,caret,
               glmnet,update = F)
#install.packages("tidyverse")
# note: until recent issues between Matrix v1.6.5 and lme4 v1.1.35 are not fixed,
# one can use the older Matrix version: v1.6.4, or install lme4 from source:
#install.packages("lme4", type = "source")
#library("lme4")
pacman::p_version(lme4)
#
# load data (true features)
## genotypes and phenotypes
PGRS = read.csv("../PRS.csv")
pheno = read.csv("../health_lifestyle_long.csv")
## cross sectional/training data
dMRI = read.csv("dMRI_train.csv")
T1w = read.csv("T1w_train.csv")
multi = read.csv("multi_train.csv")
## longitudinal/test data
dMRI_T1 = read.csv("dMRI_test1.csv")
dMRI_T2 = read.csv("dMRI_test2.csv")
T1w_T1 = read.csv("T1w_test1.csv") 
T1w_T2 = read.csv("T1w_test2.csv") 
multi_T1 = read.csv("multi_test1.csv") 
multi_T2 = read.csv("multi_test2.csv")
#
# Previous predictions
Lasso_train_preds = read.csv(paste(prev_preds,"Lasso_training_predictions.csv",sep=""))
Lasso_T1_preds = read.csv(paste(prev_preds,"Lasso_test1_predictions.csv",sep=""))
Lasso_T2_preds = read.csv(paste(prev_preds,"Lasso_test2_predictions.csv",sep=""))
XGB_train_preds = read.csv(paste(prev_preds,"XGB_training_predictions.csv",sep=""))
XGB_T1_preds = read.csv(paste(prev_preds,"XGB_test1_predictions.csv",sep=""))
XGB_T2_preds = read.csv(paste(prev_preds,"XGB_test2_predictions.csv",sep=""))
#
#
#
# merge data frames (only demo vars will be used)
df = rbind(dMRI_T1, dMRI_T2)
df$TP = c(replicate(nrow(dMRI_T1), 0), replicate(nrow(dMRI_T1), 1))
#
# copy demographics for descriptives and make sure eids are correct for correct merging
cross = dMRI %>% dplyr::select(eid, age, sex, site)
cross2 = T1w %>% dplyr::select(eid)
cross3 = multi %>% dplyr::select(eid)
# remove demographics from dfs
#dMRI = dMRI %>% dplyr::select(-c(eid, age, sex, site))
#T1w = T1w %>% dplyr::select(-c(eid, age, sex, site))
#multi = multi %>% dplyr::select(-c(eid, age, sex, site))
#Demographics for cross sectional data
summary(cross$age)
sd(cross$age)
table(cross$sex)/nrow(cross)
table(cross$site)/nrow(cross)
#
# we check the same stats for the longitudinal data at time point 1
summary(dMRI_T1$age)
sd(dMRI_T1$age)
table(dMRI_T1$sex)/nrow(dMRI_T1)
table(dMRI_T1$site)/nrow(dMRI_T1)
# 
# and tp2
summary(dMRI_T2$age)
sd(dMRI_T2$age)
table(dMRI_T1$sex)/nrow(dMRI_T1)
table(dMRI_T1$site)/nrow(dMRI_T1)

# inter-scan interval
range(dMRI_T2$age-dMRI_T1$age)
mean(dMRI_T2$age-dMRI_T1$age)
sd(dMRI_T2$age-dMRI_T1$age)
#
# build age distribution plots
cross$sex = factor(cross$sex)
levels(cross$sex) = c("Female", "Male")
p1 = cross %>%
  ggplot( aes(x=age, fill=sex,group=sex)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., sex %in% c("Female")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., !sex %in% c("Female")))+
  xlab('Training Sample Age')+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  xlim(40,90) + geom_hline(yintercept = 0, size = .4)
# time point 2 create new df called cross1
cross1 = dMRI_T1
cross1$sex = factor(cross1$sex)
levels(cross1$sex) = c("Female", "Male")
p2 = cross1 %>%
  ggplot( aes(x=age, fill=sex,group=sex)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., sex %in% c("Female")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., !sex %in% c("Female")))+
  xlab('Test Sample Age at Visit 1')+ylab('') + theme_bw() + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  xlim(40,90) + geom_hline(yintercept = 0, size = .4)
# Time point 2
cross1 = dMRI_T2
cross1$sex = factor(cross1$sex)
levels(cross1$sex) = c("Female", "Male")
p3 = cross1 %>%
  ggplot( aes(x=age, fill=sex,group=sex)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., sex %in% c("Female")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., !sex %in% c("Female")))+
  xlab('Test Sample Age at Visit 2')+ylab('') + theme_bw() + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  xlim(40,90) + geom_hline(yintercept = 0, size = .4)
plot00 = ggpubr::ggarrange(p1,p2,p3, nrow = 1, common.legend = T, legend = "bottom")
plot00 = annotate_figure(plot00,top = text_grob("Sample Age Distributions by Sex", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"age_distribution.pdf",sep=""), plot00, width = 8, height = 3)
# remove old dfs and plot objects
rm(cross,p1,p2,p3)
cross = dMRI %>% dplyr::select(-eid, -age, -sex, -site) %>% names()
cross2 = T1w %>% dplyr::select(-eid, -age, -sex, -site) %>% names()
cross_dMRI = c()
dMRIn = data.frame(scale(dMRI[cross]))
dMRIn$sex = dMRI$sex
dMRIn$age = dMRI$age
dMRIn$site = dMRI$site
for (i in 1:length(cross)){
  f = formula(paste(cross[i]," ~ age+sex+site", sep=""))
  model = lm(f, data = dMRIn)
  cross_dMRI[i] = lm.beta(model)$coefficients[2]
}
mean(abs(cross_dMRI))
sd(cross_dMRI)
hist(cross_dMRI)
cross_T1w = c()
T1n = data.frame(scale(T1w[cross2]))
T1n$sex = T1w$sex
T1n$age = T1w$age
T1n$site = T1w$site
for (i in 1:length(cross2)){
  f = formula(paste(cross2[i]," ~ age+sex+site", sep=""))
  model = lm(f, data = T1n)
  cross_T1w[i] = lm.beta(model)$coefficients[2]
}
mean(cross_T1w)
sd(cross_T1w)
#
var.test(cross_T1w,cross_dMRI)
# one can check whether absolute effect sizes change the results, but in fact the variance diff is still sig
var.test(abs(cross_T1w),abs(cross_dMRI), alternative = "two.sided")
#
# quick and dirty power analysis for univariate associations (those will have the lowest power)
#library(pwr)
pwr.f2.test(u = 5,v = 2160,sig.level = 0.05,power = .8)
# u = 5 for metric, sex, site, age, the sex-age interaction
# v = 2166 (polygenic risk score data sets) - 6
#
############################################################################ #
############################################################################ #
# 1) Brain age estimation ####
############################################################################ #
############################################################################ #
# 1.0) Use simple linear regression to train the models
# and test whether simple linear models do the trick
data1 = T1w %>% select(-c(sex, site, eid))
data2 = dMRI %>% select(-c(sex, site, eid))
data3 = multi %>% select(-c(sex, site, eid))
# we use 10-fold CV
# train_control = trainControl(method = "cv",
#                               number = 10)
# model1 = train(age ~., data = data1,
#                method = "lm",
#                trControl = train_control)
# model2 = train(age ~., data = data2, 
#                method = "lm",
#                trControl = train_control)
# model3 = train(age ~., data = data3, 
#                method = "lm",
#                trControl = train_control)
# saveRDS(model1, paste(save_path,"CV_models/LM_T1", sep = ""))
# saveRDS(model2, paste(save_path,"CV_models/LM_dMRI", sep = ""))
# saveRDS(model3, paste(save_path,"CV_models/LM_multi", sep = ""))
# alternatively, read models
model1 = readRDS(paste(save_path,"CV_models/LM_T1", sep = ""))
model2 = readRDS(paste(save_path,"CV_models/LM_dMRI", sep = ""))
model3 = readRDS(paste(save_path,"CV_models/LM_multi", sep = ""))
#
# Note: the 10-fold CV produces the same model as a single shot lm 
#summary(model)
#mod1 = lm(age~.,data = data1)
#summary(mod1)
#
# save model coefficients only
model1 = glm(age ~., data = data1)
model1 = data.frame(model1$coefficients)
write.csv(model1, paste(save_path,"LM_dMRI_Model_Coefficients.csv",sep=""))
model2 = glm(age ~., data = data2)
model2 = data.frame(model2$coefficients)
write.csv(model2, paste(save_path,"LM_T1w_Model_Coefficients.csv",sep=""))
model3 = glm(age ~., data = data3)
model3 = data.frame(model3$coefficients)
write.csv(model3, paste(save_path,"LM_multi_Model_Coefficients.csv",sep=""))
rm(model1,model2,model3)
# make vector of stats formula
perfrow = function(trained_model, data, label){
  r2 = cor(predict(trained_model, data), label)^2
  MAE = Metrics::mae(actual = label,predicted = predict(trained_model, data))
  RMSE = Metrics::rmse(actual = label,predicted = predict(trained_model, data))
  cor = ci_cor(label,predict(trained_model, data))$estimate
  CI95l = ci_cor(label,predict(trained_model, data))$interval[1]
  CI95u = ci_cor(label,predict(trained_model, data))$interval[2]
  perf_row = data.frame(r2, MAE, RMSE, cor, CI95l, CI95u)
  return(perf_row)
}
# apply the formula across data frames (1: dMRI, 2: T1w, 3: multi)
mods = list(model1, model1, model1, model2, model2, model2, model3, model3, model3)
dats = list(data1, T1w_T1, T1w_T2, unlist(data2), unlist(dMRI_T1), unlist(dMRI_T2), data3, multi_T1, multi_T2)
perfdat = data.frame(matrix(nrow=9, ncol=6))
colnames(perfdat) = c("r2", "MAE", "RMSE", "cor", "CI95l", "CI95u")
for (i in 1:length(mods)){
  perfdat[i,] = perfrow(trained_model = mods[[i]], data = dats[[i]], label = dats[[i]]$age)
}
LM_performance_table = perfdat
rm(mods, dats)
#print("We write a performance table for the LM models using the 10-fold CV procedure. XGB and Lasso results can be found another place (python code).")
#write.csv(LM_performance_table, paste(save_path,"LM_performance_table.csv",sep=""))
#
#
print("We now assess the predictions from the other models (XGB & Lasso).")
# load data
# we renew the formula
perfrow = function(prediction, label){
  r2 = cor(prediction, label)^2
  MAE = Metrics::mae(actual = label,predicted = prediction)
  RMSE = Metrics::rmse(actual = label,predicted = prediction)
  cor = ci_cor(label,prediction)$estimate
  CI95l = ci_cor(label,prediction)$interval[1]
  CI95u = ci_cor(label,prediction)$interval[2]
  perf_row = data.frame(r2, MAE, RMSE, cor, CI95l, CI95u)
  return(perf_row)
}
# renew empty data frames
perfdat1 = data.frame(matrix(nrow=3, ncol=6))
colnames(perfdat1) = c("r2", "MAE", "RMSE", "cor", "CI95l", "CI95u")
perfdat2 = perfdat1
perfdat01 = perfdat1
perfdat02 = perfdat1
perfdat3 = perfdat1
perfdat4 = perfdat1
for (i in 1:3){
  perfdat3[i,] = (perfrow(XGB_train_preds[,5+i], XGB_train_preds$age))
  perfdat01[i,] = (perfrow(XGB_T1_preds[,5+i], XGB_T1_preds$age))
  perfdat02[i,] = (perfrow(XGB_T2_preds[,5+i], XGB_T2_preds$age))
  perfdat4[i,] = (perfrow(Lasso_train_preds[,5+i], Lasso_train_preds$age))
  perfdat1[i,] = (perfrow(Lasso_T1_preds[,5+i], Lasso_T1_preds$age))
  perfdat2[i,] = (perfrow(Lasso_T2_preds[,5+i], Lasso_T2_preds$age))
}
perfdat = rbind(perfdat01, perfdat02, perfdat1, perfdat2, perfdat3, perfdat4)
perfdat = rbind(LM_performance_table, perfdat)
row.names(perfdat) = c("LM T1 training", "LM T1 TP1", "LM T1 TP2", "LM dMRI training", "LM dMRI TP1","LM dMRI TP2", "LM multi training", "LM multi TP1", "LM multi TP2",
                       "XGB T1 training", "XGB T1 TP1", "XGB T1 TP2", "XGB dMRI training", "XGB dMRI TP1","XGB dMRI TP2", "XGB multi training", "XGB multi TP1", "XGB multi TP2",
                       "Lasso T1 training", "Lasso T1 TP1", "Lasso T1 TP2", "Lasso dMRI training", "Lasso dMRI TP1","Lasso dMRI TP2", "Lasso multi training", "Lasso multi TP1", "Lasso multi TP2")
write.csv(perfdat, paste(save_path,"Model_comparison_table.csv",sep=""))
rm(perfdat, perfdat01, perfdat02, perfdat1, perfdat2, perfdat3, perfdat4)
#
#
#
#
############################################################################ #
############################################################################ #
# 1.2) Estimate age-correlations in training and test data (3 scatter plots)
############################################################################ #
############################################################################ #
#
############################################################################ #
# 1.2.1) Uncorrected age predictions
############################################################################ #
#
# make a function for plotting with flexible data frames, y-var, and labels
scatter.plot = function(data, Metric, xtext, ytext){
  ggplot(data = data, mapping = aes(x = age, y = Metric)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = 45, label.y = 90)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = 45, label.y = 88)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") +
    theme_bw()
}

# dMRI scatter
train_plot_dMRI = scatter.plot(dMRI, predict(model2, dMRI), c("Training Sample Age"), c("dMRI predicted age"))
test_plot1_dMRI = scatter.plot(dMRI_T1, predict(model2, dMRI_T1), c("Test Sample Age at Visit 1"), c(""))
test_plot2_dMRI = scatter.plot(dMRI_T2, predict(model2, dMRI_T2), c("Test Sample Age at Visit 2"), c(""))
# T1w scatter
train_plot_T1w = scatter.plot(T1w, predict(model1, T1w), c("Training Sample Age"), c("T1w MRI predicted age"))
test_plot1_T1w = scatter.plot(T1w_T1, predict(model1, T1w_T1), c("Test Sample Age at Visit 1"), c(""))
test_plot2_T1w = scatter.plot(T1w_T2, predict(model1, T1w_T2), c("Test Sample Age at Visit 2"), c(""))
# multimodal scatter 
train_plot_multi = scatter.plot(multi, predict(model3, multi), c("Training Sample Age"), c("multimodal MRI predicted age"))
test_plot1_multi = scatter.plot(multi_T1, predict(model3, multi_T1), c("Test Sample Age at Visit 1"), c(""))
test_plot2_multi = scatter.plot(multi_T2, predict(model3, multi_T2), c("Test Sample Age at Visit 2"), c(""))
# arrange the three plots for each modality and label them
plot01 = ggpubr::ggarrange(train_plot_dMRI, test_plot1_dMRI, test_plot2_dMRI,
                   train_plot_T1w, test_plot1_T1w, test_plot2_T1w,
                   train_plot_multi, test_plot1_multi, test_plot2_multi,
                   ncol = 3,nrow=3, labels = NULL, 
                   hjust=-0.8, vjust = 1.75, common.legend = T, legend = "bottom")
plot01 = annotate_figure(plot01,top = text_grob("Associations between Chronological and Predicted Ages", color = "black", face = "bold", size = 14))
# save preliminary figure
ggsave(file = paste(save_path,"uncorrected_train_test_performance.pdf",sep=""), plot01, width = 9, height = 9)
#
#
#
#
#
#
############################################################################ #
# 1.2.2) Corrected age predictions
############################################################################ #
#
#
#
# Correct predictions based on the training data's linear associations between
# chronological and predicted ages
#
# create function to estimate the corrected predicted age
correct = function(training_frame, predicted_age_train, predicted_age_test, age_test){
  corlm = lm(predicted_age_train~age,data = training_frame)
  intercept = summary(corlm)$coefficients[1]
  slope = summary(corlm)$coefficients[2]
  predicted_age_test + (age_test - (slope*predicted_age_test+intercept))
}
# apply it to the three predictions (dMRI, T1w, multimodal)
training_predictions = data.frame(eid = multi$eid,age = multi$age, pred_age_T1w = predict(model1, T1w), pred_age_multi = predict(model3, multi))
dMRI_training_preds = data.frame(eid = dMRI$eid, pred_age_dMRI = predict(model2, dMRI))
training_predictions = merge(training_predictions, dMRI_training_preds, by = "eid")
test_brainage1_T1w = data.frame(eid = T1w_T1$eid, pred_age_T1w = predict(model1, T1w_T1), age = T1w_T1$age)
test_brainage2_T1w = data.frame(eid = T1w_T2$eid, pred_age_T1w = predict(model1, T1w_T2), age = T1w_T2$age)
test_brainage1_dMRI = data.frame(eid = dMRI_T1$eid, pred_age_dMRI = predict(model2, dMRI_T1), age = dMRI_T1$age)
test_brainage2_dMRI = data.frame(eid = dMRI_T2$eid, pred_age_dMRI = predict(model2, dMRI_T2), age = dMRI_T2$age)
test_brainage1_multi = data.frame(eid = multi_T1$eid, pred_age_multi = predict(model3, multi_T1), age = multi_T2$age)
test_brainage2_multi = data.frame(eid = multi_T2$eid, pred_age_multi = predict(model3, multi_T2), age = multi_T2$age)
# dMRI
training_predictions$pred_age_dMRI_corrected = correct(training_predictions, training_predictions$pred_age_dMRI, training_predictions$pred_age_dMRI, training_predictions$age)
test_brainage1_dMRI$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_dMRI, test_brainage1_dMRI$pred_age, test_brainage1_dMRI$age)
test_brainage2_dMRI$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_dMRI, test_brainage2_dMRI$pred_age, test_brainage2_dMRI$age)
# T1w
training_predictions$pred_age_T1w_corrected = correct(training_predictions, training_predictions$pred_age_T1w, training_predictions$pred_age_T1w, training_predictions$age)
test_brainage1_T1w$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_T1w, test_brainage1_T1w$pred_age, test_brainage1_T1w$age)
test_brainage2_T1w$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_T1w, test_brainage2_T1w$pred_age, test_brainage2_T1w$age)
# multimodal
training_predictions$pred_age_multi_corrected = correct(training_predictions, training_predictions$pred_age_multi, training_predictions$pred_age_multi, training_predictions$age)
test_brainage1_multi$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_multi, test_brainage1_multi$pred_age, test_brainage1_multi$age)
test_brainage2_multi$pred_age_corrected = correct(training_predictions, training_predictions$pred_age_multi, test_brainage2_multi$pred_age, test_brainage2_multi$age)
#
#
# remove the very large models
rm(model1, model2, model3)
# Plot with the same settings as done in the plots for the uncorrected age predictions
# Yet, update the label position in the formula
scatter.plot = function(data, Metric, xtext, ytext){
  ggplot(data = data, mapping = aes(x = age, y = Metric)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = 45, label.y = 82)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = 45, label.y = 90)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") +
    theme_bw()
}
# dMRI scatter
train_plot_dMRI = scatter.plot(training_predictions, training_predictions$pred_age_dMRI_corrected, "Training Sample Age", "dMRI predicted age")
test_plot1_dMRI = scatter.plot(test_brainage1_dMRI, test_brainage1_dMRI$pred_age_corrected, "Test Sample Age at Visit 1", "")
test_plot2_dMRI = scatter.plot(test_brainage2_dMRI, test_brainage2_dMRI$pred_age_corrected, "Test Sample Age at Visit 2", "")
# T1w scatter 
train_plot_T1w = scatter.plot(training_predictions, training_predictions$pred_age_T1w_corrected, "Training Sample Age", "T1w MRI predicted age")
test_plot1_T1w = scatter.plot(test_brainage1_T1w, test_brainage1_T1w$pred_age_corrected, "Test Sample Age at Visit 1", "")
test_plot2_T1w = scatter.plot(test_brainage2_T1w, test_brainage2_T1w$pred_age_corrected, "Test Sample Age at Visit 2", "")
# multimodal scatter 
train_plot_multi = scatter.plot(training_predictions, training_predictions$pred_age_multi_corrected, "Training Sample Age", "multimodal MRI predicted age")
test_plot1_multi = scatter.plot(test_brainage1_multi, test_brainage1_multi$pred_age_corrected, "Test Sample Age at Visit 1", "")
test_plot2_multi = scatter.plot(test_brainage2_multi, test_brainage2_multi$pred_age_corrected, "Test Sample Age at Visit 2", "")
# arrange the three plots for each modality and label them
p2 = ggpubr::ggarrange(train_plot_dMRI, test_plot1_dMRI, test_plot2_dMRI,
                   train_plot_T1w, test_plot1_T1w, test_plot2_T1w,
                   train_plot_multi, test_plot1_multi, test_plot2_multi,
                   ncol = 3,nrow=3, labels = NULL, 
                   hjust=-0.8, vjust = 1.75, common.legend = T, legend = "bottom")
# save preliminary figure
ggsave(file = paste(save_path,"corrected_train_test_performance.pdf",sep=""), p2, width = 9, height = 9)
#
rm(p2, train_plot_dMRI, test_plot1_dMRI, test_plot2_dMRI,train_plot_T1w, test_plot1_T1w, test_plot2_T1w,train_plot_multi, test_plot1_multi, test_plot2_multi)
#
#
#
#
#
############################################################################ #
############################################################################ #
# 1.3) Assess model metrics
############################################################################ #
############################################################################ #
# 
##################################### START WITH UNCORRECTED BRAIN AGE
#
# linear models for R2 estimation
mtrain_dMRI = lm(pred_age_dMRI ~ age, training_predictions)
mtest1_dMRI = lm(pred_age_dMRI ~ age, test_brainage1_dMRI)
mtest2_dMRI = lm(pred_age_dMRI ~ age, test_brainage2_dMRI)
mtrain_T1w = lm(pred_age_T1w ~ age, training_predictions)
mtest1_T1w = lm(pred_age_T1w ~ age, test_brainage1_T1w)
mtest2_T1w = lm(pred_age_T1w ~ age, test_brainage2_T1w)
mtrain_multi = lm(pred_age_multi ~ age, training_predictions)
mtest1_multi = lm(pred_age_multi ~ age, test_brainage1_multi)
mtest2_multi = lm(pred_age_multi ~ age, test_brainage2_multi)
# name indicating columns
Prediction = c("dMRI training", "dMRI test visit 1", "dMRI test visit 2", 
         "T1w MRI training", "T1w MRI test visit 1", "T1w MRI test visit 2", 
         "multimodal MRI training", "multimodal MRI test visit 1", "multimodal MRI test visit 2")
# R2
R2 = c(summary(mtrain_dMRI)$r.squared,
       summary(mtest1_dMRI)$r.squared,
       summary(mtest2_dMRI)$r.squared,
       summary(mtrain_T1w)$r.squared,
       summary(mtest1_T1w)$r.squared,
       summary(mtest2_T1w)$r.squared,
       summary(mtrain_multi)$r.squared,
       summary(mtest1_multi)$r.squared,
       summary(mtest2_multi)$r.squared)
#MAE
MAE = c(Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_dMRI),
  Metrics::mae(actual = test_brainage1_dMRI$age,predicted = test_brainage1_dMRI$pred_age_dMRI),
  Metrics::mae(actual = test_brainage2_dMRI$age,predicted = test_brainage2_dMRI$pred_age_dMRI),
  Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_T1w),
  Metrics::mae(actual = test_brainage1_T1w$age,predicted = test_brainage1_T1w$pred_age_T1w),
  Metrics::mae(actual = test_brainage2_T1w$age,predicted = test_brainage2_T1w$pred_age_T1w),
  Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_multi),
  Metrics::mae(actual = test_brainage1_multi$age,predicted = test_brainage1_multi$pred_age_multi),
  Metrics::mae(actual = test_brainage2_multi$age,predicted = test_brainage2_multi$pred_age_multi))
#
# RMSE
RMSE = c(Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_dMRI),
        Metrics::rmse(actual = test_brainage1_dMRI$age,predicted = test_brainage1_dMRI$pred_age_dMRI),
        Metrics::rmse(actual = test_brainage2_dMRI$age,predicted = test_brainage2_dMRI$pred_age_dMRI),
        Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_T1w),
        Metrics::rmse(actual = test_brainage1_T1w$age,predicted = test_brainage1_T1w$pred_age_T1w),
        Metrics::rmse(actual = test_brainage2_T1w$age,predicted = test_brainage2_T1w$pred_age_T1w),
        Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_multi),
        Metrics::rmse(actual = test_brainage1_multi$age,predicted = test_brainage1_multi$pred_age_multi),
        Metrics::rmse(actual = test_brainage2_multi$age,predicted = test_brainage2_multi$pred_age_multi))
# correlations
Correlation = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI)$estimate,
         ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_dMRI)$estimate,
         ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_dMRI)$estimate,
         ci_cor(training_predictions$age,training_predictions$pred_age_T1w)$estimate,
         ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_T1w)$estimate,
         ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_T1w)$estimate,
         ci_cor(training_predictions$age,training_predictions$pred_age_multi)$estimate,
         ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_multi)$estimate,
         ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_multi)$estimate)
# 95% CI lower
CI95l = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI)$interval[1],
          ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_dMRI)$interval[1],
          ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_dMRI)$interval[1],
          ci_cor(training_predictions$age,training_predictions$pred_age_T1w)$interval[1],
          ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_T1w)$interval[1],
          ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_T1w)$interval[1],
          ci_cor(training_predictions$age,training_predictions$pred_age_multi)$interval[1],
          ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_multi)$interval[1],
          ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_multi)$interval[1])
# 95% CI upper
CI95u = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI)$interval[2],
          ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_dMRI)$interval[2],
          ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_dMRI)$interval[2],
          ci_cor(training_predictions$age,training_predictions$pred_age_T1w)$interval[2],
          ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_T1w)$interval[2],
          ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_T1w)$interval[2],
          ci_cor(training_predictions$age,training_predictions$pred_age_multi)$interval[2],
          ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_multi)$interval[2],
          ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_multi)$interval[2])
# make table
performance_table = data.frame(Prediction, R2, MAE, RMSE, Correlation, CI95l, CI95u)
# 
##################################### CONTINUE WITH CORRECTED BRAIN AGE
#
# linear models for R2 estimation
mtrain_dMRI = lm(pred_age_dMRI_corrected ~ age, training_predictions)
mtest1_dMRI = lm(pred_age_corrected ~ age, test_brainage1_dMRI)
mtest2_dMRI = lm(pred_age_corrected ~ age, test_brainage2_dMRI)
mtrain_T1w = lm(pred_age_T1w_corrected ~ age, training_predictions)
mtest1_T1w = lm(pred_age_corrected ~ age, test_brainage1_T1w)
mtest2_T1w = lm(pred_age_corrected ~ age, test_brainage2_T1w)
mtrain_multi = lm(pred_age_multi_corrected ~ age, training_predictions)
mtest1_multi = lm(pred_age_corrected ~ age, test_brainage1_multi)
mtest2_multi = lm(pred_age_corrected ~ age, test_brainage2_multi)
# R2
R2 = c(summary(mtrain_dMRI)$r.squared,
       summary(mtest1_dMRI)$r.squared,
       summary(mtest2_dMRI)$r.squared,
       summary(mtrain_T1w)$r.squared,
       summary(mtest1_T1w)$r.squared,
       summary(mtest2_T1w)$r.squared,
       summary(mtrain_multi)$r.squared,
       summary(mtest1_multi)$r.squared,
       summary(mtest2_multi)$r.squared)
#MAE
MAE = c(Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_dMRI_corrected),
        Metrics::mae(actual = test_brainage1_dMRI$age,predicted = test_brainage1_dMRI$pred_age_corrected),
        Metrics::mae(actual = test_brainage2_dMRI$age,predicted = test_brainage2_dMRI$pred_age_corrected),
        Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_T1w_corrected),
        Metrics::mae(actual = test_brainage1_T1w$age,predicted = test_brainage1_T1w$pred_age_corrected),
        Metrics::mae(actual = test_brainage2_T1w$age,predicted = test_brainage2_T1w$pred_age_corrected),
        Metrics::mae(actual = training_predictions$age,predicted = training_predictions$pred_age_multi_corrected),
        Metrics::mae(actual = test_brainage1_multi$age,predicted = test_brainage1_multi$pred_age_corrected),
        Metrics::mae(actual = test_brainage2_multi$age,predicted = test_brainage2_multi$pred_age_corrected))
#
# RMSE
RMSE = c(Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_dMRI_corrected),
         Metrics::rmse(actual = test_brainage1_dMRI$age,predicted = test_brainage1_dMRI$pred_age_corrected),
         Metrics::rmse(actual = test_brainage2_dMRI$age,predicted = test_brainage2_dMRI$pred_age_corrected),
         Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_T1w_corrected),
         Metrics::rmse(actual = test_brainage1_T1w$age,predicted = test_brainage1_T1w$pred_age_corrected),
         Metrics::rmse(actual = test_brainage2_T1w$age,predicted = test_brainage2_T1w$pred_age_corrected),
         Metrics::rmse(actual = training_predictions$age,predicted = training_predictions$pred_age_multi_corrected),
         Metrics::rmse(actual = test_brainage1_multi$age,predicted = test_brainage1_multi$pred_age_corrected),
         Metrics::rmse(actual = test_brainage2_multi$age,predicted = test_brainage2_multi$pred_age_corrected))
# correlations
Correlation = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI_corrected)$estimate,
                ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_corrected)$estimate,
                ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_corrected)$estimate,
                ci_cor(training_predictions$age,training_predictions$pred_age_T1w_corrected)$estimate,
                ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_corrected)$estimate,
                ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_corrected)$estimate,
                ci_cor(training_predictions$age,training_predictions$pred_age_multi_corrected)$estimate,
                ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_corrected)$estimate,
                ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_corrected)$estimate)
# 95% CI lower
CI95l = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI_corrected)$interval[1],
          ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_corrected)$interval[1],
          ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_corrected)$interval[1],
          ci_cor(training_predictions$age,training_predictions$pred_age_T1w_corrected)$interval[1],
          ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_corrected)$interval[1],
          ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_corrected)$interval[1],
          ci_cor(training_predictions$age,training_predictions$pred_age_multi_corrected)$interval[1],
          ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_corrected)$interval[1],
          ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_corrected)$interval[1])
# 95% CI upper
CI95u = c(ci_cor(training_predictions$age,training_predictions$pred_age_dMRI_corrected)$interval[2],
          ci_cor(test_brainage1_dMRI$age,test_brainage1_dMRI$pred_age_corrected)$interval[2],
          ci_cor(test_brainage2_dMRI$age,test_brainage2_dMRI$pred_age_corrected)$interval[2],
          ci_cor(training_predictions$age,training_predictions$pred_age_T1w_corrected)$interval[2],
          ci_cor(test_brainage1_T1w$age,test_brainage1_T1w$pred_age_corrected)$interval[2],
          ci_cor(test_brainage2_T1w$age,test_brainage2_T1w$pred_age_corrected)$interval[2],
          ci_cor(training_predictions$age,training_predictions$pred_age_multi_corrected)$interval[2],
          ci_cor(test_brainage1_multi$age,test_brainage1_multi$pred_age_corrected)$interval[2],
          ci_cor(test_brainage2_multi$age,test_brainage2_multi$pred_age_corrected)$interval[2])
# put it all into one table
performance_table_corrected = data.frame(Prediction, R2, MAE, RMSE, Correlation, CI95l, CI95u)
# bind the performance tables for corrected and uncorrected brain ages
performance_table = rbind(performance_table, performance_table_corrected)
# make a dummy to show which predictions are the corrected and which are the uncorrected ones
performance_table$Correction = c(replicate(nrow(performance_table_corrected), "No"), replicate(nrow(performance_table_corrected), "Yes"))
# save table
write.csv(performance_table, paste(save_path,"performance_table.csv",sep=""))
#
#
#
#
############################################################################ #
############################################################################ #
# 1.4) Estimate annual rate of brain age change
############################################################################ #
############################################################################ #
# we estimate the uncorrected BAG (BAGu) and the corrected BAG (BAGc)
# first for the first time points
BAGdf = list(test_brainage1_dMRI, test_brainage1_T1w, test_brainage1_multi)
for (i in 1:length(BAGdf)){
  BAGdf[[i]]$BAG1u = BAGdf[[i]][2] - BAGdf[[i]][3]
  BAGdf[[i]]$BAG1c = BAGdf[[i]][4] - BAGdf[[i]][3]
}
# then for the second time points
BAGdf2 = list(test_brainage2_dMRI, test_brainage2_T1w, test_brainage2_multi)
for (i in 1:length(BAGdf2)){
  BAGdf2[[i]]$BAG2u = BAGdf2[[i]][2] - BAGdf2[[i]][3]
  BAGdf2[[i]]$BAG2c = BAGdf2[[i]][4] - BAGdf2[[i]][3]
}
# third, estimate centercept/mean/cross-sectional BAG and the annual rate of change in BAG
# for the centercept, we need the slope between age and brain age gap at tp1
centercepts = data.frame(matrix(ncol = length(BAGdf), nrow = nrow(BAGdf[[1]])))
corrected_centercepts = centercepts
# RoC = rate of change
RoCu = centercepts
RoCc = centercepts
# ISI = inter scan interval
ISI = test_brainage2_dMRI$age - test_brainage1_dMRI$age
for (i in 1:length(BAGdf2)){
  centercepts[i] = (BAGdf2[[i]]$BAG2u - BAGdf[[i]]$BAG1u)/2
  corrected_centercepts[i] = (BAGdf2[[i]]$BAG2c - BAGdf[[i]]$BAG1c)/2
  RoCu[i] = (BAGdf2[[i]]$BAG2u - BAGdf[[i]]$BAG1u)/ISI
  RoCc[i] = (BAGdf2[[i]]$BAG2c - BAGdf[[i]]$BAG1c)/ISI
}
# small df with time point 1 age and
tmp = test_brainage1_dMRI %>% select(eid, age)
# add sex, site and ISI
tmp$ISI = ISI
tmp2 = dMRI_T1 %>% select(eid, sex, site)
tmp = merge(tmp, tmp2, by = "eid")
# merge data frames and provide meaningful names
BAG = cbind(tmp, centercepts, RoCu, corrected_centercepts, RoCc)
names(BAG) = c("eid", "age", "ISI", "sex", "site",
                "CCu_dMRI", "CCu_T1w", "CCu_multi", "RoCu_dMRI", "RoCu_T1w", "RoCu_multi",
                "CCc_dMRI", "CCc_T1w", "CCc_multi", "RoCc_dMRI", "RoCc_T1w", "RoCc_multi")
# remove unused lists and data frames
rm(RoCu, RoCc, centercepts, corrected_centercepts, tmp, tmp2)
#
#
#
# # WE CAN ALSO LOOK AT THE MEAN-DEPENDENCE OF BAG.
# # NOTE:variable names will need som changing (old code!)
# # While it does not appear to be the case here, we will repeat the analyses
# # training and predicting on each of the time points individually.
# #
# ## do an additional test on the mean-dependence of uncorrected BAG
# y = BAG2-BAG1
# x = (test_brainage2$age + test_brainage1$age)/2
# z = (BAG2 +BAG1)/2
# data = data.frame(x=x,y=y,z=z)
# mod <- gam(y ~ s(x) + s(z) + ti(x, z), data = data)
# fvisgam(mod, view=c('x', 'z'))
# summary(mod)
# 
# ## and one for corrected BAG
# y = BAG2c-BAG1c
# x = (test_brainage2$age + test_brainage1$age)/2
# z = (BAG2c +BAG1c)/2
# data = data.frame(x=x,y=y,z=z)
# mod <- gam(y ~ s(x) + s(z) + ti(x, z), data = data)
# fvisgam(mod, view=c('x', 'z'))
# summary(mod)
#
#
#
############################################################################ #
############################################################################ #
# 2. Relationship between cross-sectional brain ages ####
############################################################################ #
############################################################################ #
#
#
########## first, associations between brain ages and brain age gaps between time points
#
#
tp1 = list(test_brainage1_dMRI, test_brainage1_T1w, test_brainage1_multi)
tp2 = list(test_brainage2_dMRI, test_brainage2_T1w, test_brainage2_multi)
# EACH VARIABLE'S correlation coefficient, lower and upper 95%CI can be printed in a vector.
# The vectors are then merged into a data frame for each modality
TPcors = list()
for (i in 1:length(tp1)){
  names(tp1[[i]]) = c("eid", "pred_age", "age", "pred_age_corrected")
  names(tp2[[i]]) = c("eid", "pred_age", "age", "pred_age_corrected")
  names(BAGdf[[i]]) = c("eid", "pred_age", "age", "pred_age_corrected", "BAG1u", "BAG1c")
  names(BAGdf2[[i]]) = c("eid", "pred_age", "age", "pred_age_corrected", "BAG2u", "BAG2c")
  tmp = c(ci_cor(tp1[[i]]$pred_age, tp2[[i]]$pred_age)$estimate,ci_cor(tp1[[i]]$pred_age, tp2[[i]]$pred_age)$interval)
  tmp2 = c(ci_cor(tp1[[i]]$pred_age_corrected, tp2[[i]]$pred_age_corrected)$estimate,ci_cor(tp1[[i]]$pred_age_corrected, tp2[[i]]$pred_age_corrected)$interval)
  tmp3 = c(ci_cor(unlist(BAGdf2[[i]]$BAG2u), unlist(BAGdf[[i]]$BAG1u))$estimate,ci_cor(unlist(BAGdf2[[i]]$BAG2u), unlist(BAGdf[[i]]$BAG1u))$interval)
  tmp4 = c(ci_cor(unlist(BAGdf2[[i]]$BAG2c), unlist(BAGdf[[i]]$BAG1c))$estimate,ci_cor(unlist(BAGdf2[[i]]$BAG2c), unlist(BAGdf[[i]]$BAG1c))$interval)
  TPcors[[i]] = rbind(tmp, tmp2, tmp3, tmp4)
}
# now, we merge the data frames to represent all modalities
TPcors = data.frame(do.call(rbind,TPcors))
# name columns and rows correctly
names(TPcors) = c("Correlation", "lower 95%CI", "upper 95%CI")
TPcors$Variable = c(replicate(3, c("Predicted Age","Corrected Predicted Age", "BAG","Corrected BAG")))
# and then we add a column specifying which modality we are talking about
TPcors$Modality = c(replicate(4,"dMRI"), replicate(4,"T1w MRI"), replicate(4,"multimodal MRI"))
TPcors = TPcors %>% select(Modality, Variable, Correlation, `lower 95%CI`, `upper 95%CI`)
# Then save the Table
write.csv(TPcors, paste(save_path,"BAG_TimePointCorrelations.csv",sep=""), row.names = F)
# finally, remove now unnecessary dust from the the env
rm(tp1, tp2, tmp, tmp2, tmp3, tmp4)
#
#
################## Second: Are there differences in brain age between time points?
#
#
# For that we use mixed linear modelling controlling for the usual covariates
#
# create data frames of variables to be added to BAG/pred_age frames for regression
demo1 = dMRI_T1 %>% select(eid, site, sex)
demo2 = dMRI_T2 %>% select(eid, site, sex)
# create empty temporary dfs and lists to be filled
BAGdf_reg = list()
for (i in 1:3){
  # standardize names
  #BAG1 = data.frame(BAGdf[[i]])
  #BAG2 = data.frame(BAGdf2[[i]])
  names(BAGdf[[i]]) = c("eid", "predicted_age", "age", "predicted_age_corrected", "BAGu", "BAGc")
  names(BAGdf2[[i]]) = c("eid", "predicted_age", "age", "predicted_age_corrected", "BAGu", "BAGc")
  #row.names(BAG1) = c(1:nrow(BAG1))
  #row.names(BAG2) = (nrow(BAG1)+1):(nrow(BAG1)+nrow(BAG1))
  # add demographics
  BAGdf[[i]] = merge(BAGdf[[i]], demo1, by = "eid")
  BAGdf2[[i]] = merge(BAGdf2[[i]], demo2, by = "eid")
  BAGdf_reg[[i]] = bind_rows(BAGdf[[i]], BAGdf2[[i]])
}
# create data frames for diffusion, T1w, and multimodal data & add time point dummy
diff.dat = as.data.frame(BAGdf_reg[[1]])
diff.dat$TP = c(replicate(nrow(demo1), 1), replicate(nrow(demo1), 2))
T1w.dat = data.frame(BAGdf_reg[[2]])
T1w.dat$TP = diff.dat$TP
multi.dat = data.frame(BAGdf_reg[[3]])
multi.dat$TP = diff.dat$TP
#
# Make a function for regression modelling to spit out coefficients of interest
# those are TP difference, SE, R2 (marginal and conditional)
mod.tab = function(data){
  fitted_model_u = lmer(unlist(BAGu) ~ TP + age*sex + site + (1|eid), data = data)
  fitted_model_c = lmer(unlist(BAGc) ~ TP + age*sex + site + (1|eid), data = data)
  # also possible to not correct:
  #fitted_model_u = lmer(BAGu ~ TP + (1|eid), data = data)
  #fitted_model_c = lmer(BAGc ~ TP + (1|eid), data = data)
  a = summary(fitted_model_u)$coefficients[2] # corrected TP difference in BAG
  b = summary(fitted_model_u)$coefficients[2,2] # and its standard error
  b1 = summary(fitted_model_u)$coefficients[2,5] # p-val
  c = r.squaredGLMM(fitted_model_u)
  d = summary(fitted_model_c)$coefficients[2] # uncorrected TP difference in BAG
  e = summary(fitted_model_c)$coefficients[2,2] # and its standard error
  e1 = summary(fitted_model_u)$coefficients[2,5] # p-val
  f = r.squaredGLMM(fitted_model_c)
  coef.vec1 = c(a,b,b1,c)
  coef.vec2 = c(d,e,e1,f)
  tmpdf = data.frame(rbind(coef.vec1, coef.vec2))
  names(tmpdf) = c("Beta", "SE", "p", "R2m", "R2c")
  row.names(tmpdf) = c("Uncorrected", "Corrected")
  return(tmpdf)
  rm(a,b,b1, c,d,e, e1,f, tmpdf)
}
# Run regression models correcting for usual covariates comparing BAGs between TPs
a = mod.tab(diff.dat)
b = mod.tab(T1w.dat)
c = mod.tab(multi.dat)
TP.diff.tab = rbind(a,b,c)
row.names(TP.diff.tab) = c("dMRI Uncorrected", "dMRI Corrected",
                           "T1w Uncorrected", "T1w Corrected",
                           "multimodal Uncorrected", "multimodal Corrected")
# show mean structure in addition to these differences
M.TP = rbind(
  t(data.frame(diff.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGu))))[2]),
  t(data.frame(diff.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGc))))[2]),
  t(data.frame(T1w.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGu))))[2]),
  t(data.frame(T1w.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGc))))[2]),
  t(data.frame(multi.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGu))))[2]),
  t(data.frame(multi.dat %>% group_by(TP) %>% summarize(Mu = mean(unlist(BAGc))))[2])
)
colnames(M.TP) = c("Mean TP1", "Mean TP2")
row.names(M.TP) = c("BAGu diffusion weighted","BAGc diffusion weighted", "BAGu T1-weighted","BAGc T1-weighted", "BAGu multimodal",  "BAGc multimodal")
TP.diff.tab = cbind(TP.diff.tab, M.TP)
write.csv(TP.diff.tab, paste(save_path,"BAG_TP_differences.csv",sep=""), row.names = T)
rm(a,b,c,TP.diff.tab, M.TP)
#
#
# # For the Figure, we estimate Cohen's d and the surrounding 95% CI (see ggplot defs)
# ## uncorrected BAGs
# cohen.d(diff.dat$BAGu, diff.dat$TP, paired = T)
# cohen.d(T1w.dat$BAGu, T1w.dat$TP, paired = T)
# cohen.d(multi.dat$BAGu, multi.dat$TP, paired = T)
# ## corrected BAGs
# cohen.d(diff.dat$BAGc, diff.dat$TP, paired = T)
# cohen.d(T1w.dat$BAGc, T1w.dat$TP, paired = T)
# cohen.d(multi.dat$BAGc, multi.dat$TP, paired = T)
# make plots for corrected and uncorrected BAG (with median line)
#
## dMRI = Panel 1
diff.dat$TP = as.factor(diff.dat$TP)
diff.dat$TP = factor(diff.dat$TP, levels = c("2","1"))
diff.dat$BAGu = unlist(diff.dat$BAGu)
diff.dat$BAGc = unlist(diff.dat$BAGc)
panel1 = diff.dat %>% select(TP, BAGu, BAGc) %>% 
  rename("Time Point" = "TP", "corrected" = "BAGc", 
         "uncorrected" = "BAGu") %>%
  melt(id.vars = "Time Point") %>% ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 15) + theme(legend.position = "bottom") + ylab("") + xlab("Years") + 
  ggtitle("Diffusion MRI BAG") + xlim(-20,20)
panel1 = panel1 + annotate("text", label = (paste("d = ", round(cohen.d(diff.dat$BAGc, diff.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 2.5, size = 6, hjust = 0)
panel1 = panel1 + annotate("text", label = (paste("d = ", round(cohen.d(diff.dat$BAGu, diff.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 1.5, size = 6, hjust = 0)
## T1w = panel 2
T1w.dat$TP = as.factor(T1w.dat$TP)
T1w.dat$TP = factor(T1w.dat$TP, levels = c("2","1"))
T1w.dat$BAGu = unlist(T1w.dat$BAGu)
T1w.dat$BAGc = unlist(T1w.dat$BAGc)
panel2 = T1w.dat %>% select(TP, BAGu, BAGc) %>% 
  rename("Time Point" = "TP", "corrected" = "BAGc", 
         "uncorrected" = "BAGu") %>%
  melt(id.vars = "Time Point") %>% ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 15) + theme(legend.position = "bottom") + ylab("") + xlab("Years") + 
  ggtitle("T1w MRI BAG") + xlim(-20,20)
panel2 = panel2 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$BAGc, T1w.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 2.5, size = 6, hjust = 0)
panel2 = panel2 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$BAGu, T1w.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 1.5, size = 6, hjust = 0)
## multimodal MRI = panel3
multi.dat$TP = as.factor(multi.dat$TP)
multi.dat$TP = factor(multi.dat$TP, levels = c("2","1"))
multi.dat$BAGu = unlist(multi.dat$BAGu)
multi.dat$BAGc = unlist(multi.dat$BAGc)
panel3 = multi.dat %>% select(TP, BAGu, BAGc) %>% 
  rename("Time Point" = "TP", "corrected" = "BAGc", 
         "uncorrected" = "BAGu") %>%
  melt(id.vars = "Time Point") %>% ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 15) + theme(legend.position = "bottom") + ylab("") + xlab("Years") + 
  ggtitle("Multimodal MRI BAG") + xlim(-20,20)
panel3 = panel3 + annotate("text", label = (paste("d = ", round(cohen.d(multi.dat$BAGc, multi.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 2.5, size = 6, hjust = 0)
panel3 = panel3 + annotate("text", label = (paste("d = ", round(cohen.d(multi.dat$BAGu, multi.dat$TP, paired = T)$estimate, 3), sep = "")), x = 13.5, y = 1.5, size = 6, hjust = 0)
p = ggpubr::ggarrange(panel1, panel2, panel3, common.legend = T, ncol = 1, legend = "bottom")
ggsave(file = paste(save_path,"TP_differences_BAG.pdf",sep=""),  p, width = 9, height = 9)
rm(panel1, panel2, panel3, p)
#
#
# make plots for age, predicted and corrected predicted age (with median line)
#
# dMRI
#
panel1 = diff.dat %>% select(age, TP, predicted_age, predicted_age_corrected) %>% 
  rename("Age" = "age","Time Point" = "TP","Predicted\nAge" = "predicted_age", "Corrected\nPredicted\nAge" = "predicted_age_corrected") %>% 
  melt(id.vars = "Time Point") %>%
  ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + 
  xlab("Age") + ggtitle("Diffusion MRI Brain Age") + xlim(40,90) +
  theme(plot.title = element_text(size = 10)) +
  scale_x_continuous(breaks = seq(40, 80, 10), limits = c(40,90))
panel1 = panel1 + annotate("text", label = (paste("d = ", round(cohen.d(diff.dat$predicted_age_corrected, diff.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 3.5, size = 4, hjust = 0,fontface =2)
panel1 = panel1 + annotate("text", label = (paste("d = ", round(cohen.d(diff.dat$predicted_age, diff.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 2.5, size = 4, hjust = 0,fontface =2)
panel1 = panel1 + annotate("text", label = (paste("d = ", round(cohen.d(diff.dat$age, diff.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 1.5, size = 4, hjust = 0,fontface =2)
#
# T1w
#
panel2 = T1w.dat %>% select(age, TP, predicted_age, predicted_age_corrected) %>% 
  rename("Age" = "age","Time Point" = "TP","Predicted\nAge" = "predicted_age", "Corrected\nPredicted\nAge" = "predicted_age_corrected") %>% 
  melt(id.vars = "Time Point") %>%
  ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + 
  xlab("Age")+ ggtitle("T1w MRI Brain Age") + xlim(40,90) +
  theme(plot.title = element_text(size = 10)) +
  scale_x_continuous(breaks = seq(40, 80, 10), limits = c(40,90))
panel2 = panel2 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$predicted_age_corrected, T1w.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 3.5, size = 4, hjust = 0,fontface =2)
panel2 = panel2 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$predicted_age, T1w.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 2.5, size = 4, hjust = 0,fontface =2)
panel2 = panel2 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$age, T1w.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 1.5, size = 4, hjust = 0,fontface =2)
#
# multimodal MRI
#
multi.dat$TP = as.factor(multi.dat$TP)
multi.dat$age = T1w.dat$age
panel3 = multi.dat %>% select(age, TP, predicted_age, predicted_age_corrected) %>% 
  rename("Age" = "age","Time Point" = "TP","Predicted\nAge" = "predicted_age", "Corrected\nPredicted\nAge" = "predicted_age_corrected") %>% 
  melt(id.vars = "Time Point") %>%
  ggplot(aes(x = value, y = variable, fill = `Time Point`)) +
  geom_density_ridges(aes(fill = `Time Point`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + 
  xlab("Age")+ ggtitle("Multimodal MRI Brain Age") + xlim(40,90) +
  theme(plot.title = element_text(size = 10)) +
  scale_x_continuous(breaks = seq(40, 80, 10), limits = c(40,90))
panel3 = panel3 + annotate("text", label = (paste("d = ", round(cohen.d(multi.dat$predicted_age_corrected, multi.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 3.5, size = 4, hjust = 0, fontface =2)
panel3 = panel3 + annotate("text", label = (paste("d = ", round(cohen.d(multi.dat$predicted_age, multi.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 2.5, size = 4, hjust = 0, fontface =2)
panel3 = panel3 + annotate("text", label = (paste("d = ", round(cohen.d(T1w.dat$age, multi.dat$TP, paired = T)$estimate, 2), sep = "")), x = 84.5, y = 1.5, size = 4, hjust = 0, fontface =2)
# arrange the plots
plot02 = ggpubr::ggarrange(panel1, panel2, panel3, common.legend = T, ncol = 1, legend = "right")
plot02 = annotate_figure(plot02, top = text_grob("Time Point Differences for Chronological and Predicted Ages", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"TP_differences_age_pred_age.pdf",sep=""), plot02, width = 9, height = 9)
rm(panel1, panel2, panel3)
#
#
#
#
#
#
############################################################################ #
############################################################################ #
# 3. Relationship between brain age delta and change in brain features ####
############################################################################ #
############################################################################ #
# 3.1 select significant features from paired samples t-test of brain features
############################################################################ #
############################################################################ #
#
########## WE START WITH dMRI
#
#
# create a list of outcome variables which we will use to loop over data frames
outcome_vars = dMRI_T1 %>% select(-c(eid, age, sex, site)) %>% names()
# empty vectors to be filled in loop
p.val = c()
t.val = c()
mean_TP1 = c()
sd_TP1 = c()
mean_TP2 = c()
sd_TP2 = c()
d = c()
d_upper = c()
d_lower = c()
#
for (i in outcome_vars){
  f = formula(paste(i,"~TP + (1|eid)", sep = ""))
  # some of the p values will be rounded to 0. We need to use a more accurate approach (see below)
  t.val[i] = summary(lmer(f, data = df))$coefficients[2,4]
}
# we use Rmpfr to estimate accurate p-vals
.N = function(.) mpfr(., precBits = 200)
t.val = as.numeric(t.val)
p.val = .N(2 * pnorm(-abs(t.val)))
for (i in 1:length(outcome_vars)){
  mean_TP1[i] = mean(dMRI_T1[,i])
  sd_TP1[i] = sd(dMRI_T1[,i])
  mean_TP2[i] = mean(dMRI_T2[,i])
  sd_TP2[i] = sd(dMRI_T2[,i])
  d[i] = cohen.d(dMRI_T2[,i], dMRI_T1[,i], paired=TRUE)$estimate
  d_lower[i] = as.numeric(cohen.d(dMRI_T2[,i], dMRI_T1[,i], paired=TRUE)$conf.int[1])
  d_upper[i] = as.numeric(cohen.d(dMRI_T2[,i], dMRI_T1[,i], paired=TRUE)$conf.int[2])
}
p.adj = mpfr(p.val*length(outcome_vars),100)
paired_t = data.frame(outcome_vars, mean_TP1, sd_TP1, mean_TP2, sd_TP2, t.val, formatMpfr(p.val),formatMpfr(p.adj), d,d_lower, d_upper)
#paired_t_out = data.frame(lapply(paired_t_out,function(x) if(is.numeric(x)) round(x, 3) else x))
paired_t_out = paired_t
colnames(paired_t_out) = c("Metric", "Mean TP1","SD TP1",  "Mean TP2","SD TP1", "T", "p", "adjusted p", "Cohens's d", "Lower 95% CI", "Upper 95% CI")
# write the table
write.csv(paired_t_out, paste(save_path,"dMRI_feature_TP_diff.csv",sep=""), row.names = F)
# make a copy for later reference:
paired_t_out_dMRI = paired_t_out
# remove findings that were non-significant (using the corrected p-val)
## first make adj p numeric (back transformation from mpfr)
keep = list()
keep[[1]] = data.frame(feature = paired_t_out$Metric, 
                  p = as.numeric(paired_t_out$`adjusted p`))
keep[[1]] = keep[[1]] %>% filter(p<0.05)
# now exclude the column names from the df which do not significantly change over time
dfn = subset(df,select = names(df) %in% keep[[1]]$feature)
dfn$eid = df$eid
#
#
#
#
#
################## T1w
# merge timepoint specific data frames
df1 = rbind(T1w_T1, T1w_T2)
df1$TP = c(replicate(nrow(T1w_T1), 0), replicate(nrow(T1w_T1), 1))
# create a list of outcome variables which we will use to loop over data frames
outcome_vars = T1w_T1 %>% select(-c(eid, age, sex, site)) %>% names()
# empty vectors to be filled in loop
p.val = c()
t.val = c()
mean_TP1 = c()
sd_TP1 = c()
mean_TP2 = c()
sd_TP2 = c()
d = c()
d_upper = c()
d_lower = c()
#
for (i in outcome_vars){
  f = formula(paste(i,"~TP + (1|eid)", sep = ""))
  # some of the p values will be rounded to 0. We need to use a more accurate approach (see below)
  t.val[i] = summary(lmer(f, data = df1))$coefficients[2,4]
}
# we use Rmpfr to estimate accurate p-vals
.N = function(.) mpfr(., precBits = 200)
t.val = as.numeric(t.val)
p.val = .N(2 * pnorm(-abs(t.val)))
for (i in 1:length(outcome_vars)){
  mean_TP1[i] = mean(T1w_T1[,i])
  sd_TP1[i] = sd(T1w_T1[,i])
  mean_TP2[i] = mean(T1w_T2[,i])
  sd_TP2[i] = sd(T1w_T2[,i])
  d[i] = cohen.d(T1w_T2[,i], T1w_T1[,i], paired=TRUE)$estimate
  d_lower[i] = as.numeric(cohen.d(T1w_T2[,i], T1w_T1[,i], paired=TRUE)$conf.int[1])
  d_upper[i] = as.numeric(cohen.d(T1w_T2[,i], T1w_T1[,i], paired=TRUE)$conf.int[2])
}
p.adj = mpfr(p.val*length(outcome_vars),100)
paired_t = data.frame(outcome_vars, mean_TP1, sd_TP1, mean_TP2, sd_TP2, t.val, formatMpfr(p.val),formatMpfr(p.adj), d,d_lower, d_upper)
#paired_t_out = data.frame(lapply(paired_t_out,function(x) if(is.numeric(x)) round(x, 3) else x))
paired_t_out = paired_t
colnames(paired_t_out) = c("Metric", "Mean TP1","SD TP1",  "Mean TP2","SD TP1", "T", "p", "adjusted p", "Cohens's d", "Lower 95% CI", "Upper 95% CI")
# write the table
write.csv(paired_t_out, paste(save_path,"T1w_feature_TP_diff.csv",sep=""), row.names = F)
#
# remove findings that were non-significant (using the corrected p-val)
## first make adj p numeric (back transformation from mpfr)
keep[[2]] = data.frame(feature = paired_t_out$Metric, 
                  p = as.numeric(paired_t_out$`adjusted p`))
keep[[2]] = keep[[2]] %>% filter(p<0.05)
# make another keep df including features of both dMRI and T1w MRI
keep[[3]] = rbind(keep[[1]], keep[[2]])
# now exclude the column names from the df1 which do not significantly change over time
df1n = subset(df1,select = names(df1) %in% keep[[2]]$feature)
df1n$eid = df1$eid
#
#
################ MULTIMODAL
tmp1 = merge(T1w_T1, dMRI_T1, by = "eid")
eid = tmp1$eid
tmp1 = subset(tmp1,select = names(tmp1) %in% keep[[3]]$feature)
tmp1$eid = eid
tmp2 = merge(T1w_T2, dMRI_T2, by = "eid")
eid = tmp2$eid
tmp2 = subset(tmp2,select = names(tmp2) %in% keep[[3]]$feature)
tmp2$eid = eid
df2n = rbind(tmp1, tmp2)
rm(eid, tmp1, tmp2)
#
#
#
############################################################################ #
############################################################################ #
# 3.2 compute annual rate of change in brain features
############################################################################ #
############################################################################ #
# We have filtered the data frames to include only features that actually change between time points.
# these are dfn, df1n, df2n, which require the TP var
dfn$TP = df$TP
df1n$TP = df$TP
df2n$TP = df$TP
#
frames = list(dfn, df1n, df2n)
AROC = list()
Centercept = list()
for (i in 1:3){
  # filter the data
  tmp0 = frames[[i]] %>% filter(TP == 0) %>% select(-c(TP))
  tmp1 = frames[[i]] %>% filter(TP == 0) %>% select(-c(eid, TP))
  tmp2 = frames[[i]] %>% filter(TP == 1) %>% select(-c(eid, TP))
  # annual rate of change estimation
  AROC[[i]] = (tmp2[1:nrow(keep[[i]])]-tmp1[1:nrow(keep[[i]])])/(dMRI_T2$age - dMRI_T1$age)
  # centercept of dMRI_T1 and dMRI_T2 estimation (in principle average time-wise)
  Centercept[[i]] = (tmp2[1:nrow(keep[[i]])]+tmp1[1:nrow(keep[[i]])])*0.5
  #AROC[[i]]$eid = tmp0$eid
  #Centercept[[i]]$eid = tmp0$eid
}
# remove tmp data frames, and the single dfs. However, we keep the list for later use
rm(tmp0, tmp1, tmp2, dfn, df1n, df2n)
#
#
############ COMPUTE PRINCIPAL COMPONENTS
res.pca = list()
res.pca1 = list()
# eig.val = list()
# eig.val1 = list()
for (i in 1:3){
  # PCs of the rate of change
  res.pca[[i]] = prcomp(AROC[[i]], scale. = T, center = T)
  # #compute eigenvalues and variance explained
  # eig.val[[i]] = get_eigenvalue(res.pca[[i]])
  # PCs of the centercept of features
  res.pca1[[i]] = prcomp(Centercept[[i]], scale. = T, center = T)
}
# make screeplots
## rate of change
a = fviz_eig(res.pca[[1]], main = "Diffusion MRI features' rate of change", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,30)
b = fviz_eig(res.pca[[2]], main = "T1w MRI features' rate of change", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,30)
c = fviz_eig(res.pca[[3]], main = "Multimodal MRI features' rate of change", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,30)
## Centercepts
d = fviz_eig(res.pca1[[1]], main = "Diffusion MRI features' centercept", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,40)
e = fviz_eig(res.pca1[[2]], main = "T1w MRI features' centercept", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,40)
f = fviz_eig(res.pca1[[3]], main = "Multimodal MRI features' centercept", addlabels=TRUE, hjust = -0.3,
             linecolor ="red") + theme_minimal() + ylim(0,40)
plot = ggarrange(a,b,c,d,e,f,nrow = 2)
ggsave(file = paste(save_path,"Scree_Plots.pdf",sep=""), plot, width = 16, height = 7)
rm(a,b,c,d,e,f, plot)
#
# # for viz of loadings
# fviz_pca_ind(res.pca[[1]],
#              col.ind = "cos2", pointsize = "cos2",
#              gradient.cols = c("#FFCCFF", "#CC0066", "#000000"),
#              repel = F)
#
#
# now, we can extract the first longitudinal component and add it to the brain age data frame
# these components are based on the weighting of variables found in res.pca[[i]]$rotation
BAG$PClong_dMRI = as.numeric(res.pca[[1]]$x[,1])
BAG$PCcross_dMRI = as.numeric(res.pca1[[1]]$x[,1])
BAG$PClong_T1w = as.numeric(res.pca[[2]]$x[,1])
BAG$PCcross_T1w = as.numeric(res.pca1[[2]]$x[,1])
BAG$PClong_multi = as.numeric(res.pca[[3]]$x[,1])
BAG$PCcross_multi = as.numeric(res.pca1[[3]]$x[,1])
#
#
#
############################################################################ #
############################################################################ #
# 3.3 TEST HYPOTHESES
############################################################################ #
############################################################################ #
#
#
############################################################################ #
############################################################################ #
# 3.3.1 H1: Associations between cross-sectional and longitudinal measures
############################################################################ #
############################################################################ #
#
#
#
# WE HAVE A SINGLE DATA FRAME FOR CORRECTED AND UNCORRECTED BAG: "BAG"
#
#
# HYPOTHESIS 1a: longitudinal and cross sectional measures show a relationship
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," + ISI + age*sex + site", sep = ""))
  H1a01 = lm(f, data = BAG)
  Beta = (summary(H1a01)$coefficients[2])
  Beta_standardized = summary(lm.beta(H1a01))$coefficients[2,2]
  SE = (summary(H1a01)$coefficients[2,2])
  t = summary(H1a01)$coefficients[2,3]
  p = summary(H1a01)$coefficients[2,4]
  return(data.frame(Beta,Beta_standardized, SE, t, p))
} 
# the function returns the beta coefficient and SE in years
# make a list of all the combinations to look at
# (including corrected and uncorrected BAGs across MRI modalities)
BAG_pairs = data.frame(x = c("CCc_dMRI", "CCc_T1w", "CCc_multi",
                 "CCu_dMRI", "CCu_T1w", "CCu_multi"),
           y = c("RoCc_dMRI", "RoCc_T1w", "RoCc_multi",
                 "RoCu_dMRI", "RoCu_T1w", "RoCu_multi"))
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 5))
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])
}
names(res1) = c("Beta","Std.Beta", "SE", "t", "p")
rownames(res1) = c("dMRI corrected BAG", "T1w corrected BAG", "multimodal corrected BAG",
                   "dMRI uncorrected BAG", "T1w uncorrected BAG", "multimodal uncorrected BAG")
# now, we look at the association of longitudinal and centercept PCs
PC_pairs = data.frame(x = c("PCcross_dMRI", "PCcross_T1w", "PCcross_multi"),
                   y = c("PClong_dMRI", "PClong_T1w", "PClong_multi"))
res2 = data.frame(matrix(nrow = nrow(PC_pairs), ncol = 5))
for (i in 1:nrow(PC_pairs)){
  res2[i,] = H_test(PC_pairs$x[i], PC_pairs$y[i])
}
names(res2) = names(res1)
rownames(res2) = c("dMRI PC", "T1w PC", "multimodal PC")
res = rbind(res1, res2)
write.csv(x=res, paste(save_path,"H1a_long_cross_associations_no_interaction.csv",sep=""))
rm(res, res1, res2)
#
#
#
############################ PLOTTING
#
# create new plotting function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -3, label.y = 1.75)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -3, label.y = 2.75)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-2,3)+xlim(-3,3)+
    theme_bw()
}
#
# visualize the associations
#
# corrected BAG cross/long associations
p1 = scatter.plot2(BAG$CCc_dMRI,BAG$RoCc_dMRI, c("Centercept of corrected dMRI BAG"), c("dMRI BAG\nRate of Change"))
p2 = scatter.plot2(BAG$CCc_T1w,BAG$RoCc_T1w, c("Centercept of corrected T1w BAG"), c("T1w BAG\nRate of Change"))
p3 = scatter.plot2(BAG$CCc_multi,BAG$RoCc_multi, c("Centercept of corrected multimodal BAG"), c("multimodal BAG\nRate of Change"))
corrected = ggpubr::ggarrange(p1,p2,p3, ncol = 3, common.legend = T, legend = "right")
corrected = annotate_figure(corrected, top = text_grob("Cross-Sectional and Longitudinal Corrected BAG Associations", 
                                           color = "black", size = 12))#face = "bold"
# renew function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -6, label.y = 2.5)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -6, label.y = 5.5)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-6,6)+xlim(-6,6)+
    theme_bw()
}
# uncorrected BAG cross/long associations
p1 = scatter.plot2(BAG$CCu_dMRI,BAG$RoCu_dMRI, c("Centercept of uncorrected dMRI BAG"), c("dMRI BAG\nRate of Change"))
p2 = scatter.plot2(BAG$CCu_T1w,BAG$RoCu_T1w, c("Centercept of uncorrected T1w BAG"), c("T1w BAG\nRate of Change"))
p3 = scatter.plot2(BAG$CCu_multi,BAG$RoCu_multi, c("Centercept of uncorrected multimodal BAG"), c("multimodal BAG\nRate of Change"))
uncorrected = ggpubr::ggarrange(p1,p2,p3, ncol = 3, common.legend = T, legend = "right")
uncorrected = annotate_figure(uncorrected, top = text_grob("Cross-Sectional and Longitudinal Uncorrected BAG Associations", 
                                                           color = "black", size = 12)) #, face = "bold"
plot05 = ggpubr::ggarrange(corrected,uncorrected, ncol = 1, common.legend = T)
plot05 = annotate_figure(plot05, top = text_grob("Associations between Longitudinal and Cross-Sectional Brain Age Gap Measures",color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"H1a_long_cross_associations.pdf",sep=""), plot05, width = 11, height = 8)
rm(p1,p2,p3, uncorrected, corrected)
#
#
#
#
#
#
#
#
# HYPOTHESIS 1b
#
#
#
#
#
# H1b: Check whether there are interaction effects between cross sectional measures and the inter-scan interval predicting longitudinal changes in BAG
#
# for that, we first change the linear model from the previous Hypothesis (1a) test slightly
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," * ISI + age*sex + site", sep = "")) # * instead of + between x and ISI
  H1a01 = lm(f, data = BAG2)
  Std.Beta = (summary(H1a01)$coefficients[8]) # 8th position for interaction effect
  SE = (summary(H1a01)$coefficients[8,2]) # same for SE
  t = (summary(H1a01)$coefficients[8,3]) # t value
  p = (summary(H1a01)$coefficients[8,4]) # and p value
  # Now, we also assess the shrinkage of the effect
  Std.Beta_BAG = (summary(H1a01)$coefficients[2])
  SE_BAG = (summary(H1a01)$coefficients[2,2])
  t_BAG = summary(H1a01)$coefficients[2,3]
  p_BAG = summary(H1a01)$coefficients[2,4]
  return(data.frame(Std.Beta, SE, t, p, Std.Beta_BAG, SE_BAG, t_BAG, p_BAG))
}
# scale predictors for better comparability across effects
# (watch out, the new data frame is now part of the updated function)
BAG2 = BAG
BAG2[2] = scale(BAG[2])
BAG2[6:23] = scale(BAG[6:23])
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 8))
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])
}
names(res1) = c("Std.Beta.Interact.", "SE", "t", "p", "Std.Beta.Centerc", "SE.1", "t.1", "p.1")
#
rownames(res1) = c("dMRI corrected BAG", "T1w corrected BAG", "multimodal corrected BAG",
                   "dMRI uncorrected BAG", "T1w uncorrected BAG", "multimodal uncorrected BAG")
#
# now, we look at the association of longitudinal and centercept PCs
PC_pairs = data.frame(x = c("PCcross_dMRI", "PCcross_T1w", "PCcross_multi"),
                      y = c("PClong_dMRI", "PClong_T1w", "PClong_multi"))
res2 = data.frame(matrix(nrow = nrow(PC_pairs), ncol = 8))
for (i in 1:nrow(PC_pairs)){
  res2[i,] = H_test(PC_pairs$x[i], PC_pairs$y[i])
}
names(res2) = names(res1)
rownames(res2) = c("dMRI PC", "T1w PC", "multimodal PC")
res = rbind(res1, res2)
write.csv(res, paste(save_path, "H1b_ISI_BAG_interactions.csv",sep=""))
rm(res, res1, res2)
#
#
#
#
#
#################### As a confirmatory step, we associate PCs and BAG
# we renew the function from H1a
H_test = function(x, y){
  f = formula(paste(y," ~ ", x," + ISI + age*sex + site", sep = ""))
  H1a01 = lm(f, data = BAG2)
  Beta = (summary(H1a01)$coefficients[2])
  SE = (summary(H1a01)$coefficients[2,2])
  t = summary(H1a01)$coefficients[2,3]
  p = summary(H1a01)$coefficients[2,4]
  return(data.frame(Beta, SE, t, p))
} # the function returns the beta coefficient and SE in years
# make a list of all the modelling combinations to look at
# those are all BAGs x all PCs
# (including corrected and uncorrected BAGs across MRI modalities)
BAG_pairs = data.frame(y = c("CCc_dMRI","CCc_dMRI", "CCc_T1w", "CCc_T1w",
                             "CCc_multi", "CCc_multi","CCu_dMRI", "CCu_dMRI",
                             "CCu_T1w","CCu_T1w", "CCu_multi",  "CCu_multi", # Those are the cross-sectional brain ages. 
                             "RoCc_dMRI","RoCc_dMRI", "RoCc_T1w", "RoCc_T1w",
                             "RoCc_multi","RoCc_multi", "RoCu_dMRI","RoCu_dMRI",
                             "RoCu_T1w","RoCu_T1w", "RoCu_multi", "RoCu_multi"), # And these are the longitudinal ones.
                       x = c("PClong_dMRI", "PCcross_dMRI", "PClong_T1w", "PCcross_T1w",
                             "PClong_multi", "PCcross_multi", "PClong_dMRI", "PCcross_dMRI", 
                             "PClong_T1w", "PCcross_T1w", "PClong_multi", "PCcross_multi",
                             "PClong_dMRI", "PCcross_dMRI", "PClong_T1w", "PCcross_T1w",
                             "PClong_multi", "PCcross_multi", "PClong_dMRI", "PCcross_dMRI", 
                             "PClong_T1w", "PCcross_T1w", "PClong_multi", "PCcross_multi")) # The PCs are just the longitudinal and the centercept
# make an empty data frame
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 4))
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])
}
names(res1) = c("Std.Beta", "SE", "t", "p")
rownames(res1) = paste(BAG_pairs$y, BAG_pairs$x, sep = "_")
#
# This (following) result indicates relatively strong associations between PCs and T1w BAG
res1 %>% filter(p < (.05/nrow(res1))) # This seems to be however the only modality showing such association consistently.
#
write.csv(res1, paste(save_path,"BAG_PC_associations.csv",sep=""))
rm(res1)
# what we haven't tested yet are associations of BAGs and PCs across modalities
H_test("CCc_dMRI","PClong_T1w")
H_test("CCc_multi","PClong_T1w")
H_test("CCu_dMRI","PClong_T1w")
H_test("CCu_multi","PClong_T1w")

H_test("CCc_multi","PClong_dMRI")
H_test("CCc_T1w","PClong_dMRI")

H_test("CCc_dMRI","PClong_multi")
H_test("CCc_T1w","PClong_multi")

#
# A portion of this result can be visually more pleasingly respresented charts (see below)
#
# ...
#
# ...
#
# Now, visualize also some of the key associations here, showing only corrected BAG
# renew function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -3, label.y = 85)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -3, label.y = 105)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-60,110)+ xlim(-3,3)+
    theme_bw(base_size = 12)
}
# principal components cross/long associations
## [CORRECTED BAG]
p1 = scatter.plot2(BAG$RoCc_dMRI,BAG$PClong_dMRI, c("Rate of Change in corrected dMRI BAG"), c("Rate of Change dMRI PC"))
p2 = scatter.plot2(BAG$RoCc_dMRI,BAG$PCcross_dMRI, c("Rate of Change in corrected dMRI BAG"), c("Centercept dMRI PC"))
p3 = scatter.plot2(BAG$RoCc_T1w,BAG$PClong_T1w, c("Rate of Change in corrected T1w BAG"), c("Rate of Change T1w PC"))
p4 = scatter.plot2(BAG$RoCc_T1w,BAG$PCcross_T1w, c("Rate of Change in corrected T1w BAG"), c("Centercept T1w PC"))
p5 = scatter.plot2(BAG$RoCc_multi,BAG$PClong_multi, c("Rate of Change in\ncorrected multimodal MRI BAG"), c("Rate of Change multimodal MRI PC"))
p6 = scatter.plot2(BAG$RoCc_multi,BAG$PCcross_multi, c("Rate of Change in\ncorrected multimodal MRI BAG"), c("Centercept multimodal MRI PC"))
pc  = ggpubr::ggarrange(p1,p2,p3,p4,p5,p6, ncol = 2,nrow = 3, common.legend = T, legend = "none")
pc = annotate_figure(pc, top = text_grob("Associations of Corrected Brain Age Gaps and Principal Components",color = "black", size = 12))
#
# renew function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -7.5, label.y = 85)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -7.5, label.y = 105)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-60,110)+ xlim(-7.5,5)+
    theme_bw(base_size = 12)
}
## [UNCORRECTED BAG]
p1 = scatter.plot2(BAG$RoCu_dMRI,BAG$PClong_dMRI, c("Rate of Change in uncorrected dMRI BAG"), c("Rate of Change dMRI PC"))
p2 = scatter.plot2(BAG$RoCu_dMRI,BAG$PCcross_dMRI, c("Rate of Change in uncorrected dMRI BAG"), c("Centercept dMRI PC"))
p3 = scatter.plot2(BAG$RoCu_T1w,BAG$PClong_T1w, c("Rate of Change in uncorrected T1w BAG"), c("Rate of Change T1w PC"))
p4 = scatter.plot2(BAG$RoCu_T1w,BAG$PCcross_T1w, c("Rate of Change in uncorrected T1w BAG"), c("Centercept T1w PC"))
p5 = scatter.plot2(BAG$RoCu_multi,BAG$PClong_multi, c("Rate of Change in\nuncorrected multimodal MRI BAG"), c("Rate of Change multimodal MRI PC"))
p6 = scatter.plot2(BAG$RoCu_multi,BAG$PCcross_multi, c("Rate of Change in\nuncorrected multimodal MRI BAG"), c("Centercept multimodal MRI PC"))
pu  = ggpubr::ggarrange(p1,p2,p3,p4,p5,p6, ncol = 2,nrow = 3, common.legend = T, legend = "none")
pu = annotate_figure(pu, top = text_grob("Associations of Uncorrected Brain Age Gaps and Principal Components",color = "black", size = 12))
plot06 = ggpubr::ggarrange(pc,pu,nrow = 1,ncol=2)
plot06 = annotate_figure(plot06, top = text_grob("Associations between the Rate of Change in Brain Age Gap and Principal Components of Brain Features' Change and Centercept",color = "black", size = 14, face = "bold"))
ggsave(file = paste(save_path,"PC_rate_of_BAG_change_Associations.pdf",sep=""), plot06, width = 20, height = 10)
rm(p1,p2,p3,p4,p5,p6,pu,pc)
#
#
#
################ As a supplemental figure, we do also centercept brain age with PCs
# Now, visualize also some of the key associations here, showing only corrected BAG
#
#
# renew function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -3, label.y = 85)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -3, label.y = 105)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-60,110)+ xlim(-3,3)+
    theme_bw(base_size = 12)
}
# principal components cross/long associations
## [CORRECTED BAG]
p1 = scatter.plot2(BAG$CCc_dMRI,BAG$PClong_dMRI, c("Centercept of corrected dMRI BAG"), c("Rate of Change dMRI PC"))
p2 = scatter.plot2(BAG$CCc_dMRI,BAG$PCcross_dMRI, c("Centercept of corrected dMRI BAG"), c("Centercept dMRI PC"))
p3 = scatter.plot2(BAG$CCc_T1w,BAG$PClong_T1w, c("Centercept of corrected T1w BAG"), c("Rate of Change T1w PC"))
p4 = scatter.plot2(BAG$CCc_T1w,BAG$PCcross_T1w, c("Centercept of corrected T1w BAG"), c("Centercept T1w PC"))
p5 = scatter.plot2(BAG$CCc_multi,BAG$PClong_multi, c("Centercept of\ncorrected multimodal MRI BAG"), c("Rate of Change multimodal MRI PC"))
p6 = scatter.plot2(BAG$CCc_multi,BAG$PCcross_multi, c("Centercept of\ncorrected multimodal MRI BAG"), c("Centercept multimodal MRI PC"))
pc  = ggpubr::ggarrange(p1,p2,p3,p4,p5,p6, ncol = 2,nrow = 3, common.legend = T, legend = "bottom")
pc = annotate_figure(pc, top = text_grob("Associations of Corrected Brain Age Gaps and Principal Components",color = "black", size = 12))
#
# renew function
scatter.plot2 = function(x, y, xtext, ytext){
  ggplot(data = BAG, mapping = aes(x = x, y = y)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = -7.5, label.y = 85)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = -7.5, label.y = 105)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") + ylim(-60,110)+ xlim(-7.5,5)+
    theme_bw(base_size = 12)
}
## [UNCORRECTED BAG]
p1 = scatter.plot2(BAG$CCu_dMRI,BAG$PClong_dMRI, c("Centercept of uncorrected dMRI BAG"), c("Rate of Change dMRI PC"))
p2 = scatter.plot2(BAG$CCu_dMRI,BAG$PCcross_dMRI, c("Centercept of uncorrected dMRI BAG"), c("Centercept dMRI PC"))
p3 = scatter.plot2(BAG$CCu_T1w,BAG$PClong_T1w, c("Centercept of uncorrected T1w BAG"), c("Rate of Change T1w PC"))
p4 = scatter.plot2(BAG$CCu_T1w,BAG$PCcross_T1w, c("Centercept of uncorrected T1w BAG"), c("Centercept T1w PC"))
p5 = scatter.plot2(BAG$CCu_multi,BAG$PClong_multi, c("Centercept of\nuncorrected multimodal MRI BAG"), c("Rate of Change multimodal MRI PC"))
p6 = scatter.plot2(BAG$CCu_multi,BAG$PCcross_multi, c("Centercept of\nuncorrected multimodal MRI BAG"), c("Centercept multimodal MRI PC"))
pu  = ggpubr::ggarrange(p1,p2,p3,p4,p5,p6, ncol = 2,nrow = 3, common.legend = T, legend = "bottom")
pu = annotate_figure(pu, top = text_grob("Associations of Uncorrected Brain Age Gaps and Principal Components",color = "black", size = 12))
plot06.1 = ggpubr::ggarrange(pc,pu,nrow = 1,ncol=2)
plot06.1 = annotate_figure(plot06.1, top = text_grob("Associations between the Centercept of Brain Age Gap and Principal Components of Brain Features' Change and Centercept",color = "black", size = 14, face = "bold"))
ggsave(file = paste(save_path,"PC_BAG_centercept_Associations.pdf",sep=""), plot06.1, width = 20, height = 10)
rm(p1,p2,p3,p4,p5,p6,pu,pc)
#
#
#
###### Finally, check whether cross-sectional BAG at the first time point can predict brain feature change
# first, we show that only T1w BAG at time point 1 is associated with the respective Principal Component of change (which we already know for T1w data)
m1 = lm(PClong_dMRI ~ age*sex + site + ISI + unlist(BAGdf[[1]]$BAGc), data = BAG)
summary(m1)
m2 = lm(PClong_T1w ~ age*sex + site + ISI + unlist(BAGdf[[2]]$BAGc), data = BAG)
summary(m2)
m3 = lm(PClong_multi ~ age*sex + site + ISI + unlist(BAGdf[[3]]$BAGc), data = BAG)
summary(m3)

# we use the filtered data frames already containing the annual rate of change for each brain feature (AROC) to check regional associations
## FIRST FOR CCc BAGs
########## dMRI data
tmp = cbind(AROC[[1]], ISI = BAG$ISI, BAGc = BAG$CCc_dMRI)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[1]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ BAGc + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp1 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp1, paste(save_path,"BAG_dMRI_feature_change_associations.csv",sep=""))
print("As indicated by the model including the dMRI PC of change, WM BAG does not well in predicting brain changes.")
#
#
########## T1w data
tmp = cbind(AROC[[2]], ISI = BAG$ISI, BAGc = BAG$CCc_T1w)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[2]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ BAGc + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp2 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp2, paste(save_path,"BAG_T1w_feature_change_associations.csv",sep=""))
#
#
########## multimodal data
tmp = cbind(AROC[[3]], ISI = BAG$ISI, BAGc = BAG$CCc_multi)
tmp = scale(tmp)
tmp2 = BAGdf[[3]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[3]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ BAGc + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp3 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp3, paste(save_path,"BAG_multimodal_feature_change_associations.csv",sep=""))
#
#
#
#
## SECOND FOR THE RATE OF CHANGE OF BAG
########## dMRI data
tmp = cbind(AROC[[1]], ISI = BAG$ISI, RoC = BAG$RoCc_dMRI)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[1]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ RoC + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp4 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp4, paste(save_path,"BAG_change_dMRI_feature_change_associations.csv",sep=""))
print("In contrast to the cross-sectional BAG, the rate of change is reflective of brain feature change.")
#
#
########## T1w data
tmp = cbind(AROC[[2]], ISI = BAG$ISI, RoC = BAG$RoCc_T1w)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[2]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ RoC + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp5 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp5, paste(save_path,"BAG_change_T1w_feature_change_associations.csv",sep=""))
#
#
########## multimodal data
tmp = cbind(AROC[[3]], ISI = BAG$ISI, RoC = BAG$RoCc_multi)
tmp = scale(tmp)
tmp2 = BAGdf[[3]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[3]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ RoC + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
exp6 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
write.csv(exp6, paste(save_path,"BAG_change_multimodal_feature_change_associations.csv",sep=""))
#
#
# now, we make a figure presenting the effect size distributions.
expl1 = list(exp1,exp4) #dMRI
expl2 = list(exp2,exp5) #T1w
expl3 = list(exp3,exp6) #multimodal
vec = c("Cross sectional BAG as predictor", "Rate of BAG change as predictor")
#vec = c("dMRI_BAG", "T1w_BAG", "multi_BAG", "dMRI_BAG_change", "T1w_BAG_change", "multi_BAG_change")
for (i in 1:length(expl1)){
  expl1[[i]]$Predictor = replicate(nrow(expl1[[i]]), vec[i]) #dMRI
  expl2[[i]]$Predictor = replicate(nrow(expl2[[i]]), vec[i]) #T1w
  expl3[[i]]$Predictor = replicate(nrow(expl3[[i]]), vec[i]) #multimodal
}
exp = list(expl1, expl2, expl3)
plot_exp = list(exp1,exp2,exp3)
plot_exp2 = list(exp4,exp5,exp6)
expl1 = do.call(rbind, expl1) #dMRI
expl2 = do.call(rbind, expl2) #T1w
expl3 = do.call(rbind, expl3) #multimodal
p1 = expl1 %>%
  ggplot( aes(x=betas, fill=Predictor,group=Predictor)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., !Predictor %in% c("Cross sectional BAG as predictor")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., Predictor %in% c("Cross sectional BAG as predictor")))+
  xlab("Feature Change Association (Std. Beta)")+ylab('Density') + theme_bw(base_size = 12) + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9"))+
  ggtitle("Diffusion MRI") + xlim(-.5,.5) + ylim(-7,10) +
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size=.4) +
  geom_hline(yintercept = 0, size = .4) +
  theme(plot.title = element_text(size = 12))
p2 = expl2 %>%
  ggplot( aes(x=betas, fill=Predictor,group=Predictor)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., !Predictor %in% c("Cross sectional BAG as predictor")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., Predictor %in% c("Cross sectional BAG as predictor")))+
  xlab("Feature Change Association (Std. Beta)")+ylab('Density') + theme_bw(base_size = 12) + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9"))+
  ggtitle("T1-weighted MRI") + xlim(-.5,.5) + ylim(-7,10) +
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size=.4) + 
  geom_hline(yintercept = 0, size = .4) +
  theme(plot.title = element_text(size = 12))
p3 = expl3 %>%
  ggplot( aes(x=betas, fill=Predictor,group=Predictor)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., !Predictor %in% c("Cross sectional BAG as predictor")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., Predictor %in% c("Cross sectional BAG as predictor")))+
  xlab("Feature Change Association (Std. Beta)")+ylab('Density') + theme_bw(base_size = 12) + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  ggtitle("Multimodal MRI") + xlim(-.5,.5) + ylim(-7,10) +
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size=.4) +
  geom_hline(yintercept = 0, size = .4) +
  theme(plot.title = element_text(size = 12))
plot07 = ggpubr::ggarrange(p1,p2,p3, common.legend = T, legend = "bottom", nrow = 1)
plot07 = annotate_figure(plot07, top = text_grob("Distribution of Associations between BAG and Brain Feature Change",color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"Explaining_feature_change.pdf",sep=""), plot07, width = 13, height = 5)
#
# Estimate meta stats
print("First estimate average absolute effect sizes and their standard deviations.")
descriptor = c("TP1 BAG", "Annual change in BAG")
for (i in 1:2){
  print(descriptor[i])
  print(paste("dMRI Mean = ", round(mean(abs(exp[[1]][[i]]$betas)),2),"dMRI SD = ", round(sd(abs(exp[[1]][[i]]$betas)),2)))
  print(paste("T1w Mean = ", round(mean(abs(exp[[2]][[i]]$betas)),2),"T1w SD = ", round(sd(abs(exp[[2]][[i]]$betas)),2)))
  print(paste("multi Mean = ", round(mean(abs(exp[[3]][[i]]$betas)),2), "multi SD = ", round(sd(abs(exp[[3]][[i]]$betas)),2)))
}
print("Second, estimate the proportion of brain features for which brain age could statistically significantly explain change.")
for (i in 1:2){
  print(descriptor[i])
  print(paste("dMRI:", round(exp[[1]][[i]]%>% filter(p.adj < .05) %>% nrow()/nrow(exp[[1]][[i]]),4)*100,"%",sep=""))
  print(paste("T1w:", round(exp[[2]][[i]]%>% filter(p.adj < .05) %>% nrow()/nrow(exp[[2]][[i]]),4)*100,"%",sep=""))
  print(paste("multi:", round(exp[[3]][[i]]%>% filter(p.adj < .05) %>% nrow()/nrow(exp[[3]][[i]]),4)*100,"%",sep=""))
}
#
# Make a small figure of these values.
#
tmp1 = c()
tmp2 = c()
tmp3 = c()
tmp4 = c()
tmp5 = c()
tmp6 = c()
for (i in 1:length(plot_exp)){
  tmp1[i] = mean(abs(plot_exp[[i]]$betas))
  tmp2[i] = mean(abs(plot_exp2[[i]]$betas))
  tmp3[i] = sd(abs(plot_exp[[i]]$betas))
  tmp4[i] = sd(abs(plot_exp2[[i]]$betas))
  tmp5[i] = plot_exp[[i]]%>% filter(p.adj < .05) %>% nrow()/nrow(plot_exp[[i]])
  tmp6[i] = plot_exp2[[i]]%>% filter(p.adj < .05) %>% nrow()/nrow(plot_exp2[[i]])
}
Mean = c(tmp1, tmp2)
SD = c(tmp3,tmp4)
Percentage = c(tmp5, tmp6)
Modality = c("dMRI", "T1w MRI", "multimodal MRI")
BAG_type = c(c(replicate(3, "Cross-Sectional BAG")), c(replicate(3, "Annual Rate of BAG Change")))
plot_frame = data.frame(BAG_type, Modality, Mean, SD, Percentage)
# Make a plot with 2 panels: one for percentages & one for mean ± SD
#
## error bar plots for Mean and SD
p1 = ggplot(plot_frame, aes(Modality, Mean, group = BAG_type, color= BAG_type)) +
  geom_pointrange(aes(ymin = Mean-SD, ymax = Mean+SD), shape = 1, lwd = .6, fatten = 9,
                  position = position_dodge(0.3), color = "black")+
  geom_point(size = 4,position = position_dodge(0.3))+
  scale_color_manual(values = c("#56B4E9","#E69F00")) +
  ylab("Mean±SD of Absolute Stdandardized Betas") + xlab("") + theme_minimal(base_size = 12) +
  labs(color = "") + theme(legend.position="bottom") + coord_flip()
#
# Lollipop plots for percentages
plot_frame$Modality = c("dMRI BAG", "T1w MRI BAG", "multimodal MRI BAG",
                        "dMRI BAG Change", "T1w MRI BAG Change", "multimodal MRI BAG Change")
perc.label = paste(round(plot_frame$Percentage,2)*100,"%", sep = "")
plot_frame$Percentage = plot_frame$Percentage*100
p2 = ggplot(plot_frame, aes(Modality, Percentage, group = BAG_type)) +  
  geom_segment(aes(x = Modality, xend = Modality, y = 0, yend = Percentage), color = "gray", lwd = 1) +
  geom_point(size = 12, pch = 21, bg = c("#E69F00","#E69F00","#E69F00","#56B4E9","#56B4E9","#56B4E9")) +
  geom_text(aes(label = perc.label), color = "black", size = 3) + xlab("") +
  coord_flip() + theme_minimal(base_size = 12) + ylab("% of FDR-corrected p<.05 Feature Associations")
plot08 = ggpubr::ggarrange(p1,p2,nrow = 1, common.legend = T, legend = "bottom")
ggsave(file = paste(save_path,"long_BAG_Feature_Assoc_Dist_lollipop.pdf",sep=""), p1, width = 7, height = 5)
ggsave(file = paste(save_path,"long_BAG_Feature_Assoc_Dist_percentage.pdf",sep=""), p2, width = 7, height = 5)
ggsave(file = paste(save_path,"long_BAG_Feature_Assoc_Dist.pdf",sep=""), plot08, width = 14, height = 5)
rm(exp, plot_exp,plot_exp2, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6)
#
#
#
#
#
#
# A potential explanation for the observed small effects can be found in the 
# distribution of effects for TP-changes per regions indicating stronger changes in T1w features.
# see /Users/max/Documents/Projects/LongBAG/results/predefined_hyperparams/unscaled/T1w_feature_TP_diff.csv
# and /Users/max/Documents/Projects/LongBAG/results/predefined_hyperparams/unscaled/dMRI_feature_TP_diff.csv
part1 = data.frame(d = na.omit(paired_t_out_dMRI$`Cohens's d`), data = replicate(nrow(na.omit(paired_t_out_dMRI[1:11])),"dMRI"))
part2 = data.frame(d = na.omit(paired_t_out$`Cohens's d`), data = replicate(nrow(na.omit(paired_t_out)),"T1w"))
ds = rbind(part1,part2)
# make labels for mean absolute Cohen's d of Feature change between time points
this_label = paste("Mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(part1$d))),2),"**", sep = "")
this_label1 = paste("Mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(part2$d))),2),"**", sep = "")
rm(part1,part2)
plot04 = ds %>%
  ggplot( aes(x=d, fill=data,group=data)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., data %in% c("T1w")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., !data %in% c("T1w")))+
  xlab("Feature Change (Cohen's d)")+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9"))+
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size=.4) +
  geom_hline(yintercept = 0) + xlim(-1,1) +
  geom_richtext(aes(x = 0.2, y = 1.5, label = this_label),
                stat = "unique", angle = 0,
                color = "black", fill = "#E69F00",
                label.color = NA, hjust = 0, vjust = 0) +
  geom_richtext(aes(x = 0.2, y = -1.5, label = this_label1),
                stat = "unique", angle = 0,
                color = "black", fill = "#56B4E9",
                label.color = NA, hjust = 0, vjust = 0)
plot04 = annotate_figure(plot04,top = text_grob("Distribution of Brain Features' Time Point Differences", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"TP_differences_features.pdf",sep=""), plot04, width = 8, height = 5)
#
#
#
#
##### #
# make another plot including the time point difference but this time separated for diffusion approaches
##### # T1w
a11 = ifelse(grepl("thickness",paired_t_out$Metric),'Thickness', NA)
b11 = ifelse(grepl("volume",paired_t_out$Metric),'Volume', NA)
c11 = ifelse(grepl("are",paired_t_out$Metric),'Area', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
paired_t_out$Group = a11
# make data frames for each type of metric
Vol = paired_t_out %>% na.omit() %>% select(!`SD TP1`) %>% filter(Group == "Volume") %>% select(Metric, `Cohens's d`, Group)
names(Vol) = c("Region","d", "Metric")
Area = paired_t_out %>% na.omit() %>% select(!`SD TP1`) %>% filter(Group == "Area") %>% select(Metric, `Cohens's d`, Group)
names(Area) = c("Region","d", "Metric")
Thick = paired_t_out %>% na.omit() %>% select(!`SD TP1`) %>% filter(Group == "Thickness") %>% select(Metric, `Cohens's d`, Group)
names(Thick) = c("Region","d", "Metric")
# create figs
this_label = paste("**Area**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(Area$d))),2),"**", sep = "")
p_area = Area %>% ggplot(aes(x=d)) + geom_density(fill = "#E69F00", alpha = 0.7) + xlab("Feature Change (Cohen's d)")+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) + labs(x = NULL, y = NULL) + xlim(-1,1) +ylim(0,8) +
  geom_richtext(aes(x = -1, y = 7, label = this_label),
                stat = "unique", angle = 0,
                color = "black", fill = "#E69F00",
                label.color = NA, hjust = 0, vjust = 0)
this_label1 = paste("**Thickness**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(Thick$d))),2),"**", sep = "")
p_thick = Thick %>% ggplot(aes(x=d)) + geom_density(fill = "#E69F00", alpha = 0.7) + xlab("Feature Change (Cohen's d)")+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) + labs(x = NULL, y = NULL) + xlim(-1,1) + ylim(0,8) +
  geom_richtext(aes(x = -1, y = 7, label = this_label1),
                stat = "unique", angle = 0,
                color = "black", fill = "#E69F00",
                label.color = NA, hjust = 0, vjust = 0)
this_label2 = paste("**Volume**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(Vol$d))),2),"**", sep = "")
p_vol = Vol %>% ggplot(aes(x=d)) + geom_density(fill = "#E69F00", alpha = 0.7) + xlab("Feature Change (Cohen's d)")+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) + labs(x = NULL, y = NULL) + xlim(-1,1) + ylim(0,8) +
  geom_richtext(aes(x = -1, y = 7, label = this_label2),
                stat = "unique", angle = 0,
                color = "black", fill = "#E69F00",
                label.color = NA, hjust = 0, vjust = 0)
#figure = ggarrange(p_area,p_thick, p_vol, nrow = 1)
#T1fig = annotate_figure(figure, left = textGrob("Density", rot = 90, vjust = 1, gp = gpar(cex = 1.3)),
#                bottom = textGrob("Feature Change in T1w Metrics (Cohen's d)", gp = gpar(cex = 1.3)))
#
#### # dMRI
# make a function for plotting
plot_func = function(data, label){
  data %>% ggplot(aes(x=d)) + geom_density(fill = "#56B4E9", alpha = 0.7) + xlab("Feature Change (Cohen's d)")+ylab('Density') + theme_bw() + 
    theme(legend.title=element_blank()) + labs(x = NULL, y = NULL) + xlim(-1,1)  + ylim(0,4) +
    geom_richtext(aes(x = -1, y = 3.5, label = label),
                  stat = "unique", angle = 0,
                  color = "black", fill = "#56B4E9",
                  label.color = NA, hjust = 0, vjust = 0)
}
# DKI
a11 = ifelse(grepl("ak",paired_t_out_dMRI$Metric),'AK', NA)
b11 = ifelse(grepl("rk",paired_t_out_dMRI$Metric),'RK', NA)
c11 = ifelse(grepl("mk",paired_t_out_dMRI$Metric),'MK', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
paired_t_out_dMRI$Group = a11
DKI = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(DKI) =  c("Region","d", "Metric")
this_label_dki = paste("**DKI**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(DKI$d))),2),"**", sep = "")
p_dki = plot_func(DKI, this_label_dki)
#
# DTI
a11 = ifelse(grepl("^AD_",paired_t_out_dMRI$Metric),'AD', NA)
b11 = ifelse(grepl("^RD_",paired_t_out_dMRI$Metric),'RD', NA)
c11 = ifelse(grepl("^MD_",paired_t_out_dMRI$Metric),'MD', NA)
d11 = ifelse(grepl("^FA_",paired_t_out_dMRI$Metric),'FA', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
a11 = ifelse(is.na(a11) == T, d11, a11)
paired_t_out_dMRI$Group = a11
DTI = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(DTI) = c("Region","d", "Metric")
this_label_dti = paste("**DTI**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(DTI$d))),2),"**", sep = "")
p_dti = plot_func(DTI, this_label_dti)
#
# BRIA (note that micro AX and Drad extra were removed due to quality issues)
a11 = ifelse(grepl("v_intra",paired_t_out_dMRI$Metric),'V intra', NA)
b11 = ifelse(grepl("v_extra",paired_t_out_dMRI$Metric),'V extra', NA)
c11 = ifelse(grepl("csf",paired_t_out_dMRI$Metric),'V CSF', NA)
d11 = ifelse(grepl("micro_Rd",paired_t_out_dMRI$Metric),'micro RD', NA)
e11 = ifelse(grepl("micro_ADC",paired_t_out_dMRI$Metric),'micro ADC', NA)
f11 = ifelse(grepl("Dax_intra",paired_t_out_dMRI$Metric),'DAX intra', NA)
g11 = ifelse(grepl("Dax_extra",paired_t_out_dMRI$Metric),'DAX extra', NA)
h11 = ifelse(grepl("micro_FA",paired_t_out_dMRI$Metric),'micro FA', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
a11 = ifelse(is.na(a11) == T, d11, a11)
a11 = ifelse(is.na(a11) == T, e11, a11)
a11 = ifelse(is.na(a11) == T, f11, a11)
a11 = ifelse(is.na(a11) == T, g11, a11)
a11 = ifelse(is.na(a11) == T, h11, a11)
paired_t_out_dMRI$Group = a11
BRIA = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(BRIA) =  c("Region","d", "Metric")
this_label_BRIA = paste("**BRIA**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(BRIA$d))),2),"**", sep = "")
p_BRIA = plot_func(BRIA, this_label_BRIA)
#
# WMTI (note that axEAD was excluded due to quality problems)
a11 = ifelse(grepl("awf",paired_t_out_dMRI$Metric),'AWF', NA)
b11 = ifelse(grepl("radEAD",paired_t_out_dMRI$Metric),'radEAD', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
paired_t_out_dMRI$Group = a11
WMTI = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(WMTI) =  c("Region","d", "Metric")
this_label_WMTI = paste("**WMTI**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(WMTI$d))),2),"**", sep = "")
p_WMTI = plot_func(WMTI, this_label_WMTI)
#
# SMT
a11 = ifelse(grepl("smt_fa",paired_t_out_dMRI$Metric),'FA', NA)
b11 = ifelse(grepl("smt_md",paired_t_out_dMRI$Metric),'MD', NA)
c11 = ifelse(grepl("smt_trans_",paired_t_out_dMRI$Metric),'Transverse', NA)
d11 = ifelse(grepl("smt_long",paired_t_out_dMRI$Metric),'Longitudinal', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
a11 = ifelse(is.na(a11) == T, d11, a11)
paired_t_out_dMRI$Group = a11
SMT = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(SMT) = c("Region","d", "Metric")
this_label_SMT = paste("**SMT**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(SMT$d))),2),"**", sep = "")
p_SMT = plot_func(SMT, this_label_SMT)
#
# mcSMT
a11 = ifelse(grepl("mc_diff",paired_t_out_dMRI$Metric),'Diffusion Coefficient', NA)
b11 = ifelse(grepl("extramd",paired_t_out_dMRI$Metric),'Extra-Axonal MD', NA)
c11 = ifelse(grepl("extratra",paired_t_out_dMRI$Metric),'Extra-Axonal Transverse', NA)
d11 = ifelse(grepl("mc_intra",paired_t_out_dMRI$Metric),'Intra Neurite Volume', NA)
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
a11 = ifelse(is.na(a11) == T, d11, a11)
paired_t_out_dMRI$Group = a11
SMTmc = paired_t_out_dMRI %>% na.omit() %>% select(!`SD TP1`) %>% select(Metric, `Cohens's d`, Group)
names(SMTmc) =  c("Region","d", "Metric")
this_label_SMTmc = paste("**SMTmc**: mean absolute Cohen\\'s **d = ",round(mean(abs(na.omit(SMTmc$d))),2),"**", sep = "")
p_SMTmc = plot_func(SMTmc, this_label_SMTmc)
#
#
#
# in the end, merge all appraches / metric figs
Dens_Fig = ggarrange(p_area,p_thick, p_vol, p_BRIA, p_dki, p_dti,p_SMT,p_SMTmc, p_WMTI, nrow = 3, ncol = 3)
Dens_Fig = annotate_figure(Dens_Fig, left = textGrob("Density", rot = 90, vjust = 1, gp = gpar(cex = 1.3)),
                           bottom = textGrob("Feature Change (Cohen's d)", gp = gpar(cex = 1.3)))
ggsave(file = paste(save_path,"Changes_by_Approach.pdf",sep=""),Dens_Fig, height = 11, width = 11)
#
#
###### Do the same but outlining all the metrics
# create a plot function
plot_func2 = function(data, label){
  ggplot(data, aes(d , fill=Metric, Metric=Metric, shape=Metric, facets=Metric)) + 
    geom_density()+ facet_wrap(.~Metric) + theme_bw() + 
    xlab("Feature Change (Cohen's d)")+ylab('Density')+ 
    scale_fill_brewer(palette = "Pastel1") + xlim(-1,1)  + ylim(0,8)
}
T1_plot_df = rbind(Vol,Area, Thick)
a = plot_func2(T1_plot_df)
b = plot_func2(BRIA)
d = plot_func2(DKI)
e = plot_func2(DTI)
f = plot_func2(SMT)
g = plot_func2(SMTmc)
h = plot_func2(WMTI)
Metric_Fig = ggarrange(a,b,d,e,f,g,h, labels = c("a","b","c","d","e","f","g"), ncol = 2)
ggsave(file = paste(save_path,"Changes_by_Metric.pdf",sep=""),Metric_Fig, height = 15, width = 13)
#
# print the mean absolute change (indicated by Cohen's d)
T1_plot_df %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d)) 
BRIA %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))
DKI %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))
DTI %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))
SMT %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))
SMTmc %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))
WMTI %>% group_by(Metric) %>% summarize(Mean = mean(d), SD = sd(d), Md = median(d), Mad = mad(d))

############################################################################ #
############################################################################ #
# 3.3.2 H2: Interaction effects between inter-scan interval and changes in 
# BAG on BAG, AND between the ISI and the PC on the PC of change
############################################################################ #
############################################################################ #
#
### GENERALIZED ADDITIVE MODELS
#
#
# H2a: model rate of change in BAG from ISI, centercept BAG, and their interaction, as in (H1a)
# H2b: And the principal component of change from the ISI and the centercept principal component of change and their interaction, as in (H1b).
#
# for reference, check 3 and 4:
#
# WMBAGchange ~ WMBAG + ISI + WMBAG*ISI + age + sex + site + age*sex (3)
# PCcross-sectional ~ PClongitudinal + ISI + Pclongitudinal*ISI + age + sex + site + age*sex (4)
#
# renew function to estimate GAMs (cubic splines with k = 4 knots) and get the model coefficients
# save also the models for later plotting
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
BAG_pairs = data.frame(x = c("CCc_dMRI", "CCc_T1w", "CCc_multi",
                             "CCu_dMRI", "CCu_T1w", "CCu_multi",
                             "PCcross_dMRI", "PCcross_T1w", "PCcross_multi"),
                       y = c("RoCc_dMRI", "RoCc_T1w", "RoCc_multi",
                             "RoCu_dMRI", "RoCu_T1w", "RoCu_multi",
                             "PClong_dMRI", "PClong_T1w", "PClong_multi"))
res1 = data.frame(matrix(nrow = nrow(BAG_pairs), ncol = 6))
plot.mod = list()
for (i in 1:nrow(BAG_pairs)){
  res1[i,] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[1]]
  plot.mod[[i]] = H_test(BAG_pairs$x[i], BAG_pairs$y[i])[[2]]
}
names(res1) = c("Std.Beta", "SE", "t", "p", "F_smooth.Interact", "p.Interact")
rownames(res1) = paste(BAG_pairs$y, BAG_pairs$x, sep = "_")
write.csv(res1, paste(save_path,"H1b_ISI_BAG_smooth_interactions.csv",sep=""))
rm(res1)
# now plot the non-linear interactions
BAG_pairs = data.frame(x = c("CCc_dMRI", "CCc_T1w", "CCc_multi",
                             "CCu_dMRI", "CCu_T1w", "CCu_multi"),
                       y = c("RoCc_dMRI", "RoCc_T1w", "RoCc_multi",
                             "RoCu_dMRI", "RoCu_T1w", "RoCu_multi"))
titles =c("dMRI corrected BAG", "T1w corrected BAG", "multimodal corrected BAG",
                   "dMRI uncorrected BAG", "T1w uncorrected BAG", "multimodal uncorrected BAG")
xlabs = c("Centercept BAG dMRI", "Centercept BAG T1w","Centercept BAG multimodal MRI", 
          "Centercept BAG dMRI", "Centercept BAG T1w","Centercept BAG multimodal MRI")
# create Conotour plot showing the effect of the ISI*Centercept BAG on the rate of change in BAG
pdf(paste(save_path,"contour.pdf", sep=""))
par(mfrow=c(2,3), cex=.7)
for (i in 1:nrow(BAG_pairs)){
  vis.gam(plot.mod[[i]], view = c(BAG_pairs$x[i], "ISI"), plot.type = "contour",color = "terrain", main = titles[i], xlab = xlabs[i], ylab = "Inter-Scan Interval")
}
dev.off()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
############################################################################ #
############################################################################ #
# 3.3.3 H3: Phenotype and genotype associations with ...
############################################################################ #
############################################################################ #
#           - annual change in brain age delta
#           - brain age delta (time point 1 and 2)
#           - principal components of white matter
############################################################################ #
#
#
#
#
#
#
#
#
# FIRST, WE USE LINEAR MODELS CONSIDERING BAG AND PCs AT EACH INDIVIDUAL TIME POINT
#
#
#
# fix eid in PGRS
PGRS$eid = PGRS$FID
PGRS = PGRS %>% dplyr::select(-c(X, FID))
# merge PGRS and phenotypes into pheno frame
pheno = merge(pheno, PGRS, by = "eid")
# scale also the numeric phenotypes on self-reported depression, neuroticism, and WHR
#pheno[3:ncol(pheno)] = scale(pheno[3:ncol(pheno)], center = T)
#
# estimate principal components of all variables for the two data sets
dT1 = dMRI_T1 %>% select(-c(eid,sex,age,site))
dT2 = dMRI_T2 %>% select(-c(eid,sex,age,site))
tT1 = T1w_T1 %>% select(-c(eid,sex,age,site))
tT2 = T1w_T2 %>% select(-c(eid,sex,age,site))
mT1 = multi_T1 %>% select(-c(eid,sex,age,site))
mT2 = multi_T2 %>% select(-c(eid,sex,age,site))
l1 = list(dT1, tT1, mT1)
l2 = list(dT2, tT2, mT2)
res.pca = list()
res.pca1 = list()
for (i in 1:3){
  # PCs of the rate of change
  res.pca[[i]] = prcomp(l1[[i]], scale. = T, center = T)
  # #compute eigenvalues and variance explained
  # eig.val[[i]] = get_eigenvalue(res.pca[[i]])
  # PCs of the centercept of features
  res.pca1[[i]] = prcomp(l2[[i]], scale. = T, center = T)
}
#
# extract the first of each modality's components and merge them correctly based on the eid
dmritmp1 = data.frame(eid = dMRI_T1$eid,dMRI_PC_1 = as.numeric(res.pca[[1]]$x[,1]))
dmritmp2 = data.frame(eid = dMRI_T2$eid,dMRI_PC_2 = as.numeric(res.pca1[[1]]$x[,1]))
t1tmp1 = data.frame(eid = T1w_T1$eid, T1w_PC_1 = as.numeric(res.pca[[2]]$x[,1]))
t1tmp2 = data.frame(eid = T1w_T2$eid, T1w_PC_2 = as.numeric(res.pca1[[2]]$x[,1]))
multitmp1 = data.frame(eid = multi_T1$eid, multi_PC_1 = as.numeric(res.pca[[3]]$x[,1]))
multitmp2 = data.frame(eid = multi_T2$eid, multi_PC_2 = as.numeric(res.pca1[[3]]$x[,1]))
# merge the data frames for each time point
tmp1 = merge(merge(dmritmp1, t1tmp1, by = "eid"), multitmp1, by = "eid")
tmp2 = merge(merge(dmritmp2, t1tmp2, by = "eid"), multitmp2, by = "eid")
# scale the PCs
tmp1[2:4] = scale(tmp1[2:4])
tmp2[2:4] = scale(tmp2[2:4])
# add scaled BAGs
tmp1$dMRIBAG_1 = scale(BAGdf[[1]]$BAGc)
tmp1$T1wBAG_1 = scale(BAGdf[[2]]$BAGc)
tmp1$multimodalBAG_1 = scale(BAGdf[[3]]$BAGc)
tmp2$dMRIBAG_2 = scale(BAGdf2[[1]]$BAGc)
tmp2$T1wBAG_2 = scale(BAGdf2[[2]]$BAGc)
tmp2$multimodalBAG_2 = scale(BAGdf2[[3]]$BAGc)
#
# now, make a selection of phenotypes and merge them with each time point frame
p1 = data.frame(eid = pheno$eid, Depression_1 = pheno$RDS1, 
           Neuroticism_1 = pheno$N1, WHR_1 = pheno$WHR1, 
           Smoking_1 = pheno$smoking1, Hypertension_1 = pheno$hypertension1, 
           Diabetic_1 = pheno$diabetic1,ANX = pheno$ANX, ADHD = pheno$ADHD, 
           ASD = pheno$ASD, BIP = pheno$BIP, MDD = pheno$MDD, OCD = pheno$OCD, 
           SCZ = pheno$SCZ, AD = pheno$AD)
p2 = data.frame(eid = pheno$eid, Depression_2 = pheno$RDS2, 
                Neuroticism_2 = pheno$N2, WHR_2 = pheno$WHR2, 
                Smoking_2 = pheno$smoking2, 
                Hypertension_2 = pheno$hypertension2, 
                Diabetic_2 = pheno$diabetic2)
# scale the phenotypes
p1[2:5] = scale(p1[2:5])
p1[8:15] = scale(p1[8:15])
p2[2:5] = scale(p2[2:5])
# merge pheno, PCs, and BAGs
cross_dat = merge(p2,p1, by = "eid")
cross_dat = merge(merge(merge(tmp1, p1, by = "eid"),tmp2, by = "eid"), p2, by = "eid")
#tmp1 = merge(tmp1, p1, by = "eid")
#tmp2 = merge(tmp2, p1, by = "eid")
# get age,sex,site
d1 = BAGdf[[1]] %>% select(eid,sex, site, age)
names(d1) = c("eid","sex", "site", "age_1")
d2 = BAGdf2[[1]] %>% select(eid,age)
names(d2) = c("eid","age_2")
d1$age_1 = scale(d1$age_1)
d2$age_2 = scale(d2$age_2)
cross_dat = merge(merge(cross_dat, d1, by = "eid"), d2, by = "eid")
# tmp1 = merge(tmp1, d1, by = "eid")
# tmp2 = merge(tmp2, d2, by = "eid")
# put it all together into long format & remove old dat
#data = rbind(tmp1, tmp2)
rm(dmritmp1, dmritmp2, t1tmp1, t1tmp2, multitmp1, multitmp2, p1, d1,d2) # use: tmp1, tmp2
#
# Now, we can do linear modelling on BAGs, PCs, brain age of all three modalities
#
# We start predicting BAGs and PCs from phenotypes
#
predicted1 = c("dMRI_PC_1","T1w_PC_1","multi_PC_1",
              "dMRIBAG_1","T1wBAG_1","multimodalBAG_1")
predicted2 =c("dMRI_PC_2","T1w_PC_2","multi_PC_2",
              "dMRIBAG_2","T1wBAG_2","multimodalBAG_2")
name_labels1 = c("dMRI PC\ntime 1","T1w PC\ntime 1","multi PC\ntime 1",
                 "dMRIBAG\ntime 1","T1wBAG\ntime 1", "multimodalBAG\ntime 1")
name_labels2 = c("dMRI PC\ntime 2","T1w PC\ntime 2","multi PC\ntime 2",
                 "dMRIBAG\ntime 2","T1wBAG\ntime 2", "multimodalBAG\ntime 2")
name_labels = c(name_labels1, name_labels2)
pheno_names1 = names(cross_dat[8:21])
pheno_names2 = pheno_names1
pheno_names2[1:6] = names(cross_dat[28:33])
pheno_labels = c(pheno_names1, pheno_names2)
#
betas1 = data.frame()
betas2 = betas1
ps1 = betas1
ps2 = betas1
for (o in 1:length(predicted1)){
  for (i in 1:length(pheno_names1)){
    f1 = formula(paste(predicted1[o]," ~ age_1*sex + site +",pheno_names1[i], sep = ""))
    f2 = formula(paste(predicted2[o]," ~ age_2*sex + site +",pheno_names2[i], sep = ""))
    tmp_model1 = lm(f1, data = cross_dat)
    betas1[i,o] = summary(tmp_model1)$coefficients[6]
    ps1[i,o] = summary(tmp_model1)$coefficients[6,4]
    tmp_model2 = lm(f2, data = cross_dat)
    betas2[i,o] = summary(tmp_model2)$coefficients[6]
    ps2[i,o] = summary(tmp_model2)$coefficients[6,4]
  }
}
# add names to associations in data frame
names(betas1) = name_labels1
names(betas2) = name_labels2
names(ps1) = name_labels1
names(ps2) = name_labels2
#
# melt p and beta frames for plotting
plot_dat1 = melt(betas1)
p_frame1 = melt(ps1)
plot_dat2 = melt(betas2)
p_frame2 = melt(ps2)
#
# merge beta and p values
plot_dat1$p = p_frame1$value
plot_dat2$p = p_frame2$value
#
plot_dat = rbind(plot_dat1,plot_dat2)
plot_dat$names = c("Depression","Neuroticism","WHR","Smoking","Hypertension", "Diabetic",
                   "ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD")
#
# min p before FDR adjustment
min(plot_dat$p)
# min p after FDR adjustment
min(p.adjust(plot_dat$p, method = "fdr"))
#
# show all the findings surviving FDR correction
plot_dat %>% select(names, variable, value, p) %>% dplyr::filter(p.adjust(p, method = "fdr") < .05)
#
# TILE PLOT FOR GLOBAL METRICS AT EACH TIME POINT
# create colors for frames around tiles
plot_dat$colors = c(ifelse(p.adjust(plot_dat$p, method = "fdr") < .05,"black", "white"))
#
#
#################### CREATE PLOTS FOR TP1
# create plots, start with polygenic risk scores
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[1:6]) %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
panel1 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + 
  scale_x_discrete(labels=name_labels1)
# then health factors
dat1 = plot_dat %>%filter(variable %in%levels(plot_dat$variable)[1:6]) %>% filter(names %in% c("WHR","Smoking","Hypertension","Diabetic"))
panel2 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels=name_labels1)
# then psychological factors
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[1:6]) %>% filter(names %in% c("Depression","Neuroticism"))
panel3 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels=name_labels1)
#
#
#################### CREATE PLOTS FOR TP2
#
#
# create plots, start with polygenic risk scores
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[7:12]) %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
panel4 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + 
  scale_x_discrete(labels=name_labels2)
# then health factors
dat1 = plot_dat %>%filter(variable %in%levels(plot_dat$variable)[7:12]) %>% filter(names %in% c("WHR","Smoking","Hypertension","Diabetic"))
panel5 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels=name_labels2)
# then psychological factors
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[7:12]) %>% filter(names %in% c("Depression","Neuroticism"))
panel6 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels=name_labels2)
#
#
plot03.2 = ggpubr::ggarrange(panel3,panel6, panel2, panel5, panel1, panel4, ncol = 2,nrow=3, common.legend = T, legend = "bottom", align = "hv",heights=c(.35,.575,1), labels=c("d","g", "e","h", "f", "i"))
plot03.2 = annotate_figure(plot03.2,top = text_grob("Cross-Sectional Brain Ages' and Principal Components' Pheno- and Genotype Associations", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"Cross_Sectional_BAG_PC_Phenotype_Associations.pdf",sep=""),plot03.2, width = 12, height = 7)
#
#
#
#
#
#
#
#
#
#
#
# SECOND, WE USE CENTERCEPT & RATE OF CHANGE IN BAG, PCs AND PHENOTYPES (VANILLA LINEAR MODELS)
#
#
pheno_df = pheno
# we first assess which of the selected variables changed significantly over time, 
# using paired samples t-tests IN THE WHOLE POPULATION
t.test(pheno_df$RDS1,pheno_df$RDS2, paired = T)
t.test(pheno_df$N1,pheno_df$N2, paired = T) # is sig (neuroticism decreases)
t.test(pheno_df$WHR1,pheno_df$WHR2, paired = T) # is sig (WHR increases)
t.test(pheno_df$smoking1,pheno_df$smoking2, paired = T)
chi = chisq.test(pheno_df$hypertension1, pheno_df$hypertension2)
chi # for hypertension there seems to be a difference, which is however driven by
chi$observed # missingness in the data, as outlined here.
chi = chisq.test(pheno_df$diabetic1, pheno_df$diabetic2)
chi$observed # Same problem for diabetics.
#
# Now, we do the same in the IMAGING SAMPLE
pheno_df = merge(pheno_df, BAG, by = "eid")
t.test(pheno_df$RDS1,pheno_df$RDS2, paired = T) # marginally sig (decrease in depression)
cohen.d(pheno_df$RDS1,pheno_df$RDS2, paired = T, na.rm = T)
t.test(pheno_df$N1,pheno_df$N2, paired = T) # sig (decrease in neuroticism)
cohen.d(pheno_df$N1,pheno_df$N2, paired = T, na.rm = T)
t.test(pheno_df$WHR1,pheno_df$WHR2, paired = T) # is sig (increase in WHR)
cohen.d(pheno_df$WHR1,pheno_df$WHR2, paired = T, na.rm = T)
t.test(pheno_df$smoking1,pheno_df$smoking2, paired = T)
chi = chisq.test(pheno_df$hypertension1, pheno_df$hypertension2)
# for hypertension there seems to be a difference, which is however driven by
chi$observed # missingness in the data, as outlined here.
chi = chisq.test(pheno_df$diabetic1, pheno_df$diabetic2)
chi$observed # Same problem for diabetics.
print("Based on the absence of time differences, we can simply use the cross-sectional measures of a single time point, assuming constancy.")
print("(Only WHR seems to change over time (increases).)")
#
print("We hence reduce the data frame to the time point 2 phenotype and PGRS measures.")
print("This has already been done in p2")
tpmp = cross_dat %>% select(eid, names(cross_dat[28:33]), names(cross_dat[14:21]))
pheno_df = merge(tpmp, BAG, by = "eid")
#
# We also further examine the relationship of WHR and BAG
## we estimate the rate of change in WHR
pheno_df$RoC_WHR = (pheno_df$WHR2-pheno_df$WHR1)/(pheno_df$ISI)
pheno_df$RoC_Neur = (pheno_df$N2-pheno_df$N1)/(pheno_df$ISI)
pheno_df$RoC_Dep = (pheno_df$RDS2-pheno_df$RDS1)/(pheno_df$ISI)
WHR_models = list()
Neur_models = list()
Dep_models = list()
for (i in 1:18){
  WHR_models[[i]] = (lm(RoC_WHR ~ pheno_df[,26+i] + sex + site + age + ISI, data = pheno_df))
  Neur_models[[i]] = (lm(RoC_Neur ~ pheno_df[,26+i] + sex + site + age + ISI, data = pheno_df))
  Dep_models[[i]] = (lm(RoC_Dep ~ pheno_df[,26+i] + sex + site + age + ISI, data = pheno_df))
}
# check models
for (i in 1:length(Neur_models)){
  print(summary(Dep_models[[i]]))
}
#
#
#
# Present rate of change and centercept (BAG&PC) associations with phenotypes
## for standardized coefficients, we scale variables
names(pheno_df) = c("eid", "Depression", "Neuroticism", "WHR", "Smoking", 
                    "Hypertension", "Diabetic","ANX","ADHD","ASD","BIP","MDD",
                    "OCD","SCZ","AD", "age", "ISI", "sex", "site",
                    "CCu_dMRI","CCu_T1w","CCu_multi","RoCu_dMRI","RoCu_T1w",
                    "RoCu_multi","CCc_dMRI","CCc_T1w", "CCc_multi","RoCc_dMRI",
                    "RoCc_T1w","RoCc_multi","PClong_dMRI","PCcross_dMRI","PClong_T1w",
                    "PCcross_T1w","PClong_multi", "PCcross_multi")
# scale the last unscaled variables
pheno_df$age = as.numeric(scale(pheno_df$age))
pheno_df$ISI = as.numeric(scale(pheno_df$ISI))
pheno_df[20:ncol(pheno_df)] = scale(pheno_df[20:ncol(pheno_df)])
#
#
# we are only interested in corrected BAGs and PCs
predicted = c("RoCc_dMRI","RoCc_T1w","RoCc_multi",
              "PClong_dMRI","PClong_T1w","PClong_multi")
pheno_names = names(pheno_df[2:15])
betas = data.frame()
ps = betas
for (o in predicted){
  for (i in pheno_names){
    f = formula(paste(o," ~ age*sex + site + ISI +",i, sep = ""))
    tmp_model = lm(f, data = pheno_df)
    betas[i,o] = summary(tmp_model)$coefficients[7]
    ps[i,o] = summary(tmp_model)$coefficients[7,4]
  }
}
# add names to associations in data frame
betas$names = rownames(betas)
ps$names = rownames(betas)
#
# melt p and beta frames for plotting
plot_dat1 = melt(betas)
p_frame1 = melt(ps)
#
# merge beta and p values
plot_dat1$p = p_frame1$value
#
# min p before FDR adjustment
min(p_frame1$value)
# min p after FDR adjustment
min(p.adjust(p_frame1$value, method = "fdr"))
#
# we can also filter findings based on p-values before correction to get an overview...
p_frame1 %>% select(names, variable, value) %>% dplyr::filter(value < .05)
#
# TILE PLOT FOR GLOBAL METRICS AT EACH TIME POINT
# create colors for frames around tiles
p_frame1$colors = c(ifelse(p.adjust(p_frame1$value, method = "fdr") < .05,"black", "white"))
#
#
# create plots, start with polygenic risk scores
dat1 = plot_dat1 %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
pd1 = p_frame1 %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
panel1 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = pd1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels= c("BAG change dMRI","BAG change T1w","BAG change multi",
                             "dMRI PC of Change", "T1w PC of Change", "multimodal PC of Change"))
# then health factors
dat1 = plot_dat1 %>% filter(names %in% c("WHR", "Smoking","Hypertension", "Diabetic"))
pd1 = p_frame1 %>% filter(names %in% c("WHR", "Smoking","Hypertension", "Diabetic"))
panel2 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = pd1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels= c("BAG change dMRI","BAG change T1w","BAG change multi",
                             "dMRI PC of Change", "T1w PC of Change", "multimodal PC of Change"))
# then psychological factors
dat1 = plot_dat1 %>% filter(names %in% c("Depression", "Neuroticism"))
pd1 = p_frame1 %>% filter(names %in% c("Depression", "Neuroticism"))
panel3 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = pd1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) +
  scale_x_discrete(labels= c("BAG change dMRI","BAG change T1w","BAG change multi",
                             "dMRI PC of Change", "T1w PC of Change", "multimodal PC of Change"))
plot03.1 = ggpubr::ggarrange(panel3,panel2,panel1, ncol = 1, common.legend = T, legend = "bottom", align = "hv",heights=c(.35,.575,1), labels=c("a","b","c"))
plot03.1 = annotate_figure(plot03.1,top = text_grob("Brain Age Change's and Principal Components of Change's Pheno- and Genotype Associations", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"LM_BAG_PC_change_Phenotype_Associations.pdf",sep=""),plot03.1, width = 14, height = 7)
#
#
#
#
#
############################################################################ #
###################### Patch figures together ############################## #
############################################################################ #
#
# Fig.1: Age can be predicted accurately across time from different modalities presenting brain changes. 
#
panel1 = egg::ggarrange(plot00, plot01, ncol = 1, heights = c(.5,2), labels = c("a", "b"))
panel2 = egg::ggarrange(plot02,plot04, ncol = 1, heights = c(1.6,.4), labels = c("c", "d"))
Fig1 = ggpubr::ggarrange(panel1, panel2, ncol = 2)
ggsave(file = paste(save_path,"Fig1.pdf",sep=""),Fig1, height = 13, width = 15)
# Fig.2: Cross sectional Brain Age estimates are associated
#
Fig2 = egg::ggarrange(plot05, plot06.1, plot07,plot08, ncol = 1, heights = c(1,2.2,.75,.85), labels = c("a", "b", "c", ""))
ggsave(file = paste(save_path,"Fig2.pdf",sep=""),Fig2, height = 18, width = 15)
#
# Fig.3: Geno- and Phenotype Associations with Brain Age
#
Fig3 = egg::ggarrange(plot03.1,plot03.2, ncol = 1, heights = c(.8,1))
ggsave(file = paste(save_path,"Fig3.pdf",sep=""),Fig3, height = 15, width = 12)
#
#
#
#
#
#
#
#
#
############################################################################ #
###################### 4. SUPPLEMENT #########################################
############################################################################ #
#
#
############################################################################ #
# Additional / supplemental analyses on the phenotypes (full sample)
############################################################################ #
#
#
### Correlate within one time point.
ph1 = pheno %>% select(-c(eid, X, hypertension1, hypertension2, diabetic1, diabetic2))
res = cor(ph1, use = "pairwise.complete")
ph = pheatmap(res, display_numbers = T, color = colorRampPalette(c('#0072B2','#F0E442'))(100), cluster_rows = F, cluster_cols = F, fontsize = 15, fontsize_number = 15, main = "UKB-wide Phenotype Associations")
ggsave(file = paste(save_path,"all_pheno_heatmap.pdf",sep=""),ph, height = 11, width = 11)
# Additional / supplemental visualisations phenotype associations (imaging sample)
ph1 = merge(BAG, pheno, by = "eid") %>% select(-c(eid,sex,site,X,hypertension1,hypertension2,diabetic1, diabetic2))
res = cor(ph1, use = "pairwise.complete")
ph = pheatmap(res, display_numbers = T, color = colorRampPalette(c('#0072B2','#F0E442'))(100), cluster_rows = F, cluster_cols = F, fontsize = 15, fontsize_number = 15, main = "Imaging Sample Phenotype Associations")
ggsave(file = paste(save_path,"imaging_pheno_heatmap.pdf",sep=""),ph, height = 20, width = 20)
#
#
#
############################################################################ #
# Check for phenotype associations when using hemispheric or whole brain averages.
############################################################################ #
#
sel1 = dMRI_T1 %>% select(eid, FA_Mean, AD_Mean, RD_Mean, MD_Mean, #DTI
                   mk_Mean, rk_Mean, ak_Mean, #Kurtosis
                   smt_fa_Mean,smt_md_Mean, smt_trans_Mean, #SMT
                   smt_mc_extramd_Mean, smt_mc_extratrans_Mean, smt_mc_diff_Mean, smt_mc_intra_Mean, #SMT mc
                   awf_Mean, radEAD_Mean, #WMTI
                   Dax_intra_Mean,Dax_extra_Mean, micro_FA_Mean,v_intra_Mean,v_extra_Mean,v_csf_Mean,
                   micro_Ax_Mean,micro_Rd_Mean,micro_ADC_Mean) # BRIA
sel2 = dMRI_T2 %>% select(eid, FA_Mean, AD_Mean, RD_Mean, MD_Mean, #DTI
                          mk_Mean, rk_Mean, ak_Mean, #Kurtosis
                          smt_fa_Mean,smt_md_Mean, smt_trans_Mean, #SMT
                          smt_mc_extramd_Mean, smt_mc_extratrans_Mean, smt_mc_diff_Mean, smt_mc_intra_Mean, #SMT mc
                          awf_Mean, radEAD_Mean, #WMTI
                          Dax_intra_Mean,Dax_extra_Mean, micro_FA_Mean,v_intra_Mean,v_extra_Mean,v_csf_Mean,
                          micro_Ax_Mean,micro_Rd_Mean,micro_ADC_Mean) # BRIA
sel3 = T1w_T1 %>% select(eid, lh_MeanThickness_thickness, rh_MeanThickness_thickness, lh_WhiteSurfArea_area, rh_WhiteSurfArea_area)
sel4 = T1w_T2 %>% select(eid, lh_MeanThickness_thickness, rh_MeanThickness_thickness, lh_WhiteSurfArea_area, rh_WhiteSurfArea_area)
# merge it like a cookie dough
glob = merge(merge(merge(merge(cross_dat,sel1), sel3, by = "eid"),sel2, by = "eid"), sel4, by = "eid")
glob[38:95] = scale(glob[38:95])
predicted1 = names(glob[38:66])
predicted2 = names(glob[67:95])
# name_labels1 = c("dMRI PC\ntime 1","T1w PC\ntime 1","multi PC\ntime 1",
#                  "dMRIBAG\ntime 1","T1wBAG\ntime 1", "multimodalBAG\ntime 1")
# name_labels2 = c("dMRI PC\ntime 2","T1w PC\ntime 2","multi PC\ntime 2",
#                  "dMRIBAG\ntime 2","T1wBAG\ntime 2", "multimodalBAG\ntime 2")
# name_labels = c(name_labels1, name_labels2)
# pheno_names1 = names(cross_dat[8:21])
# pheno_names2 = pheno_names1
# pheno_names2[1:6] = names(cross_dat[28:33])
# pheno_labels = c(pheno_names1, pheno_names2)
#
betas1 = data.frame()
betas2 = betas1
ps1 = betas1
ps2 = betas1
for (o in 1:length(predicted1)){
  for (i in 1:length(pheno_names1)){
    f1 = formula(paste(predicted1[o]," ~ age_1*sex + site +",pheno_names1[i], sep = ""))
    f2 = formula(paste(predicted2[o]," ~ age_2*sex + site +",pheno_names2[i], sep = ""))
    tmp_model1 = lm(f1, data = glob)
    betas1[i,o] = summary(tmp_model1)$coefficients[6]
    ps1[i,o] = summary(tmp_model1)$coefficients[6,4]
    tmp_model2 = lm(f2, data = glob)
    betas2[i,o] = summary(tmp_model2)$coefficients[6]
    ps2[i,o] = summary(tmp_model2)$coefficients[6,4]
  }
}
# add names to associations in data frame
names(betas1) = predicted1
names(betas2) = predicted2
names(ps1) = predicted1
names(ps2) = predicted2
#
# melt p and beta frames for plotting
plot_dat1 = melt(betas1)
p_frame1 = melt(ps1)
plot_dat2 = melt(betas2)
p_frame2 = melt(ps2)
#
# merge beta and p values
plot_dat1$p = p_frame1$value
plot_dat2$p = p_frame2$value
#
plot_dat = rbind(plot_dat1,plot_dat2)
plot_dat$names = c("Depression","Neuroticism","WHR","Smoking","Hypertension", "Diabetic",
                   "ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD")
#
# min p before FDR adjustment
min(plot_dat$p)
# min p after FDR adjustment
min(p.adjust(plot_dat$p, method = "fdr"))
#
# show all the findings surviving FDR correction
plot_dat %>% select(names, variable, value, p) %>% dplyr::filter(p.adjust(p, method = "fdr") < .05)
#
# TILE PLOT FOR GLOBAL METRICS AT EACH TIME POINT
# create colors for frames around tiles
plot_dat$colors = c(ifelse(p.adjust(plot_dat$p, method = "fdr") < .05,"black", "white"))
#
#
#################### CREATE PLOTS FOR TP1
# create plots, start with polygenic risk scores
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[1:29]) %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
panel1 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
# then health factors
dat1 = plot_dat %>%filter(variable %in%levels(plot_dat$variable)[1:29]) %>% filter(names %in% c("WHR","Smoking","Hypertension","Diabetic"))
panel2 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
  
# then psychological factors
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[1:29]) %>% filter(names %in% c("Depression","Neuroticism"))
panel3 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
  
#
#
#################### CREATE PLOTS FOR TP2
#
#
# create plots, start with polygenic risk scores
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[30:58]) %>% filter(names %in% c("ANX","ADHD","ASD","BIP","MDD","OCD","SCZ","AD"))
panel4 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
  
# then health factors
dat1 = plot_dat %>%filter(variable %in%levels(plot_dat$variable)[30:58]) %>% filter(names %in% c("WHR","Smoking","Hypertension","Diabetic"))
panel5 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient"))+ theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
  
# then psychological factors
dat1 = plot_dat %>% filter(variable %in%levels(plot_dat$variable)[30:58]) %>% filter(names %in% c("Depression","Neuroticism"))
panel6 = ggplot(dat1, aes(variable, names,fill = value)) +
  geom_tile(lwd = 0.75, linetype = 1, color = dat1$colors, width = 0.9, height = 0.9) +  
  geom_text(aes(label=round(value,2)),size=4)+
  scale_fill_gradient(low = "#0072B2", high = "#F0E442") +
  ylab("") + xlab("") + theme_bw() + 
  guides(fill=guide_legend(title="Standardized Beta Coefficient")) + theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))
#
#
plotS03 = ggpubr::ggarrange(panel3,panel6, panel2, panel5, panel1, panel4, ncol = 2,nrow=3, common.legend = T, legend = "bottom", align = "hv",heights=c(.35,.575,1), labels=c("d","g", "e","h", "f", "i"))
plotS03 = annotate_figure(plotS03,top = text_grob("Global and Hemispheric Metrics' Pheno- and Genotype Associations", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"Cross_Sectional_Global_Features_Phenotype_Associations.pdf",sep=""), plotS03, width = 35, height = 15)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
############################################################################ #
# Check distribution of tp1 to tp2 correlations.
############################################################################ #
##### start with dMRI
# create a list of outcome variables which we will use to loop over data frames
outcome_vars = dMRI_T1 %>% select(-c(eid, age, sex, site)) %>% names()
# empty vectors to be filled in loop
corvec1 = c()
#
for (i in 1:length(outcome_vars)){
  corvec1[i] = cor(dMRI_T1[i], dMRI_T2[i])
}
###### then for T1w data
outcome_vars = T1w_T1 %>% select(-c(eid, age, sex, site)) %>% names()
# empty vectors to be filled in loop
corvec2 = c()
#
for (i in 1:length(outcome_vars)){
  corvec2[i] = cor(T1w_T1[i], T1w_T2[i])
}
# plot it like a bad boy
part1 = data.frame(d = na.omit(corvec1), data = replicate(length(corvec1),"dMRI"))
part2 = data.frame(d = na.omit(corvec2), data = replicate(length(corvec2),"T1w"))
ds = rbind(part1,part2)
# make labels for mean effect of Feature correlations between time points
this_label = paste("Mean Pearson\\'s **r = ",round(mean((na.omit(part1$d))),2),"**", sep = "")
this_label1 = paste("Mean Pearson\\'s **r = ",round(mean((na.omit(part2$d))),2),"**", sep = "")
rm(part1,part2)
plotS02 = ds %>%
  ggplot( aes(x=d, fill=data,group=data)) +
  geom_density(aes(y=-1*..density..),alpha=0.6,
               data = ~ subset(., data %in% c("T1w")))+
  geom_density(aes(y=..density..),alpha=0.6,
               data = ~ subset(., !data %in% c("T1w")))+
  xlab("Pearson's r)")+ylab('Density') + theme_bw() + 
  theme(legend.title=element_blank()) +
  scale_fill_manual(values = c("#E69F00","#56B4E9"))+
  geom_vline(xintercept = 0, linetype="dashed", color = "black", size=.4) +
  geom_hline(yintercept = 0) + xlim(0,1.5) +
  geom_richtext(aes(x = 0.2, y = 1.5, label = this_label),
                stat = "unique", angle = 0,
                color = "black", fill = "#E69F00",
                label.color = NA, hjust = 0, vjust = 0) +
  geom_richtext(aes(x = 0.2, y = -1.5, label = this_label1),
                stat = "unique", angle = 0,
                color = "black", fill = "#56B4E9",
                label.color = NA, hjust = 0, vjust = 0)
plotS02 = annotate_figure(plotS02,top = text_grob("Distribution of Brain Features' Correlations between Time 1 and 2", color = "black", face = "bold", size = 14))
ggsave(file = paste(save_path,"TP_correlations_features.pdf",sep=""),plotS02, width = 8, height = 5)
#
#
#
#
#
#
############################################################################ #
# Check factor loadings and their distribution
############################################################################ #
#
# res.pca = ROC PC
# res.pca1 = Centercept PC
# $rotation prints the loads / contributions of the variables to the respective components
Centercept_dfs = list()
AROC_dfs = list()
for (i in 1:3){
  # first for annual rate of change components
  weight_matrix = data.frame(res.pca[[i]]$rotation)
  weight_matrix$Metric = rownames(weight_matrix)
  AROC_dfs[[i]] = weight_matrix
  # then for centercept components
  weight_matrix = data.frame(res.pca1[[i]]$rotation)
  weight_matrix$Metric = rownames(weight_matrix)
  Centercept_dfs[[i]] = weight_matrix
}
# create a list of labels to sort metrics according to the approach taken to estimate them
approaches = list(data.frame(Metric = dMRI_T2%>%select(-c(eid,sex,age,site)) %>% names(), Approach = c(replicate(dMRI_T2 %>% select(starts_with(c("dax", "drad", "micro", "v"))) %>% names() %>% length(),"BRIA"),
                                                                                                       replicate(dMRI_T2 %>% select(starts_with(c("ak_", "rk_", "mk_"))) %>% names() %>% length(), "DKI"),
                                                                                                       replicate(dMRI_T2 %>% select(starts_with(c("ad_", "rd_", "md_", "fa_"))) %>% names() %>% length(), "DTI"),
                                                                                                       #replicate(280, "RSI"),
                                                                                                       replicate(dMRI_T2 %>% select(starts_with(c("smt_mc_"))) %>% names() %>% length(), "SMT"),
                                                                                                       replicate(dMRI_T2 %>% select(starts_with(c("smt_"))) %>% select(!starts_with(c("smt_mc_"))) %>% names() %>% length(), "mcSMT"),
                                                                                                       replicate(dMRI_T2 %>% select(starts_with(c("rad", "ax", "awf"))) %>% select(!starts_with(c("smt_mc_"))) %>% names() %>% length(), "WMTI"))),
                  ## T1w approaches
                  data.frame(Metric = T1w_T2%>%select(-c(eid,sex,age,site)) %>% names(), 
                             Approach = c(replicate(T1w_T2 %>% select(ends_with(c("thickness"))) %>% names() %>% length()/2,"Thickness"), 
                                          replicate(T1w_T2 %>% select(ends_with(c("volume"))) %>% names() %>% length()/2,"Volume"),
                                          replicate(T1w_T2 %>% select(ends_with(c("area"))) %>% names() %>% length()/2,"Area"),
                                          replicate(T1w_T2 %>% select(ends_with(c("thickness"))) %>% names() %>% length()/2,"Thickness"),
                                          replicate(T1w_T2 %>% select(ends_with(c("volume"))) %>% names() %>% length()/2,"Volume"),
                                          replicate(T1w_T2 %>% select(ends_with(c("area"))) %>% names() %>% length()/2,"Area"))),
                  data.frame(Metric = multi_T2%>%select(-c(eid,sex,age,site)) %>% names(),
                             Approach = c(replicate(T1w_T2 %>% select(ends_with(c("thickness"))) %>% names() %>% length()/2,"Thickness"), 
                                          replicate(T1w_T2 %>% select(ends_with(c("volume"))) %>% names() %>% length()/2,"Volume"),
                                          replicate(T1w_T2 %>% select(ends_with(c("area"))) %>% names() %>% length()/2,"Area"),
                                          replicate(T1w_T2 %>% select(ends_with(c("thickness"))) %>% names() %>% length()/2,"Thickness"),
                                          replicate(T1w_T2 %>% select(ends_with(c("volume"))) %>% names() %>% length()/2,"Volume"),
                                          replicate(T1w_T2 %>% select(ends_with(c("area"))) %>% names() %>% length()/2,"Area"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("dax", "drad", "micro", "v"))) %>% names() %>% length(),"BRIA"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("ak_", "rk_", "mk_"))) %>% names() %>% length(), "DKI"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("ad_", "rd_", "md_", "fa_"))) %>% names() %>% length(), "DTI"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("smt_mc_"))) %>% names() %>% length(), "SMT"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("smt_"))) %>% select(!starts_with(c("smt_mc_"))) %>% names() %>% length(), "mcSMT"),
                                          replicate(dMRI_T2 %>% select(starts_with(c("rad", "ax", "awf"))) %>% select(!starts_with(c("smt_mc_"))) %>% names() %>% length(), "WMTI")))
)
# Add the approach names to the respective data frames and rescale the weights so that they range between 0 and 1 (for plotting)
for (i in 1:length(Centercept_dfs)){
  Centercept_dfs[[i]] = merge(Centercept_dfs[[i]], approaches[[i]], by = "Metric")
  Centercept_dfs[[i]]$PC1 = scales::rescale(Centercept_dfs[[i]]$PC1)
  Centercept_dfs[[i]]$PC2 = scales::rescale(Centercept_dfs[[i]]$PC2)
  AROC_dfs[[i]] = merge(AROC_dfs[[i]], approaches[[i]], by = "Metric")
  AROC_dfs[[i]]$PC1 = scales::rescale(AROC_dfs[[i]]$PC1)
  AROC_dfs[[i]]$PC2 = scales::rescale(AROC_dfs[[i]]$PC2)
}
# now, we can estimate the relative contribution of each modality / group of variables to the components
rel.weights1 = list() # centercept
rel.weights2 = list() # rate of change
for (i in 1:3){
  Comp_Cont1 = Centercept_dfs[[i]] %>% group_by(Approach) %>% summarize(IC_Weight_sum = sum(PC1)) %>% ungroup %>% data.frame
  Comp_Cont1weighted = (Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))/sum(Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))
  Comp_Cont2 = Centercept_dfs[[i]] %>% group_by(Approach) %>% summarize(IC_Weight_sum = sum(PC2)) %>% ungroup %>% data.frame
  Comp_Cont2weighted = (Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))/sum(Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))
  rel.weights1[[i]] = data.frame(PC1_rel_weight = Comp_Cont1weighted, PC2_rel_weight = Comp_Cont2weighted)
  Comp_Cont1 = AROC_dfs[[i]] %>% group_by(Approach) %>% summarize(IC_Weight_sum = sum(PC1)) %>% ungroup %>% data.frame
  Comp_Cont1weighted = (Comp_Cont1$IC_Weight_sum/table(AROC_dfs[[i]]$Approach))/sum(Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))
  Comp_Cont2 = AROC_dfs[[i]] %>% group_by(Approach) %>% summarize(IC_Weight_sum = sum(PC2)) %>% ungroup %>% data.frame
  Comp_Cont2weighted = (Comp_Cont2$IC_Weight_sum/table(AROC_dfs[[i]]$Approach))/sum(Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Approach))
  rel.weights2[[i]] = data.frame(PC1_rel_weight = Comp_Cont1weighted, PC2_rel_weight = Comp_Cont2weighted)
}
plist1 = list()
plist2 = list()
for (i in 1:3){
  dat = rel.weights1[[i]] %>% select(PC1_rel_weight.Var1, PC1_rel_weight.Freq, PC2_rel_weight.Freq)
  names(dat) = c("Modality","IC1", "IC2")
  dat = melt(dat)
  names(dat) = c("Modality","Component", "Value")
  plist1[[i]] = dat
  dat = rel.weights2[[i]] %>% select(PC1_rel_weight.Var1, PC1_rel_weight.Freq, PC2_rel_weight.Freq)
  names(dat) = c("Modality","IC1", "IC2")
  dat = melt(dat)
  names(dat) = c("Modality","Component", "Value")
  plist2[[i]] = dat
}
panel1 = ggplot(plist1[[1]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:6]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
panel2 = ggplot(plist1[[2]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[7:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
# we want to keep the same order as in plot 1 and 2
panel3 = plist1[[3]] %>%
  arrange(Value) %>%
  mutate(Modality = factor(Modality, levels= c(levels(plist1[[1]]$Modality),levels(plist1[[2]]$Modality)))) %>%
  ggplot(aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
Centercept_Plot = ggpubr::ggarrange(panel1, panel2, panel3, ncol = 3, common.legend = F, align = c("h"))
Centercept_Plot = annotate_figure(Centercept_Plot,top = text_grob("Modalities' Relative Contribution to Principal Components of the Centercepts", face = "bold", size = 14))
#
# now, do the AROC
#
panel1 = ggplot(plist2[[1]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:6]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
panel2 = ggplot(plist2[[2]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[7:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
# we want to keep the same order as in plot 1 and 2
panel3 = plist2[[3]] %>%
  arrange(Value) %>%
  mutate(Modality = factor(Modality, levels= c(levels(plist2[[1]]$Modality),levels(plist2[[2]]$Modality)))) %>%
  ggplot(aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom', legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
AROC_Plot = ggpubr::ggarrange(panel1, panel2, panel3, ncol = 3, common.legend = F, align = c("h"))
AROC_Plot = annotate_figure(AROC_Plot,top = text_grob("Modalities' Relative Contribution to Principal Components of the Annual Rates of Change", face = "bold", size = 14))
PCA_plot = ggarrange(AROC_Plot, Centercept_Plot, ncol = 1)
ggsave(file = paste(save_path,"PCA_Relative_Contributios.pdf",sep=""),PCA_plot, width = 16, height = 10)
#
#
#### Estimate the weights per feature
# order: dMRI; T1w; multimodal
# we need to add labels for each metric
#DKI
a11 = ifelse(grepl("ak",Centercept_dfs[[1]]$Metric),'AK', NA)
b11 = ifelse(grepl("rk",Centercept_dfs[[1]]$Metric),'RK', NA)
c11 = ifelse(grepl("mk",Centercept_dfs[[1]]$Metric),'MK', NA)
# DTI
d11 = ifelse(grepl("^AD_",Centercept_dfs[[1]]$Metric),'AD', NA)
e11 = ifelse(grepl("^RD_",Centercept_dfs[[1]]$Metric),'RD', NA)
f11 = ifelse(grepl("^MD_",Centercept_dfs[[1]]$Metric),'MD', NA)
g11 = ifelse(grepl("^FA_",Centercept_dfs[[1]]$Metric),'FA', NA)
# BRIA (note that Drad extra metrics were removed due to quality issues)
h11 = ifelse(grepl("v_intra",Centercept_dfs[[1]]$Metric),'V intra', NA)
i11 = ifelse(grepl("v_extra",Centercept_dfs[[1]]$Metric),'V extra', NA)
j11 = ifelse(grepl("micro_Ax",Centercept_dfs[[1]]$Metric),'micro AX', NA)
k11 = ifelse(grepl("csf",Centercept_dfs[[1]]$Metric),'V CSF', NA)
l11 = ifelse(grepl("micro_Rd",Centercept_dfs[[1]]$Metric),'micro RD', NA)
m11 = ifelse(grepl("micro_ADC",Centercept_dfs[[1]]$Metric),'micro ADC', NA)
n11 = ifelse(grepl("Dax_intra",Centercept_dfs[[1]]$Metric),'DAX intra', NA)
o11 = ifelse(grepl("Dax_extra",Centercept_dfs[[1]]$Metric),'DAX extra', NA)
p11 = ifelse(grepl("micro_FA",Centercept_dfs[[1]]$Metric),'micro FA', NA)
# WMTI (note that axEAD was excluded due to quality problems)
q11 = ifelse(grepl("awf",Centercept_dfs[[1]]$Metric),'AWF', NA)
r11 = ifelse(grepl("radEAD",Centercept_dfs[[1]]$Metric),'radEAD', NA)
s11 = ifelse(is.na(a11) == T, b11, a11)
# SMT
t11 = ifelse(grepl("smt_fa",Centercept_dfs[[1]]$Metric),'FA', NA)
u11 = ifelse(grepl("smt_md",Centercept_dfs[[1]]$Metric),'MD', NA)
v11 = ifelse(grepl("smt_trans_",Centercept_dfs[[1]]$Metric),'Transverse', NA)
w11 = ifelse(grepl("smt_long",Centercept_dfs[[1]]$Metric),'Longitudinal', NA)
x11 = ifelse(is.na(a11) == T, b11, a11)
y11 = ifelse(is.na(a11) == T, c11, a11)
z11 = ifelse(is.na(a11) == T, d11, a11)
# mcSMT
a111 = ifelse(grepl("mc_diff",Centercept_dfs[[1]]$Metric),'Diffusion Coefficient', NA)
b111 = ifelse(grepl("extramd",Centercept_dfs[[1]]$Metric),'Extra-Axonal MD', NA)
c111 = ifelse(grepl("extratra",Centercept_dfs[[1]]$Metric),'Extra-Axonal Transverse', NA)
d111 = ifelse(grepl("mc_intra",Centercept_dfs[[1]]$Metric),'Intra Neurite Volume', NA)
# 
# merge
a11 = ifelse(is.na(a11) == T, b11, a11)
a11 = ifelse(is.na(a11) == T, c11, a11)
a11 = ifelse(is.na(a11) == T, d11, a11)
a11 = ifelse(is.na(a11) == T, e11, a11)
a11 = ifelse(is.na(a11) == T, f11, a11)
a11 = ifelse(is.na(a11) == T, g11, a11)
a11 = ifelse(is.na(a11) == T, h11, a11)
a11 = ifelse(is.na(a11) == T, i11, a11)
a11 = ifelse(is.na(a11) == T, j11, a11)
a11 = ifelse(is.na(a11) == T, k11, a11)
a11 = ifelse(is.na(a11) == T, l11, a11)
a11 = ifelse(is.na(a11) == T, m11, a11)
a11 = ifelse(is.na(a11) == T, n11, a11)
a11 = ifelse(is.na(a11) == T, o11, a11)
a11 = ifelse(is.na(a11) == T, p11, a11)
a11 = ifelse(is.na(a11) == T, q11, a11)
a11 = ifelse(is.na(a11) == T, r11, a11)
a11 = ifelse(is.na(a11) == T, s11, a11)
a11 = ifelse(is.na(a11) == T, t11, a11)
a11 = ifelse(is.na(a11) == T, u11, a11)
a11 = ifelse(is.na(a11) == T, v11, a11)
a11 = ifelse(is.na(a11) == T, w11, a11)
a11 = ifelse(is.na(a11) == T, x11, a11)
a11 = ifelse(is.na(a11) == T, y11, a11)
a11 = ifelse(is.na(a11) == T, z11, a11)
a11 = ifelse(is.na(a11) == T, a111, a11)
a11 = ifelse(is.na(a11) == T, b111, a11)
a11 = ifelse(is.na(a11) == T, c111, a11)
a11 = ifelse(is.na(a11) == T, d111, a11)
#Centercept_dfs[[1]] %>% filter(is.na(Group)) %>% select(Metric, Approach)
Centercept_dfs[[1]]$Group = a11
AROC_dfs[[1]]$Group = a11
Centercept_dfs[[2]]$Group = Centercept_dfs[[2]]$Approach
AROC_dfs[[2]]$Group = AROC_dfs[[2]]$Approach
Centercept_dfs[[3]]$Group = c(Centercept_dfs[[1]]$Group, Centercept_dfs[[2]]$Approach)
AROC_dfs[[3]]$Group = c(AROC_dfs[[1]]$Group, AROC_dfs[[2]]$Approach)
# now, we can estimate the relative contribution of each group of variables to the components
rel.weights1 = list() # centercept
rel.weights2 = list() # rate of change
for (i in 1:3){
  Comp_Cont1 = Centercept_dfs[[i]] %>% group_by(Group) %>% summarize(IC_Weight_sum = sum(PC1)) %>% ungroup %>% data.frame
  Comp_Cont1weighted = (Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))/sum(Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))
  Comp_Cont2 = Centercept_dfs[[i]] %>% group_by(Group) %>% summarize(IC_Weight_sum = sum(PC2)) %>% ungroup %>% data.frame
  Comp_Cont2weighted = (Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))/sum(Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))
  rel.weights1[[i]] = data.frame(PC1_rel_weight = Comp_Cont1weighted, PC2_rel_weight = Comp_Cont2weighted)
  Comp_Cont1 = AROC_dfs[[i]] %>% group_by(Group) %>% summarize(IC_Weight_sum = sum(PC1)) %>% ungroup %>% data.frame
  Comp_Cont1weighted = (Comp_Cont1$IC_Weight_sum/table(AROC_dfs[[i]]$Group))/sum(Comp_Cont1$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))
  Comp_Cont2 = AROC_dfs[[i]] %>% group_by(Group) %>% summarize(IC_Weight_sum = sum(PC2)) %>% ungroup %>% data.frame
  Comp_Cont2weighted = (Comp_Cont2$IC_Weight_sum/table(AROC_dfs[[i]]$Group))/sum(Comp_Cont2$IC_Weight_sum/table(Centercept_dfs[[i]]$Group))
  rel.weights2[[i]] = data.frame(PC1_rel_weight = Comp_Cont1weighted, PC2_rel_weight = Comp_Cont2weighted)
}
plist1 = list()
plist2 = list()
for (i in 1:3){
  dat = rel.weights1[[i]] %>% select(PC1_rel_weight.Var1, PC1_rel_weight.Freq, PC2_rel_weight.Freq)
  names(dat) = c("Modality","IC1", "IC2")
  dat = melt(dat)
  names(dat) = c("Modality","Component", "Value")
  plist1[[i]] = dat
  dat = rel.weights2[[i]] %>% select(PC1_rel_weight.Var1, PC1_rel_weight.Freq, PC2_rel_weight.Freq)
  names(dat) = c("Modality","IC1", "IC2")
  dat = melt(dat)
  names(dat) = c("Modality","Component", "Value")
  plist2[[i]] = dat
}
# manualcolors<-c('black','forestgreen', 'red2', 'orange', 'cornflowerblue', 
#                 'magenta', 'darkolivegreen4', 'indianred1', 'tan4', 'darkblue', 
#                 'mediumorchid1','firebrick4',  'yellowgreen', 'lightsalmon', 'tan3',
#                 "tan1",'darkgray', 'wheat4', '#DDAD4B', 'chartreuse', 
#                 'seagreen1', 'moccasin', 'mediumvioletred', 'seagreen','cadetblue1',
#                 "darkolivegreen1")
# manualcolors2 = c("tan2" ,   "tomato3" , "#7CE3D8")
# manualcolors3 = c('black','forestgreen', 'red2', 'orange', 'cornflowerblue', 
#                 'magenta', 'darkolivegreen4', 'indianred1', 'tan4', 'darkblue', 
#                 'mediumorchid1','firebrick4',  'yellowgreen', 'lightsalmon', 'tan3',
#                 "tan1",'darkgray', 'wheat4', '#DDAD4B', 'chartreuse', 
#                 'seagreen1', 'moccasin', 'mediumvioletred', 'seagreen','cadetblue1',
#                 "darkolivegreen1" ,"tan2" ,   "tomato3" , "#7CE3D8")
#
# centercept plots
panel1 = ggplot(plist1[[1]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_colour_manual(values=manualcolors) +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:24]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
panel2 = ggplot(plist1[[2]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_colour_manual(values=manualcolors[]) +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:24]) +   
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
# we want to keep the same order as in plot 1 and 2
panel3 = plist1[[3]] %>%
  arrange(Value) %>%
  #mutate(Modality = factor(Modality, levels= c(levels(plist1[[1]]$Modality),levels(plist1[[2]]$Modality)))) %>%
  ggplot(aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_colour_manual(values=manualcolors) +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:24]) +   
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
Centercept_Plot = ggpubr::ggarrange(panel1, panel2, panel3, ncol = 3, common.legend = F, align = c("h"))
Centercept_Plot = annotate_figure(Centercept_Plot,top = text_grob("Modalities' Relative Contribution to Principal Components of the Centercepts", face = "bold", size = 14))
#
# aroc plots
panel1 = ggplot(plist2[[1]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:6]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
panel2 = ggplot(plist2[[2]], aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[7:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom',legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
# we want to keep the same order as in plot 1 and 2
panel3 = #plist2[[3]] %>%
  #arrange(Value) %>%
  #mutate(Modality = factor(Modality, levels= c(levels(plist2[[1]]$Modality),levels(plist2[[2]]$Modality)))) %>%
  ggplot(plist2[[3]], aes(x = Component, y = Value, fill = Modality)) +
  #ggplot(aes(x = Component, y = Value, fill = Modality)) +
  geom_col(colour = "black") +
  #scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:9]) + 
  ylab("Relative Contribution") + theme_bw() +
  theme(legend.position='bottom', legend.title = element_blank()) + xlab("") +
  scale_x_discrete(labels=c("IC1" = "Principal Component 1", "IC2" = "Principal Component  2"))
AROC_Plot = ggpubr::ggarrange(panel1, panel2, panel3, ncol = 3, common.legend = F, align = c("h"))
AROC_Plot = annotate_figure(AROC_Plot,top = text_grob("Modalities' Relative Contribution to Principal Components of the Annual Rates of Change", face = "bold", size = 14))
PCA_plot = ggarrange(AROC_Plot, Centercept_Plot, ncol = 1)
ggsave(file = paste(save_path,"PCA_REGIONAL_Relative_Contributios.pdf",sep=""),PCA_plot, width = 22, height = 10)
#
#
#
#
#
## Finally, we make a plot which represents the associations between BAG and changes in features by feature type
# we already have labels for the different features
# exp1-3 are the feature change & CCc BAG associations (dMRI, T1w, multi)
# exp4-6 are the feature change & ARoC BAG associations
# make column for row names
exp1$Metric = rownames(exp1)
exp2$Metric = rownames(exp2)
exp3$Metric = rownames(exp3)
exp4$Metric = rownames(exp4)
exp5$Metric = rownames(exp5)
exp6$Metric = rownames(exp6)
# merge the data frames to have metrics grouped (for each approach)
# dMRI
a01 = Centercept_dfs[[1]] %>% select(Metric,Group)
ex1 = merge(exp1, a01, by = "Metric")
ex4 = merge(exp4, a01, by = "Metric")
# T1w
a01 = Centercept_dfs[[2]] %>% select(Metric,Group)
ex2 = merge(exp2, a01, by = "Metric")
ex5 = merge(exp5, a01, by = "Metric")
# multi
a01 = Centercept_dfs[[3]] %>% select(Metric,Group)
ex3 = merge(exp3, a01, by = "Metric")
ex6 = merge(exp6, a01, by = "Metric")
# create a plot function
plot_func3 = function(data, label){
  ggplot(data, aes(betas , fill=Group, Group=Group, shape=Group, facets=Group)) + 
    geom_density()+ facet_wrap(.~Group) + theme_bw() + 
    xlab("Association (standardized beta)")+ylab('Density')+ 
    #scale_fill_brewer(palette = "Pastel1") + 
    xlim(-1,1)  + ylim(0,25)  +
    geom_vline(xintercept = 0, colour = "black", lty = 2)
}
#T1_plot_df = rbind(Vol,Area, Thick)
a = plot_func3(ex1)
b = plot_func3(ex2)
d = plot_func3(ex3)
e = plot_func3(ex4)
f = plot_func3(ex5)
g = plot_func3(ex6)
#Metric_Fig = ggarrange(a,b,d,e,f,g, labels = c("a","b","c","d","e","f"), ncol = 2)
ggsave(file = paste(save_path,"SupplementalFigure8.pdf",sep=""),a, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure9.pdf",sep=""),b, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure10.pdf",sep=""),d, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure11.pdf",sep=""),e, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure12.pdf",sep=""),f, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure13.pdf",sep=""),g, height = 10, width = 10)
#
# estimate mean effects
ex1 = ex1 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex2 = ex2 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex3 = ex3 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex4 = ex4 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex5 = ex5 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex6 = ex6 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
# make two forrest plot like error bar charts (first ARoC, second CC)
exf = rbind(ex1,ex2,ex3)
exf$Modality = c(replicate(24,"dMRI"),replicate(3,"T1w"),replicate(nrow(exf)-27,"multimodal"))
exf2 = rbind(ex4,ex5,ex6)
exf2$Modality = exf$Modality
#
# plot
CC_avg = ggplot(exf, aes(y = Group, x = M, xmin = M - SD, xmax = M + SD)) + 
  geom_vline(xintercept = 0, colour = "black", lty = 2) +
  geom_point() + 
  geom_errorbarh(height = 0) + 
  facet_grid(. ~ Modality, scales = "free")+
  xlab("Average associations between the Annual Rates of BAG and Feature Changes (Mean±SD)") + ylab("")
ARoC_avg = ggplot(exf2, aes(y = Group, x = M, xmin = M - SD, xmax = M + SD)) + 
  geom_vline(xintercept = 0, colour = "black", lty = 2) +
  geom_point() + 
  geom_errorbarh(height = 0) + 
  facet_grid(. ~ Modality, scales = "free")+
  xlab("Average associations between BAG Centercepts and the Annual Rate of Change per Feature (Mean±SD)") + ylab("")
ggsave(file = paste(save_path,"SupplementalFigure14.pdf",sep=""),CC_avg, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure15.pdf",sep=""),ARoC_avg, height = 10, width = 10)
#
#
# Finally, predict feature change in dMRI data from T1w brain age, and in T1w data from dMRI brain age
########## dMRI data
tmp = cbind(AROC[[1]], ISI = BAG$ISI, CCc = BAG$CCc_T1w)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[1]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ CCc + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp1 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
mean(abs(exp1$betas))
########## T1w data
tmp = cbind(AROC[[2]], ISI = BAG$ISI, CCc = BAG$CCc_dMRI)
tmp = scale(tmp)
tmp2 = BAGdf[[2]]
tmp2[2:6] = scale(tmp2[2:6])
tmp = cbind(tmp, tmp2)
tmp = data.frame(tmp)
brain_features = names(AROC[[2]])
betas = c()
ps = betas
SE = betas
for (o in brain_features){
  f = formula(paste(o," ~ CCc + ISI + age*sex + site", sep = ""))
  tmp_model = lm(f, data = tmp)
  betas[o] = summary(tmp_model)$coefficients[2]
  SE[o] = summary(tmp_model)$coefficients[2,2]
  ps[o] = summary(tmp_model)$coefficients[2,4]
}
data.frame(betas, ps = p.adjust(ps, method = "fdr")) %>% filter(ps < .05) %>% nrow()/length(betas)
exp2 = data.frame(betas, SE, p = ps, p.adj = p.adjust(ps, method = "fdr"))
mean(abs(exp2$betas))
#
# now check feature group level associations
## first in terms of density
exp1$Metric = rownames(exp1)
exp2$Metric = rownames(exp2)
a01 = Centercept_dfs[[1]] %>% select(Metric,Group)
ex1 = merge(exp1, a01, by = "Metric")
a01 = Centercept_dfs[[2]] %>% select(Metric,Group)
ex2 = merge(exp2, a01, by = "Metric")
a = plot_func3(ex1)
b = plot_func3(ex2)
ggsave(file = paste(save_path,"SupplementalFigure16.pdf",sep=""),a, height = 10, width = 10)
ggsave(file = paste(save_path,"SupplementalFigure17.pdf",sep=""),b, height = 10, width = 10)
#
## then as bar plot
ex1 = ex1 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
ex2 = ex2 %>% group_by(Group) %>% summarize(M = mean(betas), SD = sd(betas)) %>% data.frame
exf = rbind(ex1,ex2)
exf$Modality = c(replicate(24,"dMRI"),replicate(3,"T1w"))
#
# plot
CC_avg = ggplot(exf, aes(y = Group, x = M, xmin = M - SD, xmax = M + SD)) + 
  geom_vline(xintercept = 0, colour = "black", lty = 2) +
  geom_point() + 
  geom_errorbarh(height = 0) + 
  facet_grid(. ~ Modality, scales = "free")+
  xlab("Average associations between BAG Centercepts and the Annual Rate of Change per Feature (Mean±SD)") + ylab("")
ggsave(file = paste(save_path,"SupplementalFigure18.pdf",sep=""),CC_avg, height = 10, width = 10)
#
#
#
print("The End.")


