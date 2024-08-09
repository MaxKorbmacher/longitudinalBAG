Code, trained models, figures, supplemental data (full tables), all figures and tables for the paper "Disentangeling Neurodegeneration: The Brain Age Gap describes Healthy Ageing", Korbmacher, M. et. al. (2024).

**Code:** <br/>
Sample_Size_Estimation.R estimation of power / necessary sample size assuming varying parameter shrinkages
nested_CV_example.py example of nested CV procedure applied when training Lasso and XGB models
Lasso_retrain.py code to re-fit Lasso models and to obtain predictions
CV_predict.py predictions using XGB models
TimeIsolatedBrainAge.R contains the analyses of brain ages predicted at each time point doing a 50-50 split and predicting within and across time points.

**Models:**
XGB_T1w_model.txt, XGB_dMRI_model.txt, XGB_multimodal_model.txt; Lasso_T1_model.pkl, Lasso_dMRI_model.pkl, Lasso_multi_model.pkl

**Supplement:**
Supplementary data files can be found in SupplData1.zip & SupplData2.zip.
Supplementary figures: SupplFigures.zip
Supplementary tables: SupplTables.zip
