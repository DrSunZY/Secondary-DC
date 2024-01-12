setwd("~/Documents/2-Work/R/DC_MachineLearning")

library(readxl)
library(mlr3)
library(mlr3verse)
library(data.table)
library(e1071)
library(mlr3filters)
set.seed(777)

# input data
load("temp_clinical_z.RData")
load("temp_splits.RData")
load("temp_data_before_z.RData")
load("temp_data_after_z.RData")



#####
splits$train 
splits$test


###############################################################################
# combination of post_ct and clinical data
comb_2 = cbind(clinical_z, data_after_z[,-"outcome"])

# initiate the new task with training data
task_comb_2 = as_task_classif(comb_2[splits$train,], target = "outcome")

## feature selection
set.seed(77)
lrn_ranger = lrn("classif.ranger", importance = "impurity")

flt_importance = flt("importance", learner = lrn_ranger)
flt_importance$calculate(task_comb_2)

output_importance_comb_2 = as.data.table(flt_importance)
print(output_importance_comb_2)
################################################################################


load("temp_importance_1.RData")

imp_1 = output_importance_1[1:63,]
imp_c = output_importance_comb_2[1:63,]
# match the two datasets with the same gene name
feature_check_63 = intersect(imp_c$feature, imp_1$feature)
feature_check_63









