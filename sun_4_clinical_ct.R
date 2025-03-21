setwd("~/Desktop/DC_Machinelearning/R")


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




#####
splits$train # for level_1 model construction and validation
splits$test # for level_2 test


##########################
# combination of pre_ct and clinical data
comb_1 = cbind(clinical_z, data_before_z[,-"outcome"])

# initiate the new task with training data
task_comb_1 = as_task_classif(comb_1[splits$train,], target = "outcome")

## feature selection
set.seed(77)
lrn_ranger = lrn("classif.ranger", importance = "impurity")

flt_importance = flt("importance", learner = lrn_ranger)
flt_importance$calculate(task_comb_1)

output_importance_comb_1 = as.data.table(flt_importance)
print(output_importance_comb_1)

###############################################################################
#### learners ###############################################################
learners = lrns(c("classif.randomForest", 
                  "classif.kknn", 
                  "classif.cforest", 
                  "classif.ctree", 
                  "classif.gbm", 
                  "classif.rpart"), 
                predict_type="prob")



keep = names(head(flt_importance$scores, 8)) 
print(length(keep))
task_real = as_task_classif(comb_1[splits$train,], target = "outcome")

{
  task_real$select(keep)
  print(task_real$feature_names)
  set.seed(77)
  grid= benchmark_grid(
    task= task_real,
    learners= learners,
    resamplings = rsmp("loo") # leave one out method
  )
  
  bmr = benchmark(grid, store_models = TRUE)
  bmr
  measures = msrs(c("classif.acc",
                    "classif.auc"))
  print(bmr$aggregate(measures))
  
  #############################################
  result_train_acc = vector("double", 4)
  result_train_auc = vector("double", 4)
  
  result_test_acc = vector("double", 4)
  result_test_auc = vector("double", 4)
  
  #  prediction of new data for each model
  for (resultno in 1:6) {
    
    output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
    result_train_acc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
    result_train_auc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
    for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
      prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(comb_1[splits$test,], task_real)
      output_acc[i] = prediction$score(msr("classif.acc"))
      output_auc[i] = prediction$score(msr("classif.auc"))
    }
    result_test_acc[resultno] = mean(output_acc)
    result_test_auc[resultno] = mean(output_auc)
    
  }
  
  print(result_train_acc)
  print(result_test_acc)
  
  print(result_train_auc)
  print(result_test_auc)
  
}


################################################################################
#####################
# ROC analysis for randomForest with top 9 features
# keep = names(head(flt_importance$scores, 9)) 
# print(length(keep))
# task_real = as_task_classif(comb_1[splits$train,], target = "outcome")
# 
# task_real$select(keep)
# print(task_real$feature_names)
# set.seed(77)
# grid= benchmark_grid(
#   task= task_real,
#   learners= learners,
#   resamplings = rsmp("loo") # leave one out method
# )
# 
# bmr = benchmark(grid, store_models = TRUE)
# bmr
# measures = msrs(c("classif.acc",
#                   "classif.auc"))
# print(bmr$aggregate(measures))


#######################
# result for randomForest with 9 features

resultno = 3
output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
r9_result_train_acc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
r9_result_train_auc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
  prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(comb_1[splits$test,], task_real)
  output_acc[i] = prediction$score(msr("classif.acc"))
  output_auc[i] = prediction$score(msr("classif.auc"))
}
r9_result_test_acc = mean(output_acc)
r9_result_test_auc = mean(output_auc)

print(output_auc)# i=2 shows the auc = 0.833

prediction_roc = bmr$resample_results$resample_result[[resultno]]$learners[[39]]$predict_newdata(comb_1[splits$test,], task_real)
prediction_roc$score(msr("classif.sensitivity"))
prediction_roc$score(msr("classif.specificity"))


library(ggplot2)

autoplot(prediction_roc, type = "roc")+
  geom_line(linewidth = 2)+
  theme(text = element_text(size = 16))


prediction_roc$score(msr("classif.sensitivity"))
prediction_roc$score(msr("classif.specificity"))
autoplot


#######################
# result for gbm with 9 features

resultno = 5
output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
r9_result_train_acc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
r9_result_train_auc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
  prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(comb_1[splits$test,], task_real)
  output_acc[i] = prediction$score(msr("classif.acc"))
  output_auc[i] = prediction$score(msr("classif.auc"))
}
r9_result_test_acc = mean(output_acc)
r9_result_test_auc = mean(output_auc)

print(output_auc)# i=2 shows the auc = 0.833

prediction_roc = bmr$resample_results$resample_result[[resultno]]$learners[[1]]$predict_newdata(comb_1[splits$test,], task_real)



autoplot(prediction_roc, type = "roc")+
  geom_line(linewidth = 2)+
  theme(text = element_text(size = 16))


prediction_roc$score(msr("classif.sensitivity"))
prediction_roc$score(msr("classif.specificity"))
autoplot


################################################################################


load("temp_importance_1.RData")

imp_1 = output_importance_1[1:30,]
imp_c = output_importance_comb_2[1:30,]
# match the two datasets with the same gene name
feature_check_30 = intersect(imp_c$feature, imp_1$feature)
feature_check_30


imp_1 = output_importance_1[1:63,]
imp_c = output_importance_comb_2[1:63,]
# match the two datasets with the same gene name
feature_check_63 = intersect(imp_c$feature, imp_1$feature)
feature_check_63









