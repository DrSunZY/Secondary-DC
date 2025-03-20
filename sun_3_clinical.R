setwd("~/Desktop/DC_Machinelearning/R")

library(readxl)
library(mlr3)
library(mlr3verse)
library(data.table)
library(e1071)
library(mlr3filters)
set.seed(777)

# input data
clinical = read_excel("patients_65.xlsx")
str(clinical)
sapply(clinical, class)

chars = c(2,4,6,8,12,14,30)
numb = c(1,3,5,7,9,10,11,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31)


{
  clinical$SEX = as.factor(clinical$SEX)
  clinical$ADMLPRFX = as.factor(clinical$ADMLPRFX)
  clinical$ADMRPRFX = as.factor(clinical$ADMRPRFX)
  clinical$POLPRFX = as.factor(clinical$POLPRFX)
  clinical$PORPRFX = as.factor(clinical$PORPRFX)
  clinical$outcome = as.factor(clinical$outcome)
  clinical$MOI = as.factor(clinical$MOI)
}

sapply(clinical, class)

clinical_factor = clinical[,chars]


clinical_num = scale(clinical[,numb])
clinical_num = data.table(clinical_num)

clinical_z = cbind(clinical_factor, clinical_num) 

clinical_z = clinical_z[,-8] # remove the column of "No."
clinical_z = clinical_z[,-30] # remove the discharge.GCS which is not used for model construction

str(clinical_z)

save(clinical_z, file = "temp_clinical_z.RData")
##################################################


#####
load("temp_splits.RData")

splits$train # for level_1 model construction and validation
splits$test # for level_2 test

# initiate the new task with training data
task_real = as_task_classif(clinical_z[splits$train,], target = "outcome")


## feature selection
set.seed(77)
lrn_ranger = lrn("classif.ranger", importance = "impurity")

flt_importance = flt("importance", learner = lrn_ranger)
flt_importance$calculate(task_real)

output_importance_clinical = as.data.table(flt_importance)
print(output_importance_clinical)

#### learners ###############################################################
learners = lrns(c("classif.randomForest", 
                  "classif.kknn", 
                  "classif.ctree", 
                  "classif.gbm"), 
                predict_type="prob") # svm does not support factor type



keep = names(head(flt_importance$scores, 28)) 
print(length(keep))
task_real = as_task_classif(clinical_z[splits$train,], target = "outcome")

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
  for (resultno in 1:4) {
    
    output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
    result_train_acc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
    result_train_auc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
    for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
      prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(clinical_z[splits$test,], task_real)
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















