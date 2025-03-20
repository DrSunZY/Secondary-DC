setwd("~/Desktop/DC_Machinelearning/R")

library(readxl)
library(mlr3)
library(mlr3verse)
library(data.table)
library(e1071)
library(mlr3filters)
set.seed(777)

# input data
data_before_temp = read_excel("data_before_copy.xlsx")
str(data_before_temp)
sapply(data_before_temp, class)
chars = sapply(data_before_temp, is.character)

data_before_temp[ ,chars] = as.data.frame(apply(data_before_temp[ , chars], 2, as.numeric))
sapply(data_before_temp, class)

data_before = data.table(data_before_temp)

data_before$outcome = as.factor(data_before$outcome)
str(data_before)


# convert all data into z-score, except the $outcome
data_before_z = scale(data_before[,-1])
data_before_z = data.table(data_before_z)
data_before_z$outcome = data_before$outcome
str(data_before_z)

rm(data_before_temp)
rm(data_before)


save(data_before_z, file = "temp_data_before_z.RData")
#####
load("temp_splits.RData")

splits$train # for level_1 model construction and validation
splits$test # for level_2 test

# initiate the new task with training data
task_real = as_task_classif(data_before_z[splits$train,], target = "outcome")


## feature selection
set.seed(77)
lrn_ranger = lrn("classif.ranger", importance = "impurity")

flt_importance = flt("importance", learner = lrn_ranger)
flt_importance$calculate(task_real)

output_importance_2 = as.data.table(flt_importance)
print(output_importance_2)

###################
#### learners ####
learners = lrns(c("classif.randomForest", 
                  "classif.svm", 
                  "classif.kknn", 
                  "classif.cforest", 
                  "classif.ctree", 
                  "classif.gbm", 
                  "classif.rpart"), 
                predict_type="prob")


# keep = names(which(flt_importance$scores >= 0.05))
keep = names(head(flt_importance$scores, 9)) # starting from 20, and all the way to 7
print(length(keep))
task_real = as_task_classif(data_before_z[splits$train,], target = "outcome")


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
  result_train_acc = vector("double", 7)
  result_train_auc = vector("double", 7)
  
  result_test_acc = vector("double", 7)
  result_test_auc = vector("double", 7)
  
  #  prediction of new data for each model
  for (resultno in 1:7) {
    
    output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
    result_train_acc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
    result_train_auc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
    for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
      prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(data_before_z[splits$test,], task_real)
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

  

#####################
# ROC analysis for randomForest with top 9 features
# keep = names(head(flt_importance$scores, 9)) 
# print(length(keep))
# task_real = as_task_classif(data_before_z[splits$train,], target = "outcome")
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

resultno = 1
output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
r9_result_train_acc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
r9_result_train_auc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
  prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(data_before_z[splits$test,], task_real)
  output_acc[i] = prediction$score(msr("classif.acc"))
  output_auc[i] = prediction$score(msr("classif.auc"))
}
r9_result_test_acc = mean(output_acc)
r9_result_test_auc = mean(output_auc)

print(output_auc)# i=2 shows the auc = 0.833

prediction_roc = bmr$resample_results$resample_result[[resultno]]$learners[[2]]$predict_newdata(data_before_z[splits$test,], task_real)

autoplot(prediction_roc, type = "roc")+
  geom_line(linewidth = 2)+
  theme(text = element_text(size = 16))
  

prediction_roc$score(msr("classif.sensitivity"))
prediction_roc$score(msr("classif.specificity"))
autopl

