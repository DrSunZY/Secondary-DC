setwd("~/Documents/2-Work/R/DC_MachineLearning")

rm(list = ls())

library(readxl)
library(mlr3)
library(mlr3verse)
library(data.table)
library(e1071)
library(mlr3filters)
set.seed(777)

# input data
data_after_temp = read_excel("data_post_evacuation.xlsx")
str(data_after_temp)
sapply(data_after_temp, class)
chars = sapply(data_after_temp, is.character)

data_after_temp[ ,chars] = as.data.frame(apply(data_after_temp[ , chars], 2, as.numeric))
sapply(data_after_temp, class)

data_after = data.table(data_after_temp)

data_after$outcome = as.factor(data_after$outcome)
str(data_after)
rm(data_after_temp)

# convert all data into z-score, except the $outcome
data_after_z = scale(data_after[,-1])
data_after_z = data.table(data_after_z)
data_after_z$outcome = data_after$outcome
str(data_after_z)

rm(data_after_temp)
rm(data_after)


save(data_after_z, file = "temp_data_after_z.RData")
#####
load("temp_splits.RData")

splits$train 
splits$test 

# initiate the new task with training data
task_real = as_task_classif(data_after_z[splits$train,], target = "outcome")


## feature selection
set.seed(77)
lrn_ranger = lrn("classif.ranger", importance = "impurity")

flt_importance = flt("importance", learner = lrn_ranger)
flt_importance$calculate(task_real)

output_importance_1 = as.data.table(flt_importance)
print(output_importance_1)

save(output_importance_1, file = "temp_importance_1.RData")
###################
#### learners ####
learners = lrns(c("classif.randomForest", 
                  "classif.svm", 
                  "classif.kknn", 
                  "classif.cforest", 
                  "classif.ctree", 
                  "classif.gbm"), 
                predict_type="prob")



keep = names(head(flt_importance$scores, 14)) # starting from 20, and all the way to 5
print(length(keep))
task_real = as_task_classif(data_after_z[splits$train,], target = "outcome")


{
  task_real$select(keep)
  print(task_real$feature_names)
  set.seed(77)
  grid= benchmark_grid(
    task= task_real,
    learners= learners,
    resamplings = rsmp("loo") 
  )
  
  bmr = benchmark(grid, store_models = TRUE)
  bmr
  measures = msrs(c("classif.acc",
                    "classif.auc"))
  print(bmr$aggregate(measures))
  
  #############################################
  result_train_acc = vector("double", 6)
  result_train_auc = vector("double", 6)
  
  result_test_acc = vector("double", 6)
  result_test_auc = vector("double", 6)
  
  #  prediction of new data for each model
  for (resultno in 1:6) {
    
    output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
    print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
    result_train_acc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
    result_train_auc[resultno] = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
    for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
      prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(data_after_z[splits$test,], task_real)
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
# ROC analysis for randomForest with top 14 features
keep = names(head(flt_importance$scores, 14)) 
print(length(keep))
task_real = as_task_classif(data_after_z[splits$train,], target = "outcome")



task_real$select(keep)
print(task_real$feature_names)
set.seed(77)
grid= benchmark_grid(
  task= task_real,
  learners= learners,
  resamplings = rsmp("loo") 
)

bmr = benchmark(grid, store_models = TRUE)
bmr
measures = msrs(c("classif.acc",
                  "classif.auc"))
print(bmr$aggregate(measures))


resultno = 1
output_acc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
output_auc = vector("double", length(bmr$resample_results$resample_result[[resultno]]$learners))
print(bmr$resample_results$resample_result[[resultno]]$learners[[1]])
r14_result_train_acc= bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.acc"))
r14_result_train_auc = bmr$resample_results$resample_result[[resultno]]$prediction()$score(msr("classif.auc"))
for (i in seq_along(bmr$resample_results$resample_result[[resultno]]$learners)) {
  prediction = bmr$resample_results$resample_result[[resultno]]$learners[[i]]$predict_newdata(data_after_z[splits$test,], task_real)
  output_acc[i] = prediction$score(msr("classif.acc"))
  output_auc[i] = prediction$score(msr("classif.auc"))
}
r14_result_test_acc = mean(output_acc)
r14_result_test_auc = mean(output_auc)

print(output_auc)

prediction_roc = bmr$resample_results$resample_result[[resultno]]$learners[[27]]$predict_newdata(data_after_z[splits$test,], task_real)

autoplot(prediction_roc, type = "roc")+
  geom_line(linewidth = 2)+
  theme(text = element_text(size = 16))
