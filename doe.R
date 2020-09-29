require(data.table)
require(AlgDesign)

# create parameter combinations to test
doe = data.table(expand.grid(nrounds = c(50, 250),
                             eta = c(0.015, 0.025, 0.05, 0.1),
                             max_depth = c(5, 10, 20), 
                             min_child_weight = c(1, 5, 11),
                             subsample = c(0.7, 1),
                             colsample_bytree = c(0.7, 1),
                             gamma = c(0, 0.01, 0.5)))

# compute the number of levels for each variable in our design
levels_design = sapply(1:ncol(doe), function(j) length(table(doe[, j, with = FALSE])))

# build the general factorial design
doe_gen = gen.factorial(levels_design)

# compute a smaller optimal design
set.seed(42)
doe_opt = optFederov(data = doe_gen, 
                     nTrials = 16)

# update which rows to keep in doe according to doe_opt
doe = doe[doe_opt$rows]

# export design
write.csv(doe, "doe_xgboost.csv", row.names = FALSE)