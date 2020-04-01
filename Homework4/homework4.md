Homework 4
================
Anthony DeNiro
March 31, 2020

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(ElemStatLearn)
```

    ## Warning: package 'ElemStatLearn' was built under R version 3.6.2

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.6.2

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

``` r
data("vowel.train")
data("vowel.test")
```

``` r
head(vowel.train)
```

    ##   y    x.1   x.2    x.3   x.4    x.5   x.6    x.7    x.8    x.9   x.10
    ## 1 1 -3.639 0.418 -0.670 1.779 -0.168 1.627 -0.388  0.529 -0.874 -0.814
    ## 2 2 -3.327 0.496 -0.694 1.365 -0.265 1.933 -0.363  0.510 -0.621 -0.488
    ## 3 3 -2.120 0.894 -1.576 0.147 -0.707 1.559 -0.579  0.676 -0.809 -0.049
    ## 4 4 -2.287 1.809 -1.498 1.012 -1.053 1.060 -0.567  0.235 -0.091 -0.795
    ## 5 5 -2.598 1.938 -0.846 1.062 -1.633 0.764  0.394 -0.150  0.277 -0.396
    ## 6 6 -2.852 1.914 -0.755 0.825 -1.588 0.855  0.217 -0.246  0.238 -0.365

1. change response variable to a factor
=======================================

``` r
vowel.train$y <- as.factor(vowel.train$y)
```

2. Random Forest Documentation
==============================

``` r
#?randomForest()
```

3. Fit model with default parameters
====================================

``` r
rf1 <- randomForest(vowel.train[2:11], vowel.train$y)

rf1
```

    ## 
    ## Call:
    ##  randomForest(x = vowel.train[2:11], y = vowel.train$y) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 2.84%
    ## Confusion matrix:
    ##     1  2  3  4  5  6  7  8  9 10 11 class.error
    ## 1  48  0  0  0  0  0  0  0  0  0  0  0.00000000
    ## 2   0 48  0  0  0  0  0  0  0  0  0  0.00000000
    ## 3   0  0 48  0  0  0  0  0  0  0  0  0.00000000
    ## 4   0  0  0 47  0  1  0  0  0  0  0  0.02083333
    ## 5   0  0  0  0 46  1  0  0  0  0  1  0.04166667
    ## 6   0  0  0  0  0 42  0  0  0  0  6  0.12500000
    ## 7   0  0  0  0  1  0 45  2  0  0  0  0.06250000
    ## 8   0  0  0  0  0  0  0 48  0  0  0  0.00000000
    ## 9   0  0  0  0  0  0  1  0 47  0  0  0.02083333
    ## 10  0  0  0  0  0  0  0  0  1 47  0  0.02083333
    ## 11  0  0  0  0  0  1  0  0  0  0 47  0.02083333

4. Grid search to fit models with different parameter combinations, using cross validation
==========================================================================================

``` r
rf_grid <- expand.grid(mtry = c(3,4,5), nodesize = c(1,5,10,20,40,80), oob_error = NA_real_)

set.seed(1985)
inc_flds  <- createFolds(vowel.train$y, k=5)
#print(inc_flds)
#sapply(inc_flds, length)  ## not all the same length

for(i in 1:nrow(rf_grid))
{
  cvrf <- function(mtry, nodesize, flds=inc_flds) 
  {
    cverr <- rep(NA, length(flds))
    for(tst_idx in 1:length(flds)) 
    { ## for each fold
      
          ## get training and testing data
      inc_trn <- vowel.train[-flds[[tst_idx]],]
      inc_tst <- vowel.train[ flds[[tst_idx]],]
      
      
      ## fit kNN model to training data
      rf2 <- randomForest(inc_trn[2:11], inc_trn$y, mtry = mtry, nodesize = nodesize)
      
      ## compute error
      cverr[tst_idx] <- mean(rf2$err.rate[,1])
    }
    return(cverr)
  }
   avg_cv_error <- mean(cvrf(mtry = rf_grid[[i,1]], nodesize = rf_grid[[i,2]]))
   rf_grid[[i,3]] <- avg_cv_error
}

# output grid with error rate for each parameter combination
rf_grid
```

    ##    mtry nodesize  oob_error
    ## 1     3        1 0.05991673
    ## 2     4        1 0.06555915
    ## 3     5        1 0.06894067
    ## 4     3        5 0.07544526
    ## 5     4        5 0.07940779
    ## 6     5        5 0.08742338
    ## 7     3       10 0.10982226
    ## 8     4       10 0.11398621
    ## 9     5       10 0.12528149
    ## 10    3       20 0.22353303
    ## 11    4       20 0.22979219
    ## 12    5       20 0.24114876
    ## 13    3       40 0.36874549
    ## 14    4       40 0.37628051
    ## 15    5       40 0.38023806
    ## 16    3       80 0.53103197
    ## 17    4       80 0.54769681
    ## 18    5       80 0.54284630

5. Predict with tuned model to determine misclassification rate
===============================================================

``` r
rf3 <- randomForest(vowel.train[2:11], vowel.train$y, mtry = 3, nodesize = 1)

rf_preds <- predict(rf3, newdata = vowel.test[2:11])

classification_count <- 0
  
for(i in 1:nrow(vowel.test))
{
  if(rf_preds[i] == vowel.test[[i,1]])
  {
    classification_count <- classification_count + 1
  }
}

# misclassification rate
1 - (classification_count/nrow(vowel.test))
```

    ## [1] 0.4134199
