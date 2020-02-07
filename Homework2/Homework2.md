Homework 2
================
Anthony DeNiro
February 7, 2020

libraries

``` r
library(ElemStatLearn)
```

    ## Warning: package 'ElemStatLearn' was built under R version 3.6.2

``` r
library(magrittr)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(glmnet)
```

    ## Warning: package 'glmnet' was built under R version 3.6.2

    ## Loading required package: Matrix

    ## Loaded glmnet 3.0-2

read in data

``` r
data('prostate')
```

correlation coeficients

``` r
cor(prostate)
```

    ##              lcavol      lweight       age         lbph         svi
    ## lcavol   1.00000000  0.280521386 0.2249999  0.027349703  0.53884500
    ## lweight  0.28052139  1.000000000 0.3479691  0.442264395  0.15538491
    ## age      0.22499988  0.347969112 1.0000000  0.350185896  0.11765804
    ## lbph     0.02734970  0.442264395 0.3501859  1.000000000 -0.08584324
    ## svi      0.53884500  0.155384906 0.1176580 -0.085843238  1.00000000
    ## lcp      0.67531048  0.164537146 0.1276678 -0.006999431  0.67311118
    ## gleason  0.43241706  0.056882099 0.2688916  0.077820447  0.32041222
    ## pgg45    0.43365225  0.107353790 0.2761124  0.078460018  0.45764762
    ## lpsa     0.73446033  0.433319385 0.1695928  0.179809404  0.56621822
    ## train   -0.04654347 -0.009940651 0.1776155 -0.029939957  0.02679950
    ##                  lcp     gleason      pgg45        lpsa        train
    ## lcavol   0.675310484  0.43241706 0.43365225  0.73446033 -0.046543468
    ## lweight  0.164537146  0.05688210 0.10735379  0.43331939 -0.009940651
    ## age      0.127667752  0.26889160 0.27611245  0.16959284  0.177615517
    ## lbph    -0.006999431  0.07782045 0.07846002  0.17980940 -0.029939957
    ## svi      0.673111185  0.32041222 0.45764762  0.56621822  0.026799505
    ## lcp      1.000000000  0.51483006 0.63152825  0.54881317 -0.037427296
    ## gleason  0.514830063  1.00000000 0.75190451  0.36898681 -0.044171456
    ## pgg45    0.631528246  0.75190451 1.00000000  0.42231586  0.100516371
    ## lpsa     0.548813175  0.36898681 0.42231586  1.00000000 -0.033889743
    ## train   -0.037427296 -0.04417146 0.10051637 -0.03388974  1.000000000

``` r
cor(prostate, prostate$lpsa)
```

    ##                [,1]
    ## lcavol   0.73446033
    ## lweight  0.43331939
    ## age      0.16959284
    ## lbph     0.17980940
    ## svi      0.56621822
    ## lcp      0.54881317
    ## gleason  0.36898681
    ## pgg45    0.42231586
    ## lpsa     1.00000000
    ## train   -0.03388974

training and testing sets

``` r
## split prostate into testing and training subsets
prostate_train <- prostate %>%
  filter(train == TRUE) %>% 
  select(-train)


prostate_test <- prostate %>%
  filter(train == FALSE) %>% 
  select(-train)
```

linear model fitting lpsa outcome

``` r
fit <- lm(lpsa ~ ., data = prostate_train)
```

computing test error with L2 loss function for linear model

``` r
## functions to compute testing/training error w/lm
L2_loss <- function(y, yhat)
  (y-yhat)^2

error <- function(dat, fit, loss=L2_loss)
  mean(loss(dat$lpsa, predict(fit, newdata=dat)))


## testing error
error(prostate_test, fit)
```

    ## [1] 0.521274

ridge model fitting with lpsa outcome

``` r
## use glmnet to fit lasso
## glmnet fits using penalized L2 loss
## first create an input matrix and output vector
form  <- lpsa ~ 0 + lweight + age + lbph + lcp + pgg45 + lcavol + svi + gleason # 0 gives no intercept so glmnet doesnt penalize the intercept
x_inp <- model.matrix(form, data=prostate_train)
y_out <- prostate_train$lpsa

#?glmnet()
fit_ridge <- glmnet(x=x_inp, y=y_out, lambda= seq(0.5,0,-0.05), alpha = 0)
#print(fit_ridge$beta)
```

``` r
## plot path diagram
plot(x=range(fit_ridge$lambda),
     y=range(as.matrix(fit_ridge$beta)),
     type='n',
     xlab=expression(lambda),
     ylab='Coefficients')
for(i in 1:nrow(fit_ridge$beta)) {
  points(x=fit_ridge$lambda, y=fit_ridge$beta[i,], pch=19, col='#00000055')
  lines(x=fit_ridge$lambda, y=fit_ridge$beta[i,], col='#00000055')
}
abline(h=0, lty=3, lwd=2)
```

![](Homework2_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
## functions to compute testing/training error with glmnet
error_ridge <- function(dat, fit, lam, form, loss=L2_loss) {
  x_inp <- model.matrix(form, data=dat)
  y_out <- dat$lcavol
  y_hat <- predict(fit, newx=x_inp, s=lam)  ## see predict.elnet
  mean(loss(y_out, y_hat))
}

## compute training and testing errors as function of lambda
err_train_1 <- sapply(fit_ridge$lambda, function(lam) 
  error_ridge(prostate_train, fit_ridge, lam, form))

err_test_1 <- sapply(fit_ridge$lambda, function(lam) 
  error_ridge(prostate_test, fit_ridge, lam, form))

## plot test/train error
plot(x=range(fit_ridge$lambda),
     y=range(c(err_train_1, err_test_1)),
     type='n',
     xlab=expression(lambda),
     ylab='train/test error')
points(fit_ridge$lambda, err_train_1, pch=19, type='b', col='darkblue')
points(fit_ridge$lambda, err_test_1, pch=19, type='b', col='darkred')
legend('topleft', c('train','test'), lty=1, pch=19,
       col=c('darkblue','darkred'), bty='n')
```

![](Homework2_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
colnames(fit_ridge$beta) <- paste('lam =', fit_ridge$lambda)
#print(fit_ridge$beta %>% as.matrix)
```
