---
title: "TS|forecasting"
author: "Krasnukhina"
date: "2022-10-29"
output: html_document
---

```{r}
#Forecasting and seasonality
#1. Download the whole dataset
#install.packages("remotes")
#remotes::install_github("FinYang/tsdl")
```


```{r}
library(tsdl)
library(zoo)
library(xts)
library(knitr)
library(PerformanceAnalytics)
library(ggplot2)
library(lmtest)
library(forecast)
library(dplyr)
library(tseries)
library(urca)
library(DescTools)
library(PerformanceAnalytics)


tsdl_monthly <- subset(tsdl, 12)
#In the variable tsdl_monthly you will have lots of series in monthly dynamics
#Name of each variable you can find by the code
attributes(tsdl_monthly[[18]])
unemp = tsdl_monthly[[18]]
```
START
```{r}
#EDA Basic descriptive statistics, plot the variable
summary(unemp)
str(unemp)
head(unemp, n = 10)
sum(is.na(unemp))
sum(is.null(unemp))
#first view on the data
autoplot(unemp)
```
Что-то интересное:)
Кажется, что есть и сезонность, и изменяющийся тренд, и дисперсия не константа (в начале как будто побольше, чем в конце)
Реализуем необходимые преобразования. Просто первой разности не хватает, она борется только с трендом, поэтому реализуем log-diff.
```{r}
log_unemp = log(unemp)
autoplot(log_unemp)
log_unemp = as.xts(log_unemp)
# First difference
log_diff_unemp = diff.xts(log_unemp, 1, 1)

#remove first NA value
log_diff_unemp = log_diff_unemp[-1, ]
autoplot(log_diff_unemp)

```


```{r}
#remove outliers
t1 = tsoutliers(as.ts(log_diff_unemp),iterate=5, lambda = NULL)
log_diff_unemp[t1$index] = t1$replacements
autoplot(log_diff_unemp)
```


SEASONALITY
```{r}
chart.ACFplus(as.ts(log_diff_unemp))
```
Имеем четко выраженную сезонность, построим дополнительные графики.
```{r}
#1.	Plot behavior in every month
ggsubseriesplot(as.ts(log_diff_unemp))
```

```{r}
ggseasonplot(as.ts(log_diff_unemp))
```
Согласно графику наблюдаем годичную сезонность, построим периодограмму, чтобы убедиться в правильности рассуждений.
```{r}
#Plot Box plot for each month in the data, conclude about significant month in the data.
Dta<-coredata(log_diff_unemp)
boxplot(Dta~as.ts(cycle(log_diff_unemp)),xlab="Date", ylab = "Y")
```

```{r}
#Draw periodogram, interpret, conclude about seasonality
# install.packages("TSA")
library("TSA")
P1<-periodogram(log_diff_unemp, log='no',plot=TRUE,ylab="Periodogram", xlab="Frequency")
```
Видим один очевидный пик, второй по сравнению с ним не является значительным. Более того, при построении дополнительных графиков, которые расположены выше, сезонность внутри года выявлена не была.

```{r}
#plot for a part of graph
plot(P1$spec~P1$freq,subset=P1$freq<=0.1)
```
Омега = 0.084, тогда T = 1/0.084 ~ 11,9.
Наши предположения о том, что присутствует годичная сезонность, оказались верны.




UNIT ROOT TESTS
```{r}
# Apply simple augmented Dickey-Fuller. Interpret results. 
x<-ur.df(log_diff_unemp, type = "none", selectlags = "AIC")
summary(x)
```
Результаты теста ADF: тип теста - "none", уровень значимости 1%, tau1=-2,58, статистика -7,9106 => статистика<квантиля, Н0 отклоняем, ряд стационарен.

```{r}
# Apply Phillips-Perron test. Interpret results.
pp.test(log_diff_unemp)
```

Результаты теста PP: уровень значимости 1%, p-value меньше 0.01 (p-value smaller than printed p-value) <0.01 значит отклоняем Н0 => ряд стационарен 


```{r}
# Apply KPSS test. Interpret results.
kpss.test(log_diff_unemp)
```
Результаты теста KPSS: уровень значимости 1%, p-value=0.1>0.01 значит не можем отклонить Н0 => ряд стационарен 

Общий вывод о стационарности переменной:
Исходя из 3 проведенных тестов, переменная является стационарной.



4 | TRAIN & TEST SAMPLES
```{r}
#Set a sample size for training model, split the sample into train and test subsamples.
log_diff_unemp = as.xts(log_diff_unemp)

# Create train
train <- log_diff_unemp[1:179, ]

# Create test
test <- log_diff_unemp[(180):nrow(log_diff_unemp), ]
```
Для того чтобы не нарушить структуру сезонности, мы разделили выборку в пропорции 75% на train и 25% на test. Построим отдельные графики, чтобы убедиться в этом.
При удалении выбросов январь первого года был удален, поэтому тренировочная выборка начинается с февраля.
```{r}
autoplot(train)
```
```{r}
autoplot(test)
```
NAIVE MODEL
```{r}
#Fit naive model to compare forecast with. Predict with naive model for test subgroup.
X<-naive(train,h=60)
naive_fit = X$mean

#automatic approach
rmse_naive = RMSE(naive_fit, test)
mape_naive = MAPE(naive_fit, test)
theilu_naive = TheilU(test, naive_fit, type = 2)

#Запишем полученные результаты в датафрейм
naive_name = "NAIVE"
aic_naive = "NA"
naive = data.frame(naive_name, rmse_naive, mape_naive, theilu_naive, aic_naive)

names(naive)[1] <- "Model"
names(naive)[2] <- "RMSE"
names(naive)[3] <- "MAPE"
names(naive)[4] <- "TheilU"
names(naive)[5] <- "AIC"

#manual approach
# Compute errors: error
    #error <- test - naive_fit

# Calculate RMSE
    #sqrt(mean(error^2))
```

5 | SEASONAL MODELS
```{r}
#ARIMA with Fourier series
#К = 6, так как T = 12, следовательно, T/2=6, а K должно быть <= T/2, h = 60 = длина тестовой выборки
train_fourier.model  <- auto.arima(train, xreg=fourier(train,K=6), seasonal=TRUE)
train_fourier.fcast <- forecast(train_fourier.model, xreg=fourier(train, K=6, h=60), level=90)
autoplot(train_fourier.fcast) + xlab("Year")

checkresiduals(train_fourier.model)
aic_fourier = AIC(train_fourier.model)

#Рассчитаем ошибки прогноза для модели ARIMA с включением рядов Фурье
rmse_fourier = RMSE(train_fourier.fcast$mean, test)
mape_fourier = MAPE(train_fourier.fcast$mean, test)
theilu_fourier = TheilU(test, train_fourier.fcast$mean, type=2)

#Запишем полученные результаты в датафрейм
fourier_name = "FOURIER"
fourier = data.frame(fourier_name, rmse_fourier, mape_fourier, theilu_fourier, aic_fourier)

names(fourier)[1] <- "Model"
names(fourier)[2] <- "RMSE"
names(fourier)[3] <- "MAPE"
names(fourier)[4] <- "TheilU"
names(fourier)[5] <- "AIC"

print(fourier)
```



Для выбора лучшей спецификации модели SARIMA нами были перебраны все возможные сочетания параметров p,q,P,Q в интервале от 0 до 10. Для структурированного качественного перебора использовались циклы, находящиеся в конце кода, в разделе "APPLICATIONS". При оценке некоторых спецификаций цикл нарушался, поэтому мы переходили к перебору внутри интервала до ломанной спецификации, а затем оценивали следующий интервал, меняя начальный параметр последовательности, определяющей один из необходимых параметров (p, q, P, Q). Наилучшей из исследованных спецификаций стала модель SARIMA (0,0,2)(6,0,6) m=12, так как ее остатки ближе всего к white noise, при этом достаточно маленький AIC и адекватное количество значимых коэффициентов.
```{r}
#SARIMA
# Using SARIMA model for a seasonality "ts" object
x = Arima(train, order = c(0,0,2), seasonal = list(order = c(6,0,6), period = 12), method="ML")
checkresiduals(x)
aic_sarima = AIC(x)
SARIMA.fcast <- forecast(x, h=60, level=90)
autoplot(SARIMA.fcast) + xlab("Year")

#Рассчитаем ошибки прогноза
rmse_sarima = RMSE(SARIMA.fcast$mean, test)
mape_sarima = MAPE(SARIMA.fcast$mean, test)
theilu_sarima = TheilU(test, SARIMA.fcast$mean, type=2)

#Запишем полученные результаты в датафрейм
sar_name = "SARIMA"
sarima = data.frame(sar_name, rmse_sarima, mape_sarima, theilu_sarima, aic_sarima)

names(sarima)[1] <- "Model"
names(sarima)[2] <- "RMSE"
names(sarima)[3] <- "MAPE"
names(sarima)[4] <- "TheilU"
names(sarima)[5] <- "AIC"

print(sarima)
```

```{r}
#Создадим кумулятивный датафрейм с рассчитанными показателями по каждой из оцененных моделей
models = data.frame()
models = rbind(naive, fourier, sarima)
```

Определим содержательное значение рассчитанных показателей:
RMSE (Root Mean Squared Error), исходя из формулы, обозначает стандартное отклонение остатков, поскольку является корнем из MSE (дисперсии остатков). Преимущество по отношению к MSE: совпадает по размерности с Y, что обеспечивает более простую интерпретацию. В нашем случае в среднем оцененное значение имеет стандартное отклонение от теоретического равное 6 единицам.
 
MAPE Средняя абсолютная ошибка в процентах без знака ошибки является метрикой для оценки проблем регрессии. Идея этой метрики — быть чувствительной к относительным ошибкам. Например, метрика не изменяется глобальным масштабированием целевой переменной

Согласно результатам по показателю RMSE лучшей из оцененных нами моделей является модель SARIMA, так же она является лучше по показателю TheilU. 

Что касается показателей MAPE и AIC, лучшей по ним является модель ARIMA с рядами фурье.

Рассчитанные ошибки прогноза показывают различные результаты ввиду дифференцированного подхода к вычислениям. Оценка RMSE выражается в абсолютном значение, тогда как MAPE-оценка является относительным показателем, выраженным в процентах. Содержательно оценки также несколько отличаются. В нашем случае мы исследуем относительны показатель - уровень безработицы. Более информативным поэтому является показатель MAPE, который позволяет оценить, на сколько процентов в среднем прогнозная оценка отличается от наблюдаемой в реальности, что позволяет качественнее оценить масштаб бедтсвия:)
Кроме того, при выборе спецификации модели SARIMA нам не удалось достичь идельных остатков, поэтому рассчитанные показатели не совсем корректно сравнивать. Имея в виду все выщесказанное, в качестве лучшей модели мы определили модель ARIMA с рядами фурье.

В сравнении с наивным прогнозом, проведенное нами исследование свидетельствует о лучшей оценке нежели наивная. По всем рассчитанным показателям наивный прогноз уступает двум другим моделям.







APPLICATIONS
________________________________________________________________________________
FAST CYCLE FOR CHECKING AIC|RMSE|P-VALUE OF SPECIFICATIONS

d=0
D=0
s=12

for(p in 1:5){
  for(q in 1:5){
    for(P in 1:5){
      for(Q in 1:5)
        {model<-arima(x=train,order=c(p-1,d,q-1),seasonal=list(order=c(P-1,D,Q-1),period=s))
        test<-Box.test(model$residuals,lag=log(length(model$residuals)))
        RMSE=sqrt(mean(model$residuals^2))
        cat(p-1,d,q-1,P-1,D,Q-1, "AIC:",model$aic,"RMSE:",RMSE,'p-value:',test$p.value,'\n')
        }
      }
    }
  }



CYCLE FOR CHECKING RESIDUALS
for (p in seq(2, 2, 1)){
  for (q in seq(3, 5, 1)){
    # Make ARMA model with appropriate lags, write equation 
    x = Arima(train, order = c(p,0,q), seasonal = list(order = c(p, 0, q), period = 12), method = "ML")
    checkresiduals(x)
  }
}



CYCLE FOR CHECKING AIC
p = 0
q = 0
aic_bic = data.frame()
for (p in seq(0, 2, 1)){
  for (q in seq(0, 5, 1)){
    if (p == 3 & q == 0){
      next
    }
    # Make ARMA model with appropriate lags, write equation 
    x = Arima(train,order=c(p,0,q), seasonal = list(order = c(p, 0, q), period = 12))
    aic_bic = rbind(aic_bic, c(p, q, AIC(x), BIC(x)))
  }
}

names(aic_bic)[1] <- "AR(p)"
names(aic_bic)[2] <- "MA(q)"
names(aic_bic)[3] <- "AIC"
names(aic_bic)[4] <- "BIC"

aic_bic %>%
  arrange(AIC)




CYCLE FOR CEHCKING THE LEVEL OF SIGNIFICANCE OF COEFFICIENTS
for (p in seq(0, 1, 1)){
  for (q in seq(0, 1, 1)){
    # Make ARMA model with appropriate lags, write equation 
    x = Arima(train,order=c(p,0,q), seasonal = list(order = c(p, 0, q), period = 12))
    # check the common significance of coefficients 
    coeftest = coeftest(x)
    print(coeftest)
  }
}



