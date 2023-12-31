---
title: "TS | 3.rds"
author: "Sitnikova, Krasnukhina"
date: "2022-10-07"
output:
  pdf_document: default
  html_document:
    df_print: paged
  word_document: default
---

```{r}
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
```


```{r}
#download RData
ts = readRDS("/Users/olesyamba/Downloads/TS/3.rds")

#Проведем первичный визуальный анализ
autoplot(ts)
```
По результатам проведенного визуального анализа можно сказать, что предположительно ряд является стационарным, не наблюдает как смещение, так и тренд. Диспресия также кажется примерно равно константе, так как не меняется во времени.

TASK 2 | Apply unit root test for checking if the series is non-stationary 

``` {r}
# Apply simple augmented Dickey-Fuller
x<-ur.df(ts, type = "none", selectlags = "AIC")
summary(x)
#Тест односторонний (левосторонний): -9.33 < -2.58, то есть значение статистики меньше значения квантиля на 1-процентном уровне значимости, таким образом нулевая гипотеза о нестационарности ряда отклоняется. Ряд стационарен. 
```

```{r}
# Apply Phillips-Perron test
pp.test(ts)
#p-value<1-процентного уровня значимости, следовательно нулевая гипотеза о нестационарности ряда отклоняется на 1 уровне значимости. Ряд стационарен. 
```

```{r}
# Apply KPSS test
kpss.test(ts)
#p-value больше 10-процентного уровня значимости, таким образом на любом адекватном уровне значимости нет оснований отвергать нулевую гипотезау о стационарности ряда. Ряд стационарен.
```
Таким образом, на основании результатов 3 проведенных тестов (ADF, PP, KPSS) можно сделать вывод, что ряд стационарен. 

TASK 3 | Choose appropriate ARMA specification according to at least 4 criteria
```{r}
#Check the correlogram 
chart.ACFplus(ts)
```
Судя по коррелограммам, у нас может быть ARIMA(4,0,4), так как четвертый лаг выходит за пределы доверительного интервала, но стоит проверить спецификации с порядком интегрирования AR от 0 до 4  и MA от 0 до 4, так как ситуация неоднозначная, а также значимость лагов по отдельности при оценке моделей и совместную значимость лагов = от 1 до 3



Наиболее важным критерием качества для выбора спецификации является анализ остатков, начнем с него.
Проверим остатки для каждой из исследуемых спецификаций с помощью цикла:
``` {r}
for (p in seq(0, 4, 1)){
  for (q in seq(0, 4, 1)){
  # Make ARMA model with appropriate lags, write equation 
  x = Arima(ts,order=c(p,0,q))
  checkresiduals(x)
  }
}
```
По результатам анализа остатков допустимыми спецификациями являются модели ARIMA с порядками интегрирования p и q равными (0,4), (1,4), (2,3), (2,4), (3,2), (3,3), (3,4), (4,0), (4,1), (4,2), (4,3), (4,4). 

Далее сравним спецификации с помощью информационных критериев AIC и BIC:
```{r}
aic_bic = data.frame()
for (p in seq(0, 4, 1)){
  for (q in seq(0, 4, 1)){
  # Make ARMA model with appropriate lags, write equation 
  x = Arima(ts,order=c(p,0,q))
  aic_bic = rbind(aic_bic, c(p, q, AIC(x), BIC(x)))
  }
}

names(aic_bic)[1] <- "AR(p)"
names(aic_bic)[2] <- "MA(q)"
names(aic_bic)[3] <- "AIC"
names(aic_bic)[4] <- "BIC"

aic_bic %>%
  arrange(AIC)
```
Можно заметить, что в первую пятерку лучших спецификаций вошли спецификации с p и q равными (2,4), (0,4), (3,4), (1,4) и (1,3), однако последняя спецификация является некачественной по результатам анализа остатков. На данный момент фаворитами являются:(2,4), (0,4), (3,4), (1,4), они расположены в порядке убывания привлекательности, согласно значениям AIC.

Далее рассмотрим значимость коэффициентов, как критерий качества спецификаций. 
```{r}
for (p in seq(0, 4, 1)){
  for (q in seq(0, 4, 1)){
  # Make ARMA model with appropriate lags, write equation 
  x = Arima(ts,order=c(p,0,q))
  # check the common significance of coefficients 
  coeftest = coeftest(x)
  print(coeftest)
  }
}
```
Согласно результатам анализа значимости коэффициентов наиболее качественными спецификациями являются: (1,3), (2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,1). Однако если посмотреть на пересечение результатов анализа остатков и анализа с помощью информационного критерия AIC, можно заметить, что наилучшими спецификациями являются спецификации с порядками интегрирования p и q равными (2,4) (наименьший AIC), (3,4). 


Проведем тест Box-Pierce на совместную значимость лагов для каждой из возможных спецификаций порядка AR до 4:
``` {r}
# calculate Q statistics for lag = 1:4
Box.test(ts, lag = 1, type = c("Box-Pierce"))
```
На 10% уровне значимости лаг равный 1 для AR части значим
```{r}
Box.test(ts, lag = 2, type = c("Box-Pierce"))
```
Два лага совместно не значимы на любом адекватном уровне значимости
```{r}
Box.test(ts, lag = 3, type = c("Box-Pierce"))
```
Три лага совместно не значимы на любом адекватном уровне значимости
```{r}
Box.test(ts, lag = 4, type = c("Box-Pierce"))
```

Четыре лага совместно значимы на 1-процентном уровне значимости. Это говорит о том, что стоит рассмотреть спецификации модели, где в части AR будет 4 лага. Наиболее качественной из них является модель ARIMA(4,0,1), однако она проигрывает спецификации ARIMA(2,0,4) по информационному критерию AIC. Так как он приоритетнее результатов теста Box-Pierce на совместную значимость лагов, наилучшей спецификацией, по нашему мнению, является ARIMA(2,0,4).

В завершении используем автоматическое определение порядков интегрирования и сравним спецификацию, подбираеумую с помощью этой функции, с той, которую мы считаем наилучшей.
``` {r}
auto.arima(ts, stationary = TRUE, seasonal = FALSE, stepwise=FALSE, approximation=FALSE)
x04 = Arima(ts,order=c(0,0,4))
checkresiduals(x04)
```
Согласно функции автопределения порядка интегрирования, в нашем временном ряду есть лаг в MA части 4 порядка интегрирования, следует отметить, что эта спецификация имеет хорошие остатки, 2 по величине значение AIC, но в ней значим только один лаг. Мы рассматривали данную спецификацию, но не выбрали ее в качестве лучшей, так как у спецификации (2,4) AIC еще меньше и больше значимых лагов.

TASK | 3*
Мы полагаем, что результаты автоопределения порядка интегрирования не совпадают с полученными нами результатами, поскольку, судя по всему, функция рассчитывает значение AIC несколько другим способом, так как значение посчитанное данной функцией отличается от значения, рассчитанного нами. Кроме того, мы предполагаем, что функция старается максимально упростить модель и отвергает спецификации с большими (не столь нужными, ввиду несильного изменения AIC) порядками интегрирования. 


По нашему мнению, наилучшей спецификацией является спецификация с порядками интегрирования (2,4). Оценим уравнение с помощью модели ARIMA, чтоюы определить коэффициенты.
```{r}
x24 = Arima(ts,order=c(2,0,4))
  checkresiduals(x)
  coef = print(x24$coef)
```

