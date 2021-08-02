#Install Packages
library(psych) #for describe
library(forecast) #for time-series analysis
library(ggcorrplot) #for correlation plot
library(tseries) #for ARIMA model
install.packages("tsoutliers")
library(tsoutliers)

#get file
pv = read.csv(file.choose(), header = TRUE)
View(pv)
str(pv)
head(pv)
tail(pv)
summary(pv)

#data summary
colnames(pv) = c("time","nominal_power_output","daily_power_yields","dc_voltage",
                 "dc_current","total_dc_power","phase_voltage","phase_current",
                 "total_active_power","power_factor","grid_frequency","temperature","ring_temperature",
                 "wind_speed","dhi","dni","ebh","ghi")

describe(pv)
str(pv)
summary(pv)

#Visualization
#Histograms
hist(pv$nominal_power_output)
hist(pv$daily_power_yields)
hist(pv$dc_voltage)
hist(pv$dc_current)
hist(pv$total_dc_power)
hist(pv$phase_voltage)
hist(pv$phase_current)
hist(pv$total_active_power)
hist(pv$power_factor)
hist(pv$grid_frequency)
hist(pv$ring_temperature)
hist(pv$temperature)
hist(pv$wind_speed)
hist(pv$dhi)
hist(pv$dni)
hist(pv$ebh)
hist(pv$ghi)

#Overlapping Histograms
hist(pv$total_dc_power, col="red", main = "Overlap of Total_DC_Power and Total_Active_Power")
hist(pv$total_active_power, add=T, col=rgb(0, 1, 0, 0.5) )

hist(pv$temperature, col="red", xlim = c(20,45), ylim = c(0,200), main = "Overlap of Ring_Temperature and Temperature")
hist(pv$ring_temperature, add=T, col=rgb(0, 1, 0, 0.5) )

hist(pv$ghi, col="red", ylim = c(0,600), main = "Overlap of ghi, ebh, dni and dhi")
hist(pv$dhi, add=T, col=rgb(0, 1, 0, 0.5) )
hist(pv$dni, add=T, col=rgb(0, 1, 1, 0.5) )
hist(pv$ebh, add=T, col=rgb(1, 1, 0, 0.5) )

#Transforming Data to Time-Series

#Total Active Power
pv.active.power = ts(data = pv$total_active_power,
                 start = 1, frequency = 24)
str(pv.active.power)
plot(pv.active.power, main="Total_Active_Power")
decompose(pv.active.power, type = "additive")
plot(decompose(pv.active.power, type = "additive"))

#Ghi
pv.ghi = ts(data = pv$ghi,
                     start = 1, frequency = 24)
str(pv.ghi)
plot(pv.ghi, main="ghi")
decompose(pv.ghi, type = "additive")
plot(decompose(pv.ghi, type = "additive"))

#Temperature
pv.temp = ts(data = pv$temp,
            start = 1, frequency = 24)
str(pv.temp)
plot(pv.temp, main="Temperature")
decompose(pv.temp, type = "additive")
plot(decompose(pv.temp, type = "additive"))

#Outlier detection
boxplot(pv.active.power, main = "Boxplot of Total Active Power")

boxplot(pv.ghi, main = "Boxplot of GHI")

boxplot(pv.temp, main = "Boxplot of Temperature")

#correlation matrix
pv.corr = pv[,-(1:11)]
pv.corr = pv.corr[,-(4:6)]
pv.corr = pv.corr[,-2]
pv.corr = pv.corr[,-2]
View(pv.corr)
ggcorrplot(round(cor(pv.corr),1), lab=TRUE)

#Model
#ARIMA
#Forecasting pv.dc.power
train.dc.power = window(pv.dc.power, start=1, end=20)
test.dc.power = window(pv.dc.power, start=20, end=29)
plot(train.dc.power)
plot(test.dc.power)

auto.arima(train.dc.power, trace = TRUE, approximation = FALSE, stepwise = FALSE)
arima.train.dc.power = arima(train.dc.power, order = c(1,0,1),c(2,1,1))
arima.train.dc.power
forecast.dc.train = forecast(arima.train.dc.power, h=211)

plot(forecast.dc.train)
lines(test.dc.power, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.dc.power","forecast.dc.train"))

accuracy(forecast.dc.train, test.dc.power)

#Forecasting pv.active.power
train.active.power = window(pv.active.power, start=1, end=20)
test.active.power = window(pv.active.power, start=20, end=29)
plot(train.active.power)
plot(test.active.power)

auto.arima(train.active.power, trace = TRUE, approximation = FALSE, stepwise = FALSE)
arima.train.active.power = arima(train.active.power, order = c(1,0,1),c(2,1,1))
arima.train.active.power
forecast.active.train = forecast(arima.train.active.power, h=211)

plot(forecast.active.train)
lines(test.active.power, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.active.power","forecast.active.train"))

accuracy(forecast.active.train, test.active.power)

#Forecasting temperature
train.temp = window(pv.temp, start=1, end=20)
test.temp = window(pv.temp, start=20, end=29)
plot(train.temp)
plot(test.temp)

auto.arima(train.temp, trace = TRUE, approximation = FALSE, stepwise = FALSE)
arima.train.temp = arima(train.temp, order = c(1,0,1),c(2,1,1))
arima.train.temp
forecast.train.temp = forecast(arima.train.temp, h=211)

plot(forecast.train.temp)
lines(test.temp, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.temp","forecast.train.temp"))

accuracy(forecast.train.temp, test.temp)

#Forecasting ghi
train.ghi = window(pv.ghi, start=1, end=20)
test.ghi = window(pv.ghi, start=20, end=29)
plot(train.ghi)
plot(test.ghi)

auto.arima(train.ghi, trace = TRUE, approximation = FALSE, stepwise = FALSE)
arima.train.ghi = arima(train.ghi, order = c(2,0,0),c(2,1,1))
arima.train.ghi
forecast.train.ghi = forecast(arima.train.ghi, h=211)

plot(forecast.train.ghi)
lines(test.ghi, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.ghi","forecast.train.ghi"))

accuracy(forecast.train.ghi, test.ghi)

#Forecasting wind speed
train.wind.speed = window(pv.wind.speed, start=1, end=20)
test.wind.speed = window(pv.wind.speed, start=20, end=29)
plot(train.wind.speed)
plot(test.wind.speed)

auto.arima(train.wind.speed, trace = TRUE, approximation = FALSE, stepwise = FALSE)
arima.train.wind.speed = arima(train.wind.speed, order = c(0,1,2),c(2,0,0))
arima.train.wind.speed
forecast.train.wind.speed = forecast(arima.train.wind.speed, h=211)

plot(forecast.train.wind.speed)
lines(test.wind.speed, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.wind.speed","forecast.train.wind.speed"))

accuracy(forecast.train.wind.speed, test.wind.speed)

#Model
#Neural Network - total_dc_power
train.dc.power = window(pv.dc.power, start=1, end=20)
test.dc.power = window(pv.dc.power, start=20, end=29)
plot(train.dc.power)
plot(test.dc.power)

set.seed(100)
nn.train.dc.power = nnetar(train.dc.power, P = 9, size = 5)
nn.train.dc.power

forecast.dc.train.nn = forecast(nn.train.dc.power, h=211, PI=F)

plot(forecast.dc.train.nn)
lines(test.dc.power, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.dc.power","forecast.dc.train.nn"))

accuracy(forecast.dc.train.nn, test.dc.power)

#Neural Network - total_active_power
train.active.power = window(pv.active.power, start=1, end=20)
test.active.power = window(pv.active.power, start=20, end=29)
plot(train.active.power)
plot(test.active.power)

set.seed(100)
nn.train.active.power = nnetar(train.active.power, P = 18, size = 7)
nn.train.active.power

forecast.active.train.nn = forecast(nn.train.active.power, h=211, PI=F)

plot(forecast.active.train.nn)
lines(test.active.power, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.active.power","forecast.active.train.nn"))

accuracy(forecast.active.train.nn, test.active.power)

#Neural Network - temperature
train.temp = window(pv.temp, start=1, end=20)
test.temp = window(pv.temp, start=20, end=29)
plot(train.temp)
plot(test.temp)

set.seed(100)
nn.train.temp = nnetar(train.temp, P = 13, size = 2)
nn.train.temp

forecast.temp.train.nn = forecast(nn.train.temp, h=211, PI=F)

plot(forecast.temp.train.nn)
lines(test.temp, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.temp","forecast.temp.train.nn"))

accuracy(forecast.temp.train.nn, test.temp) 

#Neural Network - ghi
train.ghi = window(pv.ghi, start=1, end=20)
test.ghi = window(pv.ghi, start=20, end=29)
plot(train.ghi)
plot(test.ghi)

set.seed(100)
nn.train.ghi = nnetar(train.ghi, P = 13, size = 3)
nn.train.ghi

forecast.ghi.train.nn = forecast(nn.train.ghi, h=211, PI=F)

plot(forecast.ghi.train.nn)
lines(test.ghi, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.ghi","forecast.ghi.train.nn"))

accuracy(forecast.ghi.train.nn, test.ghi) 

#Neural Network - wind speed
train.wind.speed = window(pv.wind.speed, start=1, end=20)
test.wind.speed = window(pv.wind.speed, start=20, end=29)
plot(train.wind.speed)
plot(test.wind.speed)

set.seed(100)
nn.train.wind.speed = nnetar(train.wind.speed, P = 18, size = 9)
nn.train.wind.speed

forecast.wind.speed.train.nn = forecast(nn.train.wind.speed, h=211, PI=F)

plot(forecast.wind.speed.train.nn)
lines(test.wind.speed, col="red")
legend("topleft",lty=1,bty = "n",col=c("red","blue"),
       c("test.wind.speed","forecast.wind.speed.train.nn"))

accuracy(forecast.wind.speed.train.nn, test.wind.speed) 
