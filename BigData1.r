library(MASS)
library(class)
XY<- read.csv('prostate-cancer-1.csv')
x <- XY[,-1]
y <- XY[,1]
sample(5)
n = 79
sample(n)
epsilon <- 1/3
nte <- round(n*epsilon)
nte
ntr = n -nte
id.tr <- sample(sample(sample(n)))[1:ntr]
id.tr
id.te <- setdiff(1:n, id.tr)
id.te
union(id.tr,id.tr)
yte <- knn(x[id.tr,],x[id.te,],y[id.tr],k=1)
yte

#Calculating the errors
# method1
err.te <- (length(which(yte!= y[id.te])))/nte 
err.te

#method2
ind.err <- ifelse(yte != y[id.te],1,0)
mean(ind.err)

#method3
err.te <- 1-sum(diag(table(y[id.te],yte))) /nte
err.te
