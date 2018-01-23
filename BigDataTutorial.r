library(mlbench)
xy <- PimaIndiansDiabetes
x <- xy[,-pos]
dim(x)
S <- cov(x)
eigs <- eigen(S)
summary(eigs)
lambdas <- eigs$values
lambdas
cumsum(lambdas /sum(lambdas))
eta <- 0.9
q.opt<- min(which(cumsum(lambdas /sum(lambdas))>= eta))
pc.x <- princomp(x)#runs the principal component
summary(pc.x)
summary(summary(pc.x))
z.tilde <- summary(pc.x)$scores
dim(z.tilde)
z<- z.tilde[,1:q.opt]
z
graphics.off()
plot(z, col = 1 + as.numeric(xy[,pos]))
