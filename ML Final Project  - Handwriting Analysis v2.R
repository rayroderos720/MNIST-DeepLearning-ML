#2Ray Anthony Roderos, Nischitha Rao, Qimeng Deng
#Machine Learning Final Project - Handwriting Analysis

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Step 1 - Load the Project Files and get the details of the datasets
ptm <- proc.time()#Start time

a <- 60000 #size of training set

getwd()
setwd("C:\Users\qd2133\Downloads")
ifelse(exists("MNIST_Test.1"),"",MNIST_Test.1 <- read.csv("MNIST - TEST.csv", header=TRUE))
ifelse(exists("MNIST_Train.1"),"",MNIST_Train.1 <-read.csv("MNIST - TRAIN.csv", header=TRUE))

MNIST_Test <- MNIST_Test.1
MNIST_Train <- MNIST_Train.1

#Loading Required Libraries and setting seed

packages<-function(x){
  x<-as.character(match.call()[[2]])
  if (!require(x,character.only=TRUE)){
    install.packages(pkgs=x)
    require(x,character.only=TRUE)
  }
}

packages(caret)
packages(neuralnet)
packages(nnet)
packages(h2o)

set.seed(2)

#To Save Original Training Set

MNIST_Train_Original <- MNIST_Train

#To Reduce the Set to the chosen size

size <- a
MNIST_Train_nrow <- nrow(MNIST_Train_Original)
Training_size <- round(runif(size,0,MNIST_Train_nrow),0)
MNIST_Train <- MNIST_Train_Original[Training_size,]


#Bootstrap the sample to its original size

MNIST_Train.resample <- sample (nrow(MNIST_Train),size=nrow(MNIST_Train.1),replace = TRUE)
MNIST_Train <- MNIST_Train[MNIST_Train.resample,]


#To Visualize


rotate <- function(x){ #creates a function to rotate the image
  t(apply(x, 2, rev))
} 

#creates a 28x28 matrix from the data of the 1st to the 4th rows, makes the elements numeric
#and applies the rotate function

image1_m <- matrix((MNIST_Train[1,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE) 
image1_m <- apply(image1_m, 2, as.numeric)
image1_m <- rotate(image1_m)

image2_m <- matrix((MNIST_Train[3,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image2_m <- apply(image2_m, 2, as.numeric)
image2_m <- rotate(image2_m)

image3_m <- matrix((MNIST_Train[4,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image3_m <- apply(image3_m, 2, as.numeric)
image3_m <- rotate(image3_m)

image4_m <- matrix((MNIST_Train[5,2:ncol(MNIST_Train)]), nrow=28, ncol=28, byrow = TRUE)
image4_m <- apply(image4_m, 2, as.numeric)
image4_m <- rotate(image4_m)

#create a function to plot the matrix

plot_number <- function(x){
  x <- apply(x, 2, as.numeric)
  image(1:28, 1:28, x, col=gray((0:255)/255))
}

graphics.off() #erases all plots
par(mfrow=c(2,2),pty="s") #creates a 2x2 slots in the plot, pty="s" makees the plot square

plot_number(image1_m) #plots the 1st image (5)
plot_number(image2_m) #plots the 2nd image (4)
plot_number(image3_m) #plots the 3rd image (1)
plot_number(image4_m) #plots the 4th image (9)


#For MNIST_Test and MNIST_Train, there are 10k and 60k oberservations 
#respectively with 784 variables in the form of pixels with values
#ranging in an 8-bit grayscale (0-255 from no black to the darkest black)
#It will create a visual representation of a number if made into a 28x28 matrix
#the reasearchers have determined that it is necesasry to add more variables
#to be able to extract more descriptions about the dataset



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Save the environment

#save.image(file='ML Final Project.RData')

#Load the environment
#rm(list=ls())
#getwd()
#setwd("/Users/Knight Roderos/Documents/R")
#load('ML Final Project.RData')
#Loading Required Libraries and setting seed
#require(caret)
#require(neuralnet)
#set.seed(2)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Try neuralnet model


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#delete features with mean 0

MNIST_Col <- ncol(MNIST_Train)
index.delete <- vector(length=MNIST_Col)

for (i in 1:MNIST_Col){
  MNIST_Train[,i] <- ifelse(mean(MNIST_Train[,i])==0,index.delete[i] <- i,MNIST_Train[,i])
}

index.delete <- subset(index.delete,index.delete >0)
MNIST_Train.alpha <- MNIST_Train[,-index.delete]
MNIST_Test.alpha <- MNIST_Test[,-index.delete]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Setting up the neuralnet

MNIST_Train.Label <- MNIST_Train.alpha$Label
#maxs <- apply(MNIST_Train[,2:MNIST_Col], 2, max)
#mins <- apply(MNIST_Train[,2:MNIST_Col], 2, min)

MNIST_Train.alpha <- MNIST_Train.alpha[,-1]
MNIST.names  <- names(MNIST_Train.alpha)

#MNIST_Train <- scale(MNIST_Train,center = mins, scale = maxs - mins)

#Concatenate the strings
MNIST.forumula <- paste(MNIST.names,collapse="+")
MNIST.forumula <- paste("Label~",MNIST.forumula)

MNIST_Train.alpha <- as.data.frame(cbind(MNIST_Train.Label, MNIST_Train.alpha))
colnames(MNIST_Train.alpha)[1] <- "Label"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Training and plotting the neuralnet

#Run the Neural Net
MNIST.nn.alpha <- neuralnet(MNIST.forumula,MNIST_Train.alpha,hidden=b)

#Save the image file as checkpoint
#save.image(file='ML Final Project nn.RData')

#Plot the Neural Net
#plot(MNIST.nn.alpha)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Prediction using the neuralnet

number.of.col <- ncol(MNIST_Train.alpha)

MNIST.nn.predict.alpha <- compute(MNIST.nn.alpha, MNIST_Test.alpha[,2:number.of.col])
MNIST.round.alpha <- round(MNIST.nn.predict.alpha$net.result,0)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing Accuracy

accuracy <- MNIST.round.alpha == MNIST_Test.alpha$Label
accuracy1 <- length(accuracy[accuracy==TRUE])
p2 <- accuracy1/length(accuracy)
p2

#Computing RMSE

RMSE <- sqrt(mean(MNIST.nn.predict.alpha$net.result-MNIST_Test$Label)^2)
RMSE

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Try nnet model
modelname<-"NeuralNetwork"
mtrain<-mtrain[sample(nrow(mtrain)),]

target<-names(mtrain)[1]  #Label Name
target

inputs<-setdiff(names(mtrain),target)
inputs
length(inputs)

mx=mtrain[,-1] #contrains all pixels value
my=mtrain[,1] #labels only
mxreduced<-mx/255

mxcov<-cov(mxreduced)
mpcax<-prcomp(mxcov)

xfinal <-as.matrix(mxreduced)%*%mpcax$rotation[,1:45]
my<-class.ind(my)

testx<-mtest[,-1]
testxreduced<-testx/255
testxreduced<-as.matrix(testxreduced)%*%mpcax$rotation[,1:45]

model<-nnet(xfinal, y,size=100,softmax = TRUE,maxit = 100,MaxNWts = 60000)

predicted<-predict(model,testxreduced,type="class")
repdicted<-as.data.frame(predicted)
head(predicted)

actual<-as.double(unlist(mtrain[target]))
head(actual)

accuracy<-round(mean(actual==predicted),2)
accuracy

result<-data.frame(modelname,accuracy)[1:1,]
result

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Try h2o model
#Start a local h2o cluster
local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

#Convert digit labels to factor for classification
MNIST_Train.1[,1]<-as.factor(MNIST_Train.1[,1])

#Change the dataset to the H2O instance
Train.h2o<-as.h2o(MNIST_Train.1)
Test.h2o<-as.h2o(MNIST_Test.1)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,1),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 1
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 2
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,3),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 3
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,4),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 4
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,5),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 5
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,6),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 6
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,7),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 7
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(100,8),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 8
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 100
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(200,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 9
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 200
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(300,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 10
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 300
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(400,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 11
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 400
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(500,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 12
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 500
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(600,2),epochs = 20)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 13
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 600
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Train the Model
MNIST.h20 <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(600,2),epochs = 30)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Use model to predict testing dataset

MNIST_Predict<-h2o.predict(object=MNIST.h20, newdata=Test.h2o[,-1])
MNIST_Predict.df<-as.data.frame(MNIST_Predict)

x <-sum(diag(table(MNIST_Test.1$Label,MNIST_Predict.df[,1])))
p2 <- x/length(MNIST_Test.1$Label)

#Compute for error rate
error <- 1-p2

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Computing FOM

p1 <- size/nrow(MNIST_Train.1)
p1/2
1-p2
FOM <-(p1/2) + (1-p2)
FOM

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Creats a performance trackng table

row.tracker <- 14
ifelse(exists("NN.Tracker"),"",NN.Tracker <- matrix(,nrow=20,ncol=6))
colnames(NN.Tracker) <- c("% of Training Set","Accuracy","Error Rate","FOM","Time","Neurons per Layer")
NN.Tracker[row.tracker,1] <- p1
NN.Tracker[row.tracker,2] <- p2
NN.Tracker[row.tracker,3] <- error
NN.Tracker[row.tracker,4] <- FOM
time <- proc.time() - ptm 
time <- as.numeric(time[3])
NN.Tracker[row.tracker,5] <- time
NN.Tracker[row.tracker,6] <- 600
NN.Tracker

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Additional Requirement
MNIST_Predict.df <- h2o.deeplearning(x = 2:785, y = 1, Train.h2o, activation = "Tanh", hidden=rep(600,2),epochs = 20)

fit_kmeans_test <- kmeans(c(MNIST_Predict.df$p0,
                            MNIST_Predict.df$p1,
                            MNIST_Predict.df$p2,
                            MNIST_Predict.df$p3,
                            MNIST_Predict.df$p4,
                            MNIST_Predict.df$p5,
                            MNIST_Predict.df$p6,
                            MNIST_Predict.df$p7,
                            MNIST_Predict.df$p8,
                            MNIST_Predict.df$p9),7, iter.max=100,nstart=10)

table(data.frame(fit_kmeans_test$cluster, MNIST_Test$Label))

#Calculate Confusion

calculate.confusion <- function(states, clusters)
{
  # generate a confusion matrix of cols C versus states S
  d <- data.frame(state = states, cluster = clusters)
  td <- as.data.frame(table(d))
  # convert from raw counts to percentage of each label
  pc <- matrix(ncol=max(clusters),nrow=0) # k cols
  for (i in 1:10) # 10 labels
  {
    total <- sum(td[td$state==td$state[i],3])
    pc <- rbind(pc, td[td$state==td$state[i],3]/total)
  }
  rownames(pc) <- td[1:10,1]
  return(pc)
}

#Assign Cluster Labels
assign.cluster.labels <- function(cm, k)
{
  # take the cluster label from the highest percentage in that column
  cluster.labels <- list()
  for (i in 1:k)
  {
    cluster.labels <- rbind(cluster.labels, row.names(cm)[match(max(cm[,i]), cm[,i])])
  }
  
  # this may still miss some labels, so make sure all labels are included
  for (l in rownames(cm)) 
  { 
    if (!(l %in% cluster.labels)) 
    { 
      cluster.number <- match(max(cm[l,]), cm[l,])
      cluster.labels[[cluster.number]] <- c(cluster.labels[[cluster.number]], l)
    } 
  }
  return(cluster.labels)
}

#Creates the list of cluster labels
str(assign.cluster.labels(calculate.confusion(MNIST_Test$Label, fit_kmeans_test$cluster), 10))

#Calculate Accuracy
calculate.accuracy <- function(states, clabels)
{
  matching <- Map(function(state, labels) { state %in% labels }, states, clabels)
  tf <- unlist(matching, use.names=FALSE)
  return (sum(tf)/length(tf))
}

#Run the functions to get the accuracy
k <- length(fit_kmeans_test$size)
conf.mat <- calculate.confusion(MNIST_Test$Label, fit_kmeans_test$cluster)
cluster.labels <- assign.cluster.labels(conf.mat, k)
acc <- calculate.accuracy(MNIST_Test$Label, cluster.labels[fit_kmeans_test$cluster])
cat("For", k, "means with accuracy", acc, ", labels are assigned as:\n")
cat(str(cluster.labels))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Save the image file as checkpoint
#save.image(file='ML Final Project nn.RData')
#load('ML Final Project nn.RData')

