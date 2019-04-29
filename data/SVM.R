library(FactoMineR)


# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

#Load Images
train_x = load_image_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/train-images-idx3-ubyte")
test_x  = load_image_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/t10k-images-idx3-ubyte")

#Load Labels
train_y = as.factor(load_label_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/t10k-labels-idx1-ubyte"))

# Add labels as last colum to train and test 
train_xy = cbind(train_x, train_y)
test_xy = cbind(test_x, test_y)

dim(train_x)
#Colomn index for labels
col_lab = 785


svm_fashion<-svm(test_x,test_y,kernel = "linear",
      cost = 100000, scale = FALSE, tol = 0.00000001,shrinkage = FALSE)

plot(svm_fashion,test_x)
  






