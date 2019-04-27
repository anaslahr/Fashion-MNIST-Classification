library(FactoMineR)
library("corrplot")
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

# load images
train_x = load_image_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/train-images-idx3-ubyte")
test_x  = load_image_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/t10k-images-idx3-ubyte")

# load labels
train_y = as.factor(load_label_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("/Users/anas/Desktop/S2/DataScience/DataScienceProject2019/fashionmnist/t10k-labels-idx1-ubyte"))

train_xy = cbind(train_x, train_y)
test_xy = cbind(test_x, test_y)

dim(train_x)
label_col = 785

# Compute PCA


lab = c(0,1,2,3,4,5,6,7,8,9)
# Compute PCA
#res.PCA = PCA(train_xy, quali.sup = label_col, scale.unit = FALSE, ncp = 50)
res.PCA = PCA(test_xy, quali.sup = label_col, scale.unit = FALSE, ncp = 50)

layout(matrix(c(1,2), ncol=2))
plot.PCA(res.PCA, choix = "ind", habillage = label_col, label = NULL)
plot.PCA(res.PCA, choix = "var",  label = NULL)



test_xx <-as.matrix(test_x)

clothes.labels <-c( "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# Function to rotate matrices
rotate <- function(x) t(apply(x, 2, rev))

# # Function to plot image from a matrix x
# plot_image <- function(x, title = "", title.color = "black") {
#   image(rotate(rotate(x)), axes = FALSE,
#         col = grey(seq(0, 1, length = 256)),
#         main = list(title, col = title.color))
# }
# 
# # We plot 16 cherry-picked images from the training set
# par(mfrow=c(4, 4), mar=c(0, 0.2, 1, 0.2))
# for (i in 1:16) {
#   nr <- i * 10
#   plot_image(matrix(test_x[nr, , , 1], nrow = 28, byrow=TRUE),
#              clothes.labels[as.numeric(train_x[nr, 1] + 1)])
# }

test_xx <-as.matrix(test_x)
rec <- reconst(res.PCA,ncp=50)
dim(rec)



im_matrix <- matrix(test_xx[3,], nrow=28, byrow = TRUE)
typeof(im_matrix)

im_rec_matrix<-matrix(rec[3,] , nrow=28, byrow = TRUE)
typeof(im_rec_matrix)

layout(matrix(c(1,2), ncol=2))
image(im_matrix ,col= grey(seq(0, 1, length = 256)))
image( im_rec_matrix, col=grey(seq(0, 1, length = 256)))




