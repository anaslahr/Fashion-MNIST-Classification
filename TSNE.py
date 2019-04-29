import matplotlib
import pandas

from mnist_reader import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %matplotlib inline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import seaborn as sns



X_train, y_train = load_mnist('data/', kind='train')

print(X_train.shape)

print(y_train)

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

# Utility function to visualize the outputs of PCA and t-SNE

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f.show(), ax, sc, txts




# Subset first 20k data points to visualize
x_subset = X_train[0:20000]
y_subset = y_train[0:20000]

print(np.unique(y_subset))


time_start = time.time()

# pca = PCA(n_components=50)
# pca_result = pca.fit_transform(X_train)
#
# print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X_train)

print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))

print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))


pca_df = pandas.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

pca_df['pca1'] = pca_result_50[:,0]
pca_df['pca2'] = pca_result_50[:,1]
pca_df['pca3'] = pca_result_50[:,2]
pca_df['pca4'] = pca_result_50[:,3]

top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component
fashion_scatter(top_two_comp.values,y_train)

print('Variance explained per principal component: {}'.format(pca_50.explained_variance_ratio_))

time_start = time.time()


#Try different hyper parameters and conclude
#Perplexity help to have better cluster
fashion_tsne = TSNE(random_state=RS, perplexity=35, learning_rate=200).fit_transform(pca_result_50)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_tsne, y_train)

