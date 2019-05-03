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


def set_style():
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    set_style()
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



def call_pca(numb_component, x_data,y_data):
    print("PCA is running...")
    time_start = time.time()

    # PCA then TSNE on Encoded Images
    pca = PCA(numb_component)
    pca_result = pca.fit_transform(x_data)
    print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

    pca_df = pandas.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])
    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]

    top_two_comp = pca_df[['pca1', 'pca2']]  # taking first and second principal component
    fashion_scatter(top_two_comp.values, y_data)
    return pca_result




def call_tsne(perplexity, x_data,y_data):
    print("t-SNE is running...")
    time_start = time.time()
    RS = 123
    #Try different hyper parameters and conclude
    #Perplexity help to have better cluster
    fashion_tsne = TSNE(random_state=RS, perplexity=perplexity).fit_transform(x_data)
    print('t-SNE '+str(perplexity)+' done! Time elapsed: {} seconds'.format(time.time()-time_start))
    fashion_scatter(fashion_tsne, y_data)
    return fashion_tsne



if __name__ == '__main__':

    x_train, y_train = load_mnist('data/', kind='train')
    print(x_train.shape)
    print(y_train)
    x_subset = x_train[0:10000]
    y_subset = y_train[0:10000]
    print(np.unique(y_subset))
    pca_result = call_pca(50, x_subset, y_subset)
    call_tsne(50,pca_result,y_subset)
