# Import packages
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

###########################################
##       Reading MNIST Dataset           ##
###########################################

data = pd.read_csv(r"train.csv")
X = data.drop(labels='label', axis=1)  # data
y = data['label']  # labels

# Data preprocessing
standardized_X = StandardScaler().fit_transform(X)
print(standardized_X.shape)  # Shape should be same as shape of X

###########################################
##       Checking a random image         ##
###########################################
# Checking the value at index 100
plt.figure(figsize=(7, 7))
idx = 100

grid_data = X.iloc[idx].values.reshape(28, 28)
plt.imshow(grid_data, cmap="gray")  # Seems to be digit 9
print(y.iloc[idx])

####################################################
## Principal component analysis with 2-D Vizual   ##
####################################################

# Computing covariance matrix
covar_matrix = np.matmul(standardized_X.T, standardized_X)
print(covar_matrix.shape)  # Should be a square matrix of dimension 784

# Computing eigen value and eigen vectors
values, vectors = eigh(covar_matrix, eigvals=(782, 783))  # Taking top 2 eigen values because we are trying to project 784 dimensions in 2D frame
vectors = vectors.T

new_coordinates = np.matmul(vectors, standardized_X.T)

# Appending labels to new coordinates
new_coordinates = np.vstack((new_coordinates, y)).T

# Converting to dataframe
pca_df = pd.DataFrame(data=new_coordinates, columns=('1st principal', '2nd principal', 'label'))

###################################################################
##   Principal component analysis- required principal components ##
###################################################################

pca = PCA()
pca.n_components = 784  # Considering all the components to identify explained variance
pca_data = pca.fit_transform(standardized_X)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)
cum_var_explained = cum_var_explained.T

explained_var_data = pd.DataFrame({
                                    'Components': [i for i in range(1, 785)],
                                    'Explained Variance': cum_var_explained.flatten()
                                  })

###################################################################
##       t-distributed Stochastic Neighbor Embedding             ##
###################################################################

model = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=5000)
tsne_data = model.fit_transform(standardized_X)
tsne_data = np.vstack((tsne_data.T, y)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dimension 1', 'Dimension 2', 'label'))

###################################
##      Writing to excel         ##
###################################

with pd.ExcelWriter('PcaData.xlsx') as writer:
    pca_df.to_excel(writer, sheet_name='pca_viz_data', index=False)
    explained_var_data.to_excel(writer, sheet_name='pca_explained_var', index=False)
    tsne_df.to_excel(writer, sheet_name='tsne_data', index=False)

#----------------------------------------------------------------------------------------------------------------------#
