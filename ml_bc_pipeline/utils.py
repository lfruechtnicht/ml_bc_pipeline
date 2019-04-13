import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from scipy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

class CustomScaler(TransformerMixin):
    def __init__(self, continuous_idx, dummies_idx):
        self.continuous_idx = continuous_idx
        self.dummies_idx = dummies_idx
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[:, self.continuous_idx])
        return self

    def transform(self, X, y=None, copy=None):
        X_head = self.scaler.transform(X[:, self.continuous_idx])
        return np.concatenate((X_head, X[:, self.dummies_idx]), axis=1)


df = pd.DataFrame

def mahalanobis_r_pd(X,mean,S_inv):
    data = []
    for i in range(X.shape[0]):
        data.append(mahalanobis(X.iloc[i,:],mean,S_inv) ** 2)
    ser_ = pd.Series(data, X.index.values)
    return(ser_)


def qqplot(data):
    '''takes continuous data and returns qq plot https://en.wikipedia.org/wiki/Q–Q_plot'''

    df = data
    n = df.shape[0]
    p = df.shape[1]

    S = np.cov(df.T)
    S_inv = inv(S)

    mean = df.mean(axis=0)

    d_squared = mahalanobis_r_pd(df, mean, S_inv)

    d_squared = d_squared.sort_values()
    quantiles = np.linspace(0.5, n - 0.5, n) / n

    x = chi2.ppf(quantiles, p)
    plt.scatter(d_squared, x)
    plt.title('QQ plot for Multivariable normality')
    plt.xlabel('Squared Mahalanobis distances')
    plt.ylabel('Chi-squared quantiles')
    plt.plot(x, x, color='r')
    plt.show()


def stal_plot(df, disp=False):
    '''The stalactite plot for the detection of multivariate outliers. Atkinson, A.C. & Mulira, HM. Stat Comput (1993) 3: 27. https://doi.org/10.1007/BF00146951'''


    # needs index initialization
    n = df.shape[0]
    p = df.shape[1]
    i = np.asarray(range(0, n))

    pd.set_option('display.max_rows', n - p)
    pd.set_option('display.max_columns', n)

    thresh = chi2.ppf((n - 0.5) / n, p)

    ind = np.zeros((n - p, n))
    ind_1 = 0

    sample = np.random.choice(i, p + 1, replace=False)  # first randomly sample n+1 for the first sample mean

    x_mean = df.iloc[sample].mean()
    S = df.iloc[sample].cov()
    # print(S.shape)
    S_inv = inv(S)
    M = mahalanobis_r_pd(df, x_mean, S_inv)
    ind_2 = i[M > thresh]
    ind[ind_1, ind_2] = 1

    for e in (range(p + 2, n + 1)):
        # print(e)
        ind_1 += 1
        # print(ind_1)
        x_mean = df.loc[M.nlargest(e).index.values].mean()
        S = df.loc[M.nlargest(e).index.values].cov()
        # print(S.shape)
        S_inv = inv(S)
        M = mahalanobis_r_pd(df, x_mean, S_inv)
        ind_2 = i[M > thresh]
        # print(ind_2)
        ind[ind_1, ind_2] = 1

    out_ind = ind_2
    plot = pd.DataFrame(ind)
    if disp == True:
        plot_ = plot.replace(0, ' ')
        plot_ = plot_.replace(1, '*')
        print(plot_)

    format_dictionary = {'outl': int(sum(plot.iloc[-1])), 'index': out_ind}
    if sum(plot.iloc[-1]) == 0:
        print('The stalactite plot found no strange observations')
    else:
        print('The stalactite plot found {outl} strange observations.\nIndexes{index} are suspicious. '.format(
            **format_dictionary))
    return out_ind
