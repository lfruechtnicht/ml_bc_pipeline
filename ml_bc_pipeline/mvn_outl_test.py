from utils import *

import numpy.random as nr



def main():
    cov = np.array([[1.0, 0.1, 0.1,],
                            [0.1, 1.0, 0.1,],
                            [0.1, 0.1, 1.0]])
    mu = np.log([0.3, 0.4, 0.5])

    mvn = nr.multivariate_normal(mu, cov, size=400)

    mvn = df(mvn)
    mvn.loc[15] = [10000000,2020323891,347123904781]

    qqplot(mvn)
    stal_plot(mvn)

if __name__ == "__main__":
    main()

    #test ,kefj b,ejnwk.ewqjnfeqwk.jnf
