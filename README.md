### Machine Learning Algorithms from Scratch

Ongoing effort to create common machine learning algorithms from scratch in Python.

```Python
class PCA:
    """
    Class for implementing Principal Component Analysis
    """
    def __init__(self, n_components = None):
        """ initialize an instance of PCA

        Parameters
        ----------
        n_components: int representing number
            of components to keep.  If == None,
            keep all components
        """
        self.n_components = n_components

    def standardize(self, data):
        """ standardize input data

        Parameters
        ----------
        data: numpy array

        Returns
        -------
        standardized numpy array
        """
        sc = StandardScaler()
        self.std = sc.fit_transform(data)
        return self.std

    def covariance_matrix(self, data):
        """ computes covariance matrix

        Parameters
        ----------
        data: numpy array

        Returns
        -------
        covariance matrix of transposed
        numpy array
        """
        return np.cov(data.T)

    def explained_variance(self, values):
        """ computes explained variance by component

        Parameters
        ----------
        values: eigenvalues, computed in
            projection_matrix method

        Returns
        -------
        individual explained variance ratio
        cumulative explained variance ratio
        """
        tot = sum(values)
        var_exp_ratio = [(i / tot) for i in sorted (values, reverse = True)]
        cum_exp_ratio = np.cumsum(var_exp_ratio)
        return var_exp_ratio, cum_exp_ratio

    def projection_matrix (self, data):
        """ construct projection matrix

        Parameters
        ----------
        data: covariance matrix

        Returns
        -------
        projection matrix
        """
        #calculate eigen values and eigen vectors
        #combine to set eigen pairs
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(data)
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]),
                    self.eigen_vecs[:,i]) for i in range(len(self.eigen_vals))]
        self.eigen_pairs.sort(reverse=True)

        #determine number of components to keep
        if self.n_components == None or self.n_components > len(self.eigen_pairs):
            self.w = self.eigen_pairs[0][1][:,np.newaxis]
            for i in range(len(self.eigen_pairs) - 1):
                self.w = np.hstack((self.w, (self.eigen_pairs[i+1][1][:, np.newaxis])))
        else:
            self.w = self.eigen_pairs[0][1][:,np.newaxis]
            for i in range(self.n_components - 1):
                self.w = np.hstack((self.w, (self.eigen_pairs[i+1][1][:, np.newaxis])))
        return self.w

    def transformation(self, data):
        """ determine and select # pc's, then transform input data

        Parameters
        ----------
        data: input numpy array

        Returns
        -------
        transformed numpy array
        """
        std_data = self.standardize(data)
        cov_mat = self.covariance_matrix(std_data)
        w = self.projection_matrix(cov_mat)
        transformed = std_data.dot(w)
        return transformed
```
