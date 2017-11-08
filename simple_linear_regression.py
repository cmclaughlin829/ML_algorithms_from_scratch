import numpy as np
import matplotlib.pyplot as plt

class regression:
    """
    Class for simple linear regression (single predictor)
    """
    def __init__ (self, X, Y):
        """ initialize an instance of regression

        Parameters
        ----------
        X: numpy array of predictor
        Y: numpy array of target variable
        """
        self.X = X
        self.Y = Y

    def fit (self):
        """ fit the regression model

        Returns
        -------
        regression coefficients; b0, b1
        """
        self.cov = np.cov(self.X,self.Y)[0][1]
        self.Xmean = self.X.mean()
        self.Ymean = self.Y.mean()
        self.Xvar = self.X.var()
        self.Yvar = self.Y.var()

        top = 0
        bottom = 0
        for x, y in zip (self.X,self.Y):
            top += (x - self.Xmean)*(y - self.Ymean)
            bottom += (x - self.Xmean)**2
        self.b1 = top / bottom
        self.b0 = self.Ymean - self.b1*self.Xmean

        return self.b0, self.b1

    def predict (self, new_x, new_y):
        """ initialize an instance of regression

        Parameters
        ----------
        new_x: numpy array of predictor
        new_y: numpy array of target variable

        Returns
        -------
        predicted values of target variable
        """
        self.new_x = new_x
        self.new_y = new_y
        self.pred_y = self.b0 + self.b1*self.new_x
        return self.pred_y

    def assess (self):
        """ evaluate performance of model

        Returns
        -------
        coefficient of determination
        """
        RSS = 0
        TSS = 0
        reg_y = self.b0 + self.b1*self.X
        for y, yhat in zip (self.Y, reg_y):
            RSS += (y - yhat)**2
            TSS += (y - self.Ymean)**2
        R2 = 1-(RSS/TSS)
        return R2

    def lin_regplot (self, x_start=-15, x_end=15):
        """
        plots linear regression line along with training and
        test data points
        """
        x = np.linspace(x_start, x_end, 100)
        y = self.b0 + self.b1*x
        plt.scatter(self.new_x, self.new_y, color = 'r', label = 'Test Points')
        plt.scatter(self.X, self.Y, c = 'b', label = 'Training')
        plt.plot (x, y, color = 'b', label = 'Regression Line')
        plt.legend(loc='upper left')
