# Basic packages and libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json
import pickle
from pandas import Series, ExcelWriter
from IPython.display import HTML
import subprocess as sp

# Modeling packages
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from scipy.optimize import minimize 
import statsmodels.api as sm
from statsmodels.formula.api import ols 
import uncertainties as u
   
class rsm_model:
    def __init__(self, file_path, input_info = None): 
        '''The rsm_model class is used to display response surface link plots.
            There should be the following functions: 
            1. Read data and display stats about distributions of data,
            2. Linear model building and parity,
            3. Predicted surface if 3 variables or less,
            4. Sensitivity for feature importance and errors for each,
            5. Optimization based on whether you want minimum or maximum
            
            The input_info is either a file path or a tuple of the form ([independent variables],
            [dependent variables], [[center, deviation],[center, deviation]], 
            design from pyDOE2)
            '''
        self.result_path = file_path

        if isinstance(input_info, str):
            with open(input_info, 'rb') as f:
                i = pickle.load(f)
                input_info = (i['var_names'], 
                              i['tar_name'], 
                              i['center_dev'],
                              i['design'])

        self.comps = input_info[0]
        self.targets = input_info[1]
        self.center_dev = np.array(input_info[2])
        self.start_range = self.center_dev[:,0]-self.center_dev[:,1]
        self.end_range = self.center_dev[:,0]+self.center_dev[:,1]
        self.design = input_info[3]
        self.df = pd.read_csv(file_path, index_col = 0)
        self.df.dropna(axis=0, how='all', inplace = True)
        self.y = None
    
    def fit(self, feature_order=2):
        """ Creates a linear model with experimental results. 
            FILENAME is a string. It is the path to the results xlsx file.
            FEATURE_ORDER is an int. This dictates the max polynomial power to generate. 
        """
        dffeat = self.df[self.comps]
        self.feat = dffeat.columns.values
        self.X = dffeat.to_numpy()
        self.X_trans = PolynomialFeatures(feature_order).fit_transform(self.X)
        self.y = self.df[self.targets]

        #assign names to features 
        self.x_map =  [f'x{str(x)}' for x in range(len(self.X_trans[0]))]
        p = PolynomialFeatures(2).fit(dffeat)
        self.map_feat = dict(zip(self.comps, self.x_map))

        #poly = PolynomialFeatures(2) 
        #scaler = StandardScaler()
        #model = LinearRegression().fit(self.X, self.y)
        self.model = sm.OLS(self.y, self.X_trans).fit()
        self.y_pred = self.model.predict(self.X_trans)
        return 
    
    def params(self):
        ''' Returns the model's term coefficients and intercept
        '''
        print('Model Coefficients: ', self.model.params[1:])
        print('Model Intercept: ', self.model.intercept[0])
        return self.model.params
    
    def parity(self):
        ''' Creates parity plot between true values and ols predicted values using the model  paramters
        '''
        pred = self.model.predict(self.X_trans)
        plt.scatter(self.y, pred)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{self.targets} prediction and true values')
        return plt.show()

    def _objective(self, X, sign):
        ''' X is array of composition values
        Used with sklearn's minimize function to find optimal composition
        '''
        # X= np.atleast_2d(X)
        # Xp = PolynomialFeatures(2).fit_transform(X)
        # res = self.model.predict(Xp) - self.y
        # return np.sum(res**2)
        X= np.atleast_2d(X)
        Xp = PolynomialFeatures(2).fit_transform(X)
        return sign*self.model.predict(Xp)
    
    def optimum(self, maximize=True):
        ''' Finds optimum composition values 
        MAXIMIZE: True or False. If True, expect to maximize the target variable. 
          If False, expect to minimize the target variable. Default True
        '''
        bounds = []
        max_bound = []
        mid = []
        for i in self.comps:
            min_val = self.df[i].min()
            max_val = self.df[i].max()
            bounds.append((min_val, max_val))
            mid.append((max_val+min_val)/2)
            max_bound.append(max_val)
        if maximize == True:
            sign = -1
        elif maximize == False:
            sign = 1
        opt = minimize(self._objective, mid, sign, bounds=bounds)

        if (opt.x == np.array(max_bound)).all():
           print(f'Max bound reached for optimal prediction.')

        #print(f'Optimal conditions reached at {opt.x} of {self.feat}')
        return {f'{self.targets}': sign*opt.fun, f'composition': opt.x}
    
    def pred_unc(self, data_point):
        '''gets prediction uncertainty to 95% confidence for a data point 
        data point takes an array'''
        unc = []
        for i, j in zip(self.model.params, self.model.bse):
            unc.append([u.ufloat(i, j)])
        unc = np.array(unc)
        x = data_point.reshape(1, -1)
        x_t = PolynomialFeatures(2).fit_transform(x)[0]
        return np.dot(x_t, unc)
    
    def residual_plots(self):
        '''residual_plots plots a historgram of residuals from '''
        fig, axs = plt.subplots(1,3, figsize = (14, 5))
        axs[0].hist(self.model.resid, bins = 4)
        axs[0].set_title('Histogram of Residuals')
        sm.qqplot(self.model.resid, ax = axs[1], line = '45')
        axs[1].set_title('qq plot')
        axs[2].boxplot(self.model.resid, vert = False)
        axs[2].set_title('Box plot of Residuals')
        return fig.show()
    
    def p_values(self):
        '''
        P Values of the Coefficients 
        If the P values are less than 0.05, they are significant. 
        If none of the values are less than 0.05, the terms that have the least p values 
        are most significant.  
        '''
        pvalues = self.model.pvalues
        s = []
        sig = None
        if pvalues.min()>0.05:
           s+= [f'{self.x_map[pvalues.argmin()]} has the most significant P value of {pvalues.min():.{2}}']     
        s+=['P values are as follows where 1 is constant:'] 
        for i in range(len(pvalues)):
            s +=[f'{self.x_map[i]}: \t{pvalues[i]:.{2}}']
            if pvalues[i]<0.05:
              sig=f'{self.x_map[i]} has a P value of {pvalues[i]:.{2}} indicating that it is significant'
        if sig is not None:
            s +=[sig]
        string = '\n'.join(s)
        return string


    def plot_surface(self, design_name, indDic, conDic = {}, increment = 50):
        '''Plot surface plots two variables against each other while holding
        a third variable constant. A 3D plot and a 2D plot should render with
        the target variable indicting the color of the graph. 

        indDic is a dictionary for the dependent variables. The keys of the 
            dictionary must be of the same form as in the design.csv file

        conDic is a dictionary for the variables that you wish to keep 
            constant. Again the keys of this dictionary must be equivalent to
            the independent variable names in the design.csv file.
            
        increment is the number of data points to include between the low 
            and high range of the desired variables. Default 50
        '''
        indVar = list(indDic.keys())         # get indepenent variable names
        conVar = list(conDic.keys())         # get dependent variable names
        
        x1 = np.linspace(indDic[indVar[0]][0], indDic[indVar[0]][1], increment)  #enumerate range
        x2 = np.linspace(indDic[indVar[1]][0], indDic[indVar[1]][1], increment)  # enumerate range
        X1, X2 = np.meshgrid(x1, x2)      # create meshgrid of independent variables
        X1 = X1.flatten()
        X2 = X2.flatten()

        meshD = {indVar[0]:X1, indVar[1]:X2}     #dictionary for independent variables and their vectors
        
        if conDic is not None:
            X3 = np.ones(len(X1))* conDic[conVar[0]][0]    #vector for constant
            meshD[conVar[0]] =X3 
        
        # Make an x array that has x1, x2, x3 in the order that matches model
        x_array = np.empty((0,len(X1)), int)
        for i in self.map_feat:
            x_array = np.append(x_array, [meshD[i]], axis = 0)
        
        # transform the data and make predictions
        x_tran = PolynomialFeatures(2).fit_transform(x_array.T)  
        par = self.model.params
        y_pred = np.dot(x_tran, par, out = None)
                                    
        X1 = X1.reshape(increment, increment)
        X2 = X2.reshape(increment, increment)
        y_pred = y_pred.reshape(increment, increment)

        # Create 3D plot
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(221, projection = '3d')
        surf = ax.plot_surface(X1, X2, y_pred, rstride=1, cstride=1, 
                                cmap=plt.cm.viridis,linewidth=0, antialiased=False)
        ax.set_title(f'{design_name} 3D heatmap')
        ax.set_xlabel(indVar[0])
        ax.set_ylabel(indVar[1])
        ax.set_zlabel('y')
        ax.set_xticks(np.linspace(indDic[indVar[0]][0], indDic[indVar[0]][1], 6))
        ax.set_yticks(np.linspace(indDic[indVar[1]][0], indDic[indVar[1]][1], 6))
        #fig.savefig(f'{design_name}_surface.png')

        # Create 2D plot
        ax1 = fig.add_subplot(222)
        cp = ax1.contourf(X1, X2, y_pred)
        ax1.set_title(f'{design_name} 2D heatmap')
        ax1.set_xlabel(indVar[0])
        ax1.set_ylabel(indVar[1])
        fig.colorbar(surf, shrink=0.5, aspect=10)
        fig.tight_layout()
        plt.show()

        return

    def __repr__(self):
        ''' Representation
          Model Summary of Stats. 
          Provides P Values of the Coefficients 
          If the P values are less than 0.05, they are significant. 
          If none of the values are less than 0.05, the terms that have the least p values 
          are most significant.  
        ''' 
        s = [f'Directory name with design csv: {self.result_path}']
        s += [f'Solutions: {self.comps}']
        for i in range(len(self.comps)):
            s+=[f'{self.comps[i]}: {self.start_range[i]:.{2}} to {self.end_range[i]:.{2}}']
        s+= [f'Total Experiments: {len(self.df)}']
        s = '\n'.join(s)

        if not self.result_path:
            s += f'\nExperimental Design \n{self.df.to_string()}'


        if self.y:
            s += f'\nExperimental Design \n{self.df.to_string()}'
            #Mapping of labels to x values 
            sm = []
            sm +=['Mapping of the x labeled features is as follows:']
            for i in self.map_feat:
                sm+=[f'{i}: \t{self.map_feat[i]}']

            #P values 
            sm += ['\nP values less than 0.05 are significant']
            pvalues = self.model.pvalues
            for i in range(len(pvalues)):
               sm +=[f'{self.comps[i]}: \t{pvalues[i]:.{2}}']

            #R Squared value
            sm += [f'Model R Squared Values: {self.model.rsquared:.{2}}']

            # Optimum Prediction 
            sm += [f'\n Optimum prediction occurs at {(self.optimum())} for self.comps']
            sm = '\n'.join(sm)
            s = s+ sm
        return s


    def _repr_html_(self):
        '''HTML representation''' 
        s = f'<br>Report<br>'
        s += f'<br>Directory name with design csv: {self.result_path}'
        s += f'<br>Solutions: {self.comps}'
        for i in range(len(self.comps)):
            s += f'<br></pre>{self.comps[i]}: {self.start_range[i]:.{2}} to {self.end_range[i]:.{2}}</pre>'
        s += f'<br>Total Experiments: {len(self.df)}<br>'

        if not self.result_path:
            s += f'<br> Experimental Design <br>'
            s += self.df.to_html()
        
        if self.y:
            s += f'<br> Data <br>'
            s += self.df.to_html()
            df_feat = pd.DataFrame({})
            df_feat['coefficients'] = self.model.params
            df_feat['pvalues'] = self.model.pvalues
            df_feat['features'] = self.feat_names
            df_feat.set_index('features', inplace = True)

            #R Squared value
            s += f'<br> Model R Squared Value:<br>{self.model.rsquared:.{2}}<br>'

            # Optimum Prediction 
            s += f'<br> Optimum Prediction<br> {(self.optimum())} for {self.comps}<br>'

            #P values 
            s += f'<br>P values less than 0.05 are significant<br>'
            s += df_feat.to_html()
            self.residual_plots()
        return s

