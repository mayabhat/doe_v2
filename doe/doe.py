'''
The design module is used to generate a design either manually or from an existing DOE design. 
PATH is a directory location for generated files
COLUMN_HEADERS is required as the components you are interested in studying

In make_design.from_array, the user supplies a design array. 
In make_design.from_des, if given a range or values, make_design will map values of -1 to 1 
to desired values (ex. concentrations, temperature levels, volumes). 
'''
# Basic packages and libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
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
from pyDOE2 import *

# widgets
import ipywidgets as widgets
import functools

class inputs:
    '''The inputs class prompts users with questions in which they can input their 
    expeirmental parameters. The load_state and save_state functions calls from a 
    json file to maintain persistent information in the notebook. '''

    def __init__(self):
        self.state = self._load_state()

        self.questions = [
            {
                "question":  'How many experiments are you willing to complete?',
                "type": "num_exp",
                "answer": []
            },
            {
                  "question": 'How many variables do you have?',
                "type": "num_var",
                "answer": []
            },
            {
                "question": 'Indicate your dependent variable names separated with a comma.',
                "type": "var_name",
                "answer": []
            },
            {
                'question': 'Indicate your independent variable name(s).',
                "type": "tar_name",
                'answer': []
            },
            {
            'question': 'Are your variables continuous or discrete?',
                "type": "var_type",
                'answer': ['Continuous', 'Discrete']           
            }, 
            {'question': 'For each variable, input the low values of the desired ranges separated by a comma.',
            "type": "low_range",
                'answer': []
            },
            {'question': 'For each variable, input the high values of the desired ranges separated by a comma.',
            "type": "high_range",
                'answer': []
            }
        ]
        self.out = widgets.Output()
        self.jsonpath = 'files-for-notebook/state.json'

        for i, q in enumerate(self.questions):
            pieces = [ widgets.HTMLMath(q['question']),
                        widgets.Text(layout={"width": "max-content"},  
                                        description="Answer:",
                                        value=self.state.get(str(i), None))
                      ]
            
            pieces[1].observe(functools.partial(self._on_value_change, qid=i), names="value")
            display(widgets.VBox(pieces))
        return

    def save_inputs(self):
        # load inputs into variables
        if os.path.exists(self.jsonpath):
            data = json.load(open(self.jsonpath))
        else:
            return

        labels = ['num_exp', 'var_num', 'var_names', 'tar_name', 'var_type', 'low_vals', 'high_vals']
        keyList = list(data.keys())
        variables = {}
        for i in range(len(data)):
            variables[labels[i]] = data[keyList[i]]

        # Get dependent variable ranges
        low = [int(i.strip()) for i in variables['low_vals'].split(',')]
        high =  [int(i.strip()) for i in variables['high_vals'].split(',')]
        ranges_2D = np.array([low, high])

        # convert ranges from low and high points to center and deviation from center
        center = (ranges_2D[1] - ranges_2D[0])/2 + ranges_2D[0]
        dev = (ranges_2D[1] - ranges_2D[0])/2
        center_dev = np.vstack((center, dev)).T.tolist()

        # save variable names
        var_names = [x.strip() for x in variables['var_names'].split(',')]
        target_names = [x.strip() for x in variables['tar_name'].split(',')]

        if int(variables['var_num'])<5:
            print('Use Surface Response')
            design = ccdesign(int(variables['var_num']), center=(0, 1), face = 'cci')
        elif int(variables['var_type'] == 'discrete'):
            print('Use Latin Hyper Cube')
            design = lhs(int(variables['var_num']), 
                          samples = int(variables['num_exp']), 
                          criterion='center')
        return (var_names, target_names, center_dev, design)

    def _load_state(self):
        if os.path.exists(self.jsonpath):
            return json.load(open(self.jsonpath))
        else:
            return {}

    def _save_state(self, state):
        with open(self.jsonpath, "w") as f:
            f.write(json.dumps(state))


    def _on_value_change(self, change, qid):
        self.state[qid] = change["new"]
        self._save_state(self.state)
        print('saved state')
        with self.out:
            print(self.state)


class designs:
    def __init__(self, path, variable_names, target_names):
        """ PATH is a string that is a directory that will contain experiment files.
            VARIABLE_NAMES is a list of strings with independent variables.
            TARGET_NAMES is a list of strings with target variable names.
        """
        self.path = path
        if not os.path.exists(self.path):
           os.mkdir(self.path)
        self.comps = variable_names
        self.design = None
        self.start_range = np.zeros(len(self.comps))
        self.end_range = np.zeros(len(self.comps))
        self.targets = target_names
        self.df = pd.DataFrame(columns = self.comps + self.targets)
        self.result_path = None

    def _transform_des (self, des, range=None):
        '''Turning design from -1 to 1 to values that reflect experiment (ex. Concentration values)
        '''
        if range != None :
            map1 = np.array(range)
            map = np.array(range)
            map[:,0] = map1[:,0] - map1[:,1]
            map[:,1] = map1[:,0] + map1[:,1]
            self.end_range = map[:,1]
            self.start_range = map[:,0]
            des = (des + np.max(des)) * (map[:, 1] - map[:, 0]) / 2 + map[:, 0] 
        return des

    def make_design (self, design, range=None):
        '''Creates a design dataframe from a design. This requires a mapping to 
          turn -1, 0, 1 into concentrations.
          DESIGN: doe.bbdesign(), doe.ccdesign(), doe.fullfact() from pydoe2. 
            See documentation to figure out how to make a design https://pythonhosted.org/pyDOE/.
          RANGE: 2D array with each row being an independent variable. The first column is 
            center point, second column is distance from center. The form is
            [[center point, deviation], [center point, deviation]] 
        '''
        #create design
        self.design_unscaled = design
        
        if range != None:
          self.design = self._transform_des(self.design_unscaled, range)
        else:
          self.design = self.design_unscaled
        
        self.df = pd.DataFrame(columns = self.comps, data = self.design)
        self.df[self.targets] = ""
        #writer = ExcelWriter(self.path+'/design.xlsx')
        #self.df.to_excel(writer, index = False)
        #writer.save()
        self.df.to_csv(self.path+'/design.csv')
        return self.df

    def fit(self, file_name, feature_order=2):
        """ Creates a linear model with experimental results. 
            FILENAME is a string. It is the path to the results xlsx file.
            FEATURE_ORDER is an int. This dictates the max polynomial power to generate. 
        """
        self.result_path = os.path.join(self.path, file_name)
        self.dfmod = pd.read_excel(self.result_path, index_col = None)

        dffeat = self.dfmod[self.comps]
        self.feat = dffeat.columns.values
        self.X = dffeat.to_numpy()
        self.X_trans = PolynomialFeatures(feature_order).fit_transform(self.X)
        self.y = self.dfmod[self.targets]

        #assign names to features 
        self.x_map =  [f'x{str(x)}' for x in range(len(self.X_trans[0]))]
        p = PolynomialFeatures(2).fit(dffeat)
        self.feat_names = p.get_feature_names_out(dffeat.columns)
        self.map_feat = dict(zip(self.feat_names, self.x_map))

        #poly = PolynomialFeatures(2) 
        #scaler = StandardScaler()
        #model = LinearRegression().fit(self.X, self.y)
        self.model = sm.OLS(self.y, self.X_trans).fit()
        self.y_pred = self.model.predict(self.X_trans)
    
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
            min = self.df[i].min()
            max = self.df[i].max()
            bounds.append((min, max))
            mid.append((max+min)/2)
            max_bound.append(max)
        if maximize == True:
            sign = -1
        elif maximize == False:
            sign = 1
        opt = minimize(self._objective, mid, sign, bounds=bounds)

        if (opt.x == np.array(max_bound)).all():
           print(f'Max bound reached for optimal prediction.')

        #print(f'Optimal conditions reached at {opt.x} of {self.feat}')
        return {f'{self.targets}': sign*opt.fun[0], f'Compositions': opt.x}
    
    def residual_plots(self):
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
           s+= [f'{self.feat_names[pvalues.argmin()]} has the most significant P value of {pvalues.min():.{2}}']     
        s+=['P values are as follows where 1 is constant:'] 
        for i in range(len(pvalues)):
            s +=[f'{self.feat_names[i]}: \t{pvalues[i]:.{2}}']
            if pvalues[i]<0.05:
              sig=f'{self.feat_names[i]} has a P value of {pvalues[i]:.{2}} indicating that it is significant'
        if sig != None:
            s +=[sig]
        string = '\n'.join(s)
        return string

    def __repr__(self):
        ''' Representation
          Model Summary of Stats. 
          Provides P Values of the Coefficients 
          If the P values are less than 0.05, they are significant. 
          If none of the values are less than 0.05, the terms that have the least p values 
          are most significant.  
        ''' 
        s = [f'Directory name with design xls: {self.path}']
        s += [f'Solutions: {self.comps}']
        for i in range(len(self.comps)):
           s+=[f'{self.comps[i]}: {self.start_range[i]:.{2}} to {self.end_range[i]:.{2}}']
        s+= [f'Total Experiments: {len(self.df)}']
        s = '\n'.join(s)

        if not self.result_path:
            s += f'\nExperimental Design \n{self.df.to_string()}'


        if self.result_path:
            s += f'\nExperimental Design \n{self.dfmod.to_string()}'
            #Mapping of labels to x values 
            sm = []
            sm +=['Mapping of the x labeled features is as follows:']
            for i in self.map_feat:
                sm+=[f'{i}: \t{self.map_feat[i]}']

            #P values 
            sm += ['\nP values less than 0.05 are significant']
            pvalues = self.model.pvalues
            for i in range(len(pvalues)):
               sm +=[f'{self.feat_names[i]}: \t{pvalues[i]:.{2}}']

            #R Squared value
            sm += [f'Model R Squared Values: {self.model.rsquared:.{2}}']

            # Optimum Prediction 
            sm += [f'\n Optimum prediction occurs at {(self.optimum())} for self.comps']
            sm = '\n'.join(sm)
            s = s+ sm
        return s

    def get_url(self):
        '''Return an HTML URL to the PATH. This may not be fully functional, GDrive uses
        a lot of different paths and for different purposes, e.g. to view, download,
        etc.
        '''
        path = self.path
        _id = sp.getoutput(f"xattr -p 'user.drive.id' \"{path}\"")

        if os.path.isdir(path):
            url = f'https://drive.google.com/drive/u/0/folders/{_id}'
        elif path.endswith('ipynb'):
            url = f'https://colab.research.google.com/drive/{_id}'
        else:
            url = f'https://drive.google.com/file/d/{_id}'

        return HTML(f'<a href="{url}" target="_blank">{path}</a>')

    def _repr_html_(self):
        '''HTML representation''' 
        url = self.get_url()
        s = f'''<a href="{url}" target ="_blank">{self.path}</a>'''
        s += f'<br>Report<br>'
        s += f'<br>Directory name with design xls: {self.path}'
        s += f'<br>Solutions: {self.comps}'
        for i in range(len(self.comps)):
            s += f'<br></pre>{self.comps[i]}: {self.start_range[i]:.{2}} to {self.end_range[i]:.{2}}</pre>'
        s += f'<br>Total Experiments: {len(self.df)}<br>'

        if not self.result_path:
            s += f'<br> Experimental Design <br>'
            s += self.df.to_html()
            
        if self.result_path:
            s += f'<br> Data <br>'
            s += self.dfmod.to_html()
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

