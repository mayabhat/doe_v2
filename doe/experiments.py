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
import json
import sys
import pickle
from IPython.display import HTML
from pyDOE2 import *
import statsmodels.api as sm

# widgets
import ipywidgets as widgets
import functools

class inputs:
    '''The inputs class prompts users with questions in which they can input their 
    expeirmental parameters. The load_state and save_state functions calls from a 
    json file to maintain persistent information in the notebook. '''

    def __init__(self):
        self.jsonpath = 'files-for-notebooks/state.json'
        self.state = self._load_state()

        self.questions = [
            {
                "question":  '1. How many experiments are you willing to complete?',
                "type": "num_exp",
                "answer": []
            },
            {
                  "question": '2. How many variables do you have?',
                "type": "num_var",
                "answer": []
            },
            {
                "question": '3. Indicate your dependent variable names separated with a comma.',
                "type": "var_name",
                "answer": []
            },
            {
                'question': '4. Indicate your independent variable name(s).',
                "type": "tar_name",
                'answer': []
            },
            {
            'question': '5. Are your variables continuous or discrete?',
                "type": "var_type",
                'answer': ['Continuous', 'Discrete']           
            }, 
            {'question': '6. For each variable, input the low values of the desired ranges separated by a comma.',
            "type": "low_range",
                'answer': []
            },
            {'question': '7. For each variable, input the high values of the desired ranges separated by a comma.',
            "type": "high_range",
                'answer': []
            }
        ]
        self.out = widgets.Output()
    

        for i, q in enumerate(self.questions):
            pieces = [ widgets.HTMLMath(q['question']),
                        widgets.Text(layout={"width": "max-content"},  
                                        description="Answer:",
                                        value=self.state.get(str(i), None))
                      ]
            
            pieces[1].observe(functools.partial(self._on_value_change, qid=i), names="value")
            display(widgets.VBox(pieces))

        self.save_inputs()
        return

    def save_inputs(self):
        '''Save inputs takes the file from the inputs function and saves 
        them in variables that can be accessed in another function. It allows us to 
        assign a variable to then use later in create_experiments'''

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

        if (len(low)==len(high)==len(variables['var_names'])== int(variables['var_num'])) == False:
            print('The inputs of Questions 2, 3, 6, and 7 should have the same number of elements')
            sys.exit()

        # convert ranges from low and high points to center and deviation from center
        center = (ranges_2D[1] - ranges_2D[0])/2 + ranges_2D[0]
        dev = (ranges_2D[1] - ranges_2D[0])/2
        center_dev = np.vstack((center, dev)).T.tolist()

        if int(variables['var_num'])<5:
            print('Use Surface Response')
            design = ccdesign(int(variables['var_num']), center=(0, 1), face = 'cci')
        elif int(variables['var_type'] == 'discrete'):
            print('Use Latin Hyper Cube')
            design = lhs(int(variables['var_num']), 
                          samples = int(variables['num_exp']), 
                          criterion='center')
        # save variable names
        variables['var_names'] = [x.strip() for x in variables['var_names'].split(',')]
        variables['tar_name'] = [x.strip() for x in variables['tar_name'].split(',')]
        variables['num_exp'] = int(variables['num_exp'])
        variables['var_num'] = int(variables['var_num']) 
        variables['low_vals'] = low
        variables['high_vals'] = high 
        variables['2D_range'] = [low, high]
        variables['center_dev'] = center_dev
        variables['design'] = design

        path_input_file = 'files-for-notebooks/exp_info.pkl'
        if not os.path.exists(path_input_file.split('/')[0]):
            os.mkdir(path_input_file.split('/')[0])
        with open(path_input_file, 'wb') as f:
            pickle.dump(variables, f)
        return

    def _load_state(self):
        if os.path.exists(self.jsonpath):
            return json.load(open(self.jsonpath))
        else:
            return {}

    def _save_state(self, state):
        with open(self.jsonpath, "w") as f:
            f.write(json.dumps(self.state))


    def _on_value_change(self, change, qid):
        self.state[qid] = change["new"]
        self._save_state(self.state)
        print('saved state')
        with self.out:
            print(self.state)


class create_experiments:
    '''Create Experiments takes the arguments from the inputs class and then generates 
        the necessary experiments to run based on the number of variables specified and 
        the corresponding variables names and low and high ranges of them. 
        
        The input_info argument is in the form of ([depenedent variable names], 
        [independent variable names],[[center point, deviation]],design) 
    '''

    def __init__(self, path, input_info = None, input_path = 'files-for-notebooks/exp_info.pkl'):
        """ PATH is a string that is a directory that will contain experiment files.
            VARIABLE_NAMES is a list of strings with independent variables.
            TARGET_NAMES is a list of strings with target variable names.
        """
        # location to save experiment info
        self.path = path
        if not os.path.exists(self.path):
           os.mkdir(self.path)

        # Load inputs from input.json file
        self.input_path = input_path
        if os.path.exists(self.input_path):
            with open(self.input_path, 'rb') as f:
                variables = pickle.load(f)
            input_info = (variables['var_names'],
                            variables['tar_name'],
                            variables['center_dev'],
                            variables['design'])

        self.comps = input_info[0]
        self.targets = input_info[1]
        self.center_dev = input_info[2]
        self.design = input_info[3]
        self.start_range = np.zeros(len(self.comps))
        self.end_range = np.zeros(len(self.comps))
        
        #create dataframe of necessary experiments 
        self.df = pd.DataFrame(columns = self.comps + self.targets)
        self.result_path = None
        self.make_design(self.design)
        return

    def _transform_des (self, des):
        '''Turning design from -1 to 1 to values that reflect experiment (ex. Concentration values)
        '''
        if self.center_dev is not None :
            map1 = np.array(self.center_dev)
            map2 = np.array(self.center_dev)
            map2[:,0] = map1[:,0] - map1[:,1]
            map2[:,1] = map1[:,0] + map1[:,1]
            self.end_range = map2[:,1]
            self.start_range = map2[:,0]
            des = (des + np.max(des)) * (map2[:, 1] - map2[:, 0]) / 2 + map2[:, 0] 
        return des

    def make_design (self, design):
        '''Creates a design dataframe from a design. This requires a mapping to
          turn -1, 0, 1 into concentrations.
          DESIGN: doe.bbdesign(), doe.ccdesign(), doe.fullfact() from pydoe2.\ 
            See documentation to figure out how to make a design https://pythonhosted.org/pyDOE/.
          RANGE: 2D array with each row being an independent variable. The first column is 
            center point, second column is distance from center. The form is
            [[center point, deviation], [center point, deviation]] 
        '''
        #create design
        self.design_unscaled = design
        
        if self.center_dev is not None:
          self.design = self._transform_des(self.design_unscaled)
        else:
          self.design = self.design_unscaled
        
        self.df = pd.DataFrame(columns = self.comps, data = self.design)
        self.df[self.targets] = ""
        self.df.to_csv(self.path+'/design.csv')
        return self.df
    
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

    def _repr_html_(self):
        '''HTML representation''' 
        s = f'<br>Report<br>'
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