## Design of Experiments (DOE) Package
This github repo hosts a simple code for designing experiments, building models, and seeing basic statistical information. The package is aimed to be used to direct experimental or computational work when it comes to exploring multidimensional spaces (ex. formulation chemistries). You may want to install the library with the following:


```python
!pip install git+https://github.com/mayabhat/doe_v2 
```

Once installed, you can import the doe module as follows:


```python
from doe import doe
```

## Creating Design Files 
You can create xlsx files that contain the necessary experiments by creating a design object and specifying the experiment's desired directory, the independent variables, and the expected output variables.


```python
design = doe.doe('test-dir', ['CompA', 'CompB'], ['Target1'])
design
```




<a href="<IPython.core.display.HTML object>" target ="_blank">test-dir</a><br>Report<br><br>Directory name with design xls: test-dir<br>Solutions: ['CompA', 'CompB']<br></pre>CompA: 0.0 to 0.0</pre><br></pre>CompB: 0.0 to 0.0</pre><br>Total Experiments: 0<br><br> Experimental Design <br><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CompA</th>
      <th>CompB</th>
      <th>Target1</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>



There should now be a directory called test-dir if there was not already. To make the design, you will need to call the method make_design() as follows. make_design() takes in a design array and a range. 

The design array can be manually generated or generated using the pyDOE2 functionalities expressed in this package. Designs can be of the form  doe.bbdesign(), doe.ccdesign(), doe.fullfact, or more. See documentation to figure out how to make a design https://pythonhosted.org/pyDOE/. 

pyDOE2 outputs designs from -1 to 1. Range takes center points and distance from center points to convert -1 to 1 to scaled values. The form is [[center point, deviation], [center point, deviation]] where the length of the list is the equivalent to the number of independent variables.

make_design() outputs an xlsx file with the experiments that must be completed. Target values should be inputed as the experiments are completed in the target column. 


```python
d = doe.ccdesign(2, face = 'cci')
range = [[3, 1.5], [10, 8]]
design.make_design(d, range)
design
```




<a href="<IPython.core.display.HTML object>" target ="_blank">test-dir</a><br>Report<br><br>Directory name with design xls: test-dir<br>Solutions: ['CompA', 'CompB']<br></pre>CompA: 1.5 to 4.5</pre><br></pre>CompB: 2.0 to 1.8e+01</pre><br>Total Experiments: 16<br><br> Experimental Design <br><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CompA</th>
      <th>CompB</th>
      <th>Target1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.93934</td>
      <td>4.343146</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.06066</td>
      <td>4.343146</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.93934</td>
      <td>15.656854</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.06066</td>
      <td>15.656854</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.50000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.50000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.00000</td>
      <td>2.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.00000</td>
      <td>18.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td></td>
    </tr>
  </tbody>
</table>



## Building a Model with Results

To build a model with the obtained results, you will need to call the method fit(). Fit takes in a path to a results xlsx file. This file should be of the same format as the design xlsx with independent and dependent variable names as column headings. 

fit() should output a stdout report containing information about the linear model's parameters, p values, r^2, residual plots, and more. 


```python
design.fit('results.xlsx')
design
```




<a href="<IPython.core.display.HTML object>" target ="_blank">test-dir</a><br>Report<br><br>Directory name with design xls: test-dir<br>Solutions: ['CompA', 'CompB']<br></pre>CompA: 1.5 to 4.5</pre><br></pre>CompB: 2.0 to 1.8e+01</pre><br>Total Experiments: 16<br><br> Data <br><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CompA</th>
      <th>CompB</th>
      <th>Target1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.93934</td>
      <td>4.343146</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.06066</td>
      <td>4.343146</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.93934</td>
      <td>15.656854</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.06066</td>
      <td>15.656854</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.50000</td>
      <td>10.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.50000</td>
      <td>10.000000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.00000</td>
      <td>18.000000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>7</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.00000</td>
      <td>10.000000</td>
      <td>3</td>
    </tr>
  </tbody>
</table><br> Model R Squared Value:<br>0.093<br><br> Optimum Prediction<br> {"['Target1']": 4.853553390593341, 'Compositions': array([ 1.5, 18. ])} for ['CompA', 'CompB']<br><br>P values less than 0.05 are significant<br><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
      <th>pvalues</th>
    </tr>
    <tr>
      <th>features</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-2.273667e+00</td>
      <td>0.791929</td>
    </tr>
    <tr>
      <th>CompA</th>
      <td>1.298816e+00</td>
      <td>0.775728</td>
    </tr>
    <tr>
      <th>CompB</th>
      <td>6.158471e-01</td>
      <td>0.413186</td>
    </tr>
    <tr>
      <th>CompA^2</th>
      <td>6.661338e-16</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>CompA CompB</th>
      <td>-1.250000e-01</td>
      <td>0.496331</td>
    </tr>
    <tr>
      <th>CompB^2</th>
      <td>-7.812500e-03</td>
      <td>0.746163</td>
    </tr>
  </tbody>
</table>




    
![png](README_files/README_9_1.png)
    


## Other Functionalities
You can make a parity plot to see how the model's predicted vs experimental values are. If the model is good, these values should fall on a 45 degree x = y line. 


```python
design.parity()
```


    
![png](README_files/README_11_0.png)
    


You can get an optimum prediction by either specifying maximize or minimize with the optimum() function. If you want to maximize the target, indicate maximize = True, if you want to minimize, indicate maximize = False. 


```python
design.optimum(maximize = True)
```




    {'Compositions': array([ 1.5, 18. ]), "['Target1']": 4.853553390593341}



Residual plots can be displayed with residual_plots(). There are a few ways to tell if your model is not good enough. Theoretically, the histogram of your residuals should be normally distributed, and the data points on the qq plot should fall on the 45 degree angle line. If this is not the case, the model may need to be revisted. 


```python
design.residual_plots()

```


    
![png](README_files/README_15_0.png)
    


P values indicate which of the components in the model contribute the greatest to the variance in target values. P values under 0.05 are most significant, and those above 0.05 are not. If none are under 0.05, the next lowest can be considered statistically significant. 


```python
print(design.p_values())
```

    CompB has the most significant P value of 0.41
    P values are as follows where 1 is constant:
    1: 	0.79
    CompA: 	0.78
    CompB: 	0.41
    CompA^2: 	1.0
    CompA CompB: 	0.5
    CompB^2: 	0.75



```python

```
