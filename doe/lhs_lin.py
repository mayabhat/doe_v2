import autograd.numpy as np

class features:
  def __init__(self):
    
  def transform(self, x, poly_feat = 2):
      '''Converst the x values from 1d to 2d with polynomial features
      '''
      x = np.atleast_2d(x)
      x = PolynomialFeatures(poly_feat).fit_transform(x)
      return x

  def design(self, ran, samples = 10, poly_feat = 2):
      '''Creates latin hyper cube design
         RAN: 2D np.array of values where each row is a feature's range
              The first column in the minimum sample value, the second 
              is the maximum sample value
         SAMPLES: integer, number of samples to create, default = 10
         POLY_FEAT: integer, number of polynomial feature power 
      '''
      des = lhs(2, samples = samples, criterion = 'center')
      des = ((0 -np.min(des, axis = 0))+des)*(ran[:,1]-ran[:,0])+ ran[:,0]
      des = np.append(des, np.array([np.mean(ran, axis = 0)]), axis = 0)
      x = self.transform(des, poly_feat)
      return x


from sklearn.linear_model import LinearRegression
class LinReg:
    def __init__(self, model = LinearRegression()):
        self.model = model

    def fit(self, x, y):
        self.x = x
        self.theta = minimize(self.sse, np.mean(x, axis = 0)).x
        self.model = self.model.fit(x, y)
        #self.theta = self.model.coef_
        self.tval = t.ppf(0.975, len(x))
        self.h = hessian(self.sse)(self.theta)                #obtain hessian of sse using autograd
        self.p = self.sse(self.theta)/len(x)*np.linalg.pinv(self.h)      #inverse and scale hessian
        return 
    
    def predict(self, x):
        return self.model.predict(x)

    def _objective(self, X, model):
        ''' X is array of composition values
        Used with sklearn's minimize function to find optimal composition
        '''
        y = y_compute(X[1:3])
        X = np.atleast_2d(X)
        res = model.predict(X) - y
        return np.sum(res**2)

    def _constraint(self, x):
        gprime = elementwise_grad(self.g,0)(self.theta,x)
        uncerts = np.sqrt(gprime @ self.p @ gprime)
        return -1*(uncerts*self.tval-0.1)

    def optimum(self):
        ''' Finds optimum composition values 
        '''
        mid1 = np.array([np.mean(self.x, axis = 0)])
        con = {'type': 'ineq', 'fun': self._constraint}
        opt = minimize(self._objective, mid1, (self.model), constraints = con)
        return opt

    def g(self, theta,x):                                   #suppose we want to fit function g with parameters theta
        return (np.dot(x, theta))

    def sse(self, theta):                                   #sum squared errors objective
        return np.sum((self.g(theta, x) - y)**2)
