# run this test with : python -m unittest randomscm/tests/test_feature_importances.py 

import numpy as np
from sklearn.model_selection import GridSearchCV
from unittest import TestCase
from pyscm import SetCoveringMachineClassifier

class PySCMTests(TestCase):

    def test_feature_importance(self):
        """
        Test compatibility with sklearn GridSearchCV function
        """
        n_samples = 1000
        X1 = np.random.rand(n_samples,5) # (samples, features)
        X2 = np.random.rand(n_samples,5) # (samples, features)
        y = np.random.randint(2, size=n_samples)
        X = np.c_[X1, y, X2] # a dataset with 5 noise features + a feature equal to y + 5 noise features

        model = SetCoveringMachineClassifier()

        try:
            model.fit(X, y)
            pred = model.predict(X)
            np.testing.assert_almost_equal(actual=sum(pred == y) / len(y) , desired=1.0)
        except Exception as e:
            self.fail("error with feature importance calculation")

        

if __name__ == '__main__':
    do_tests = PySCMTests()
    do_tests.test_feature_importance()
    print("Every test for pyscm passed ! ")