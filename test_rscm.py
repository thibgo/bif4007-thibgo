# run this test with : python -m unittest randomscm/tests/test_feature_importances.py 

import numpy as np
from sklearn.model_selection import GridSearchCV
from unittest import TestCase
from randomscm.randomscm import RandomScmClassifier

class RandomSCMTests(TestCase):

    def test_feature_importance(self):
        """
        Test compatibility with sklearn GridSearchCV function
        """
        n_samples = 1000
        X1 = np.random.rand(n_samples,5) # (samples, features)
        X2 = np.random.rand(n_samples,5) # (samples, features)
        y = np.random.randint(2, size=n_samples)
        X = np.c_[X1, y, X2] # a dataset with 5 noise features + a feature equal to y + 5 noise features

        model = RandomScmClassifier()
        model.fit(X, y)

        try:
            importances = model.features_importance()
            assert len(importances) == X.shape[1]
            np.testing.assert_almost_equal(actual=sum(importances), desired=1.0)
            np.testing.assert_almost_equal(actual=max(importances), desired=importances[5])

        except Exception as e:
            self.fail("error with feature importance calculation")
    
    def test_max_features_param(self):
        """
        Test max_features parameter values
        """
        n_samples = 250
        X = np.random.rand(n_samples,800) # (samples, features)
        y = np.random.randint(2, size=n_samples)

        try:
            model = RandomScmClassifier(n_estimators=2, max_features='sqrt')
            model.fit(X, y)
            assert len(model.estim_features[0]) == int(np.sqrt(X.shape[1]))

            model = RandomScmClassifier(n_estimators=2, max_features='log2')
            model.fit(X, y)
            assert len(model.estim_features[0]) == int(np.log2(X.shape[1]))

            model = RandomScmClassifier(n_estimators=2, max_features=None)
            model.fit(X, y)
            assert len(model.estim_features[0]) == 800

            model = RandomScmClassifier(n_estimators=2, max_features='auto')
            model.fit(X, y)
            assert len(model.estim_features[0]) == int(np.sqrt(X.shape[1]))
        
            model = RandomScmClassifier(n_estimators=2, max_features=0.5)
            model.fit(X, y)
            assert len(model.estim_features[0]) == 400

            model = RandomScmClassifier(n_estimators=2, max_features=148)
            model.fit(X, y)
            assert len(model.estim_features[0]) == 148

        except Exception as e:
            self.fail("error with max_features parameter calculation")
        
        try:
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples=0.5)
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples=1)
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples=1.0)
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples=None)
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples='sqrt')
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features='sqrt', max_samples='log2')
            model.fit(X, y)
            
            model = RandomScmClassifier(n_estimators=5, max_features=1, max_samples='auto')
            model.fit(X, y)

        except Exception as e:
            self.fail("error with max_samples parameter calculation")

if __name__ == '__main__':
    do_tests = RandomSCMTests()
    do_tests.test_feature_importance()
    do_tests.test_max_features_param()
    print("Everything passed ! ")
