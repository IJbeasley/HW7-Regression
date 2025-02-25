"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np

# Our regression module
from regression import logreg, utils

# Use scikit-learn to check the correctness of our model
from sklearn.metrics import log_loss

# (you will probably need to import more things here)

def test_prediction():
	pass

def test_loss_function():
	"""
        Unit test to check that loss estimated by the regression module's loss_function is being calculated correctly,
	by comparing it to scikit learn's log_loss function using true y values from data/nsclc.csv, and randomly generated y predictions. 
        """

	# Load only y_train as our y_true - X is not needed for this test
         _, _, y_true, _ = utils.loadDataset(
                features=[
                  'Penicillin V Potassium 500 MG',
	          'Computed tomography of chest and abdomen',
		],
                   split_percent=0.8,
                   split_seed=42
                                                  )

        # Create a y_pred array, of random 0s and 1s
        # Set seed for reproducibility
        np.random.seed(42)
        y_pred = np.random.randint(0, 2, size=y_true.shape)

        # Calculate loss using loss_function from regression module
        lr_mod = LogisticRegressor(num_feats = 2)
        lr_mod.loss_function(y_true, y_pred)

        # Calculate loss using scikit learn's log_loss function
        true_loss=log_loss(y_true, y_pred)

        assert np.isclose(logreg_loss, sklearn_loss), 'Calculated loss by loss_function is not correct'


def test_gradient():
	pass

def test_training():
	pass
