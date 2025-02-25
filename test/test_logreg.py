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
from regression import utils, logreg as reg

# Preprocessing of data/nsclc.csv
from sklearn.preprocessing import StandardScaler
# To create a test set
from sklearn.model_selection import train_test_split

# To compare fitted sklearn model to regression model 
from sklearn.linear_model import LogisticRegression

# Use scikit-learn to check the correctness of our model
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


# def test_prediction():
#     """
# 
#     Unit test to check that prediction is working correctly.
# 
# 	Fit a model with our regression module functions, to data in dataset/data/nsclc.csv.
#   Then estimate the accurarcy of this model on a validation dataset (with scikit learn)
# 	- Compare this model's accurarcy to a scikit learn logistic regression model with the same model coefficents
#   - (?TO POTENTIALLY ADD LATER) Compare its accurarcy to a scikit learn logistic regression model, trained on the same data with saga solver
# 
#   """
# 
#    # Load data
#     X_train, X_val, y_train, y_val = utils.loadDataset(
#       features=[
#                 'Penicillin V Potassium 500 MG',
#                 'Computed tomography of chest and abdomen',
#                 'Plain chest X-ray (procedure)',
#                 'Low Density Lipoprotein Cholesterol',
#                  'Creatinine',
#                  'AGE_DIAGNOSIS'
#                ],
#      split_percent=0.8,
#     split_seed=42
#                                              )
# 
#     # Split validation set into validation and test
#     X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
# 
#     # Scale the data, since values vary across feature. Note that we
#     # fit on the training data and use the same scaler for X_val.
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_val = sc.transform(X_val)
# 
#     # Train/fit logistic regression module using regression module
#     lr_mod = reg.LogisticRegressor(
#                                    num_feats = X_train.shape[1],
#                                    max_iter=50
#                                    )
# 
#     lr_mod.train_model(X_train, y_train, X_val, y_val)
# 
#     # Calculate test set set predictions:
#     test_y_pred = lr_mod.make_prediction(X_test)
# 
#     # Calculate test set prediction accuarcy for regression module fitted model
#     test_y_pred_accuarcy = accuracy_score(test_y_pred, y_test)
# 
#     # Now fit scikit learn model
#     sk_lr_mod = LogisticRegression(solver='saga',
#                                    max_iter=50,
#                                    random_state=42)
# 
#     sk_lr_mod.fit(X_train, y_train)
# 
#     # Manually set feature weights and intercept
#     sk_lr_mod.intercept_ = np.array([lr_mod.W[-1]])  # Last element is bias/intercept
#     sk_lr_mod.coef_  = lr_mod.W[:-1].reshape(1, -1)
# 
#     # Predict on the test set
#     sk_y_pred = sk_lr_mod.predict(X_test)
# 
#     # Calculate test set prediction accuarcy for scikit learn fitted model
#     sk_test_y_pred_accuracy = accuracy_score(y_test, sk_y_pred)
# 
#     # Check: is the accuarcy of predictions from both the sklearn model, and regression module model consistent?
#     assert np.isclose(test_y_pred_accuarcy, sk_test_y_pred_accuracy, rtol = 0.01), "Accuracy of fitted model differs from sklearn"
# 
#     # Check: are the predictions from both the sklearn model, and regression module model consistent?
#     assert np.array_equal(test_y_pred, sk_y_pred), "Predictions of our model differ from sklearn model"
# 
# 

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

    # Create a y_pred array, of numbers between 0 and 1
    # Set seed for reproducibility
    np.random.seed(42)
    y_pred = np.random.random(size=y_true.shape) 
    
    # Calculate loss using loss_function from regression module
    lr_mod = reg.LogisticRegressor(num_feats = 2)
    logreg_loss = lr_mod.loss_function(y_true, y_pred)

    # Calculate loss using scikit learn's log_loss function
    sklearn_loss = log_loss(y_true, y_pred)
    
    assert np.isclose(logreg_loss, sklearn_loss), 'Calculated loss by loss_function is not correct'


# def test_gradient():
#             """
#             Unit test to check that gradient is being calculated correctly with calculate_gradient.
#             Fit a logistic regression model with our regression module functions, to a small subset data in dataset/data/nsclc.csv, 
#             compare the gradient estimated against gradient to calculated by hand. 
#             """
#             # Load data
#             _, X_subsample, _, y_subsample = utils.loadDataset(
#                 features=[
#                     'Penicillin V Potassium 500 MG',
#                     'Computed tomography of chest and abdomen',
#                     'AGE_DIAGNOSIS'
#                 ],
#                 split_percent=0.95,
#                 split_seed=42
#             )
#         
#             # Initialise logistic regression model 
#             lr_mod = reg.LogisticRegressor(
#                                            num_feats = X_train.shape[1], 
#                                            max_iter=50
#                                            )    
#                                        
#             # Train/fit logistic regression module using regression module
#             lr_mod.train_model(X_subsample, y_subsample)
#         
#             # Calculate gradient with calculate_gradient
#             est_grad = lr_mod.calculate_gradient(y_subsample, X_subsample)
#         
#             # TO DO: Calculate gradient by hand
#             true_grad = np.array([1/2, 1/2])
#         
#             # Compare true gradient with calculated gradient
#             # TO DO: Change tolerance
#             assert np.allclose(est_grad, true_grad, atol = 1), "Gradient is not being estimated correctly by calculate_gradient function"

def test_training():
    """
    Unit test to check that weights update during training.
    Fit a model with our regression module functions, to data in dataset/data/nsclc.csv,
    and compare the weights of the final model, with that of the initalised model.
    """

    # Load data
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)'
        ],
        split_percent=0.8,
        split_seed=42
    )
    # perform necessary data scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    # Initialise logistic regression model
    lr_mod = reg.LogisticRegressor(
                                   num_feats=X_train.shape[1],
                                   max_iter=50
                                   )

    # Get initial weights
    init_weights = lr_mod.W.copy()

    # Train/fit logistic regression module using regression module
    lr_mod.train_model(X_train, y_train, X_val, y_val)

    # Get final weights
    final_weights = lr_mod.W.copy()

    # Check: have initial weights been updated (i.e. changed during training)?
    assert np.array_equal(init_weights, final_weights) == False, 'Model weights are not being updated during training'

