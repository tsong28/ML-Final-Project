import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid


def preprocess_credit_card_data(df):
    df = df.copy()

    #drop id; unnecessary for analysis
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    #Clean invalid catagorical entries
    df = df[df['SEX'].isin([1, 2])]
    df = df[df['EDUCATION'].isin([1, 2, 3,4,5,6])]
    df = df[df['MARRIAGE'].isin([1, 2, 3])]

    #Identify features
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

    #One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    #Force dummy columns to be integers (0 and 1) in case any are in a different form
    dummy_cols = [col for col in df.columns if any(prefix in col for prefix in ['SEX_', 'EDUCATION_', 'MARRIAGE_'])]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df


def grid_evaluate(
    estimator,
    param_grid,
    X_train, X_validation,
    y_train, y_validation
):
    """
    Custom grid search with optional feature transforms and model-specific parameters.
    Returns a tuple: (results_dataframe, best_trained_model)
    """

    rows = []
    best_f1 = -1
    best_model = None

    i = 0

    for params in ParameterGrid(param_grid):
        i += 1

        # extract feature transform params
        fm    = params.pop('feature_method', None)
        deg   = params.pop('degree', None)
        ncomp = params.pop('n_components', None)
        gam   = params.pop('gamma', None)

        # apply feature transformation
        if fm == 'polynomial':
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_tr = poly.fit_transform(X_train)
            X_val = poly.transform(X_validation)
        elif fm == 'pca':
            pca = PCA(n_components=ncomp)
            X_tr = pca.fit_transform(X_train)
            X_val = pca.transform(X_validation)
        elif fm == 'rbf':
            X_tr = rbf_kernel(X_train, X_train, gamma=gam)
            X_val = rbf_kernel(X_validation, X_train, gamma=gam)
        else:
            X_tr, X_val = X_train, X_validation

        # fit model
        clf = clone(estimator).set_params(**params)
        clf.fit(X_tr, y_train)

        # predictions
        y_pred_val = clf.predict(X_val)
        y_pred_tr  = clf.predict(X_tr)

        # validation metrics
        acc_val  = accuracy_score(y_validation, y_pred_val)
        prec_val, rec_val, f1_val, _ = precision_recall_fscore_support(
            y_validation, y_pred_val, average='binary', zero_division=0
        )

        # training metrics
        acc_tr  = accuracy_score(y_train, y_pred_tr)
        prec_tr, rec_tr, f1_tr, _ = precision_recall_fscore_support(
            y_train, y_pred_tr, average='binary', zero_division=0
        )

        # save best model
        if f1_val >= best_f1:
            best_f1 = f1_val
            best_model = clf  # already fitted

        # default/cleanup for output
        if fm is None:
            fm = 'Linear'
        if fm != 'polynomial':
            deg = 1

        record = {
            'feature_method':        fm,
            'degree':                deg,
            'n_components':          ncomp,
            'gamma':                 gam,
            'accuracy_validation':   acc_val,
            'accuracy_train':        acc_tr,
            'precision_validation':  prec_val,
            'recall_validation':     rec_val,
            'f1_validation':         f1_val,
            'precision_train':       prec_tr,
            'recall_train':          rec_tr,
            'f1_train':              f1_tr,
        }
        record.update(params)  # add remaining hyperparameters
        rows.append(record)

        #print progress per parameter combination tried
        if i % 5 == 0:
            print(f"Evaluated {i} parameter combinations...")

    return pd.DataFrame(rows), best_model
