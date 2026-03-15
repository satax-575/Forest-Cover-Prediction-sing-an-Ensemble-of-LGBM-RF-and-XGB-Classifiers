from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight #for class balance in XGBClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import set_config

from preprocessors import NonLinearPreprocessor, LinearPreprocessor

#[IMP]
#to make sure column transformer gives dataframes as output
set_config(transform_output="pandas")


def build_combined_pipeline(mean_cols, mode_cols):
    combined_pline = Pipeline([
        ('imputer', ColumnTransformer(
            transformers=[
                ('mean_impute', SimpleImputer(strategy='mean'), mean_cols),
                ('mode_impute', SimpleImputer(strategy='most_frequent'), mode_cols)
            ],
            remainder='passthrough'  # keep all other columns
        )),

        ('feature_engg', NonLinearPreprocessor()),

        ('encode', ColumnTransformer(
            transformers=[
                ('encoder', OrdinalEncoder(), ['Slope_Direction'])
            ],
            remainder='passthrough'
        ))
    ])
    return combined_pline


def build_linear_pipeline(mean_cols, mode_cols):
    linear_pline = Pipeline([
        ('imputer', ColumnTransformer(
            transformers=[
                ('mean_impute', SimpleImputer(strategy='mean'), mean_cols),
                ('mode_impute', SimpleImputer(strategy='most_frequent'), mode_cols)
            ],
            remainder='passthrough'  # keep all other columns
        )),

        ('feature_engg', LinearPreprocessor())
    ])
    return linear_pline


def build_model_pipelines(combined_pline):
    best_rf_params = {'max_depth': 25, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1200}

    rf_pline = Pipeline([
        ('preprocess', combined_pline),
        ('rf', RandomForestClassifier(
            **best_rf_params,
            class_weight="balanced_subsample",   # IMPORTANT
            random_state=11,
            n_jobs=-1
        ))
    ])

    xg_best_params = {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 9, 'n_estimators': 1200, 'reg_lambda': 2, 'subsample': 0.7}

    xg_pline = Pipeline([
        ('preprocess', combined_pline),   #preprocessing pipeline
        ('xg', XGBClassifier(
            **xg_best_params,
            objective='multi:softprob',
            tree_method='hist',
            booster='gbtree',
            eval_metric='mlogloss',
            random_state=11,
            n_jobs=-1
        ))
    ])

    lg_best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': -1, 'min_child_samples': 30, 'min_split_gain': 0, 'n_estimators': 1500, 'num_leaves': 63, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.8}

    lg_pline = Pipeline([
        ('preprocess', combined_pline),
        ('lg', LGBMClassifier(
            **lg_best_params,
            objective="multiclass",
            num_class=7,
            boosting_type="gbdt",
            class_weight="balanced",
            random_state=11,
            n_jobs=-1
        ))
    ])

    return rf_pline, xg_pline, lg_pline


def build_stacking_classifier(base_estimators):
    sc = StackingClassifier(
        estimators = base_estimators,
        final_estimator = LogisticRegression(
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=2000,
            random_state=42),
        cv = 5,
        n_jobs = -1
    )
    return sc


def build_lr_svm_pipelines(linear_pline):
    log_reg = Pipeline([
        ('preprocess', linear_pline),
        ('log_reg', LogisticRegression())
    ])

    svm = Pipeline([
        ('preprocess', linear_pline),
        ('svc', SVC())
    ])

    return log_reg, svm
