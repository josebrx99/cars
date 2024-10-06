import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from typing import Any, BinaryIO, Literal, TypeVar, List, Dict

import xgboost as xgb

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')).replace("\\", "/")+ "/src")
import functions as func
import paths as ph


def MetricsRegression(y_test, X_test, predicts):
    """
    Calcular metricas para regresión.

    Example
    ---------
    models.MetricsRegression(y_test = [], X_test = [], predicts = model.predict(X_test))
    """
    rmse = round(mean_squared_error(y_test, predicts, squared=False), 4)
    r_2 = round(r2_score(y_test, predicts), 4)
    n = len(y_test)
    p = X_test.shape[1]
    r_2_adj = round(1 - ((1 - r_2) * (n - 1) / (n - p - 1)), 4)
    print(f"R2: {r_2} | R2 Adj: {r_2_adj},  RMSE: {rmse},  n: {n},  p: {p}")
    return {"r2":[r_2], "r2_adj":[r_2_adj], "rmse":[rmse], "n":[n], "p":[p]}

def ImputRegression(
        df = pd.DataFrame(), 
        y = str, 
        x = List,
        method = "lm"):
    """
    Imputar variables faltantes en una variable continua mediante regresion.

    Parameters
    ----------
    y: variable a imputar.
    x: covariables.

    Example
    ----------
    df = ImputRegression(df, y = 'motor', x = ['precio', 'año', 'marca_agrup', 'modelo_agrup', 'tipo_de_combustible', 'transmision', 
                                               'tipo_de_carroceria', 'puertas'])
    """
    from sklearn.linear_model import LinearRegression

    vars_numeric = func.detect_numeric(df[x])
    vars_cat = list(set(x) - set(vars_numeric))
    df_imput = df[[y] + vars_numeric + vars_cat].dropna()
    y_train = df_imput[y]
    X_train = pd.get_dummies(df_imput[vars_numeric + vars_cat], columns = vars_cat, drop_first=True)
    X_train = X_train.replace(True, 1).replace(False, 0)
    
    if method == "lm":
        model = LinearRegression()
        model.fit(X_train, y_train)
        
    elif method == "xgboost":
        param_grid = {'booster': ['gbtree'], 'objective': ['reg:squarederror']}
        xgb_model = xgb.XGBRegressor()
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        model =  grid_search.best_estimator_
    else:
        raise ValueError("Model not found")

    MetricsRegression(y_test = y_train, X_test = X_train, predicts = model.predict(X_train))

    df_missing = df[df[y].isna()]
    df = df[df[y].notna()]
    X_missing = pd.get_dummies(df_missing[vars_numeric + vars_cat], columns = vars_cat, drop_first=True)
    X_missing = X_missing.replace(True, 1).replace(False, 0)
    X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)
    df_missing[y] = model.predict(X_missing)

    df = pd.concat([df, df_missing], axis = 0)

    return df


def ImputCategorics(
        df = pd.DataFrame(), 
        var_id = List, 
        var_imput = List, 
        umbral = float | int
        ):
    """
    Imputar variables categoricas por su moda si esta supera un porcentaje umbral.

    Parameters
    ----------
    umbral: umbral que representa porcentaje.

    Example
    ----------
    df = models.ImputCategorics(df, "marca_modelo", "transmision", 70)
    """
        
    df["cont"] = 1
    df_imput = df.copy()
    df_imput[var_imput] = df_imput[var_imput].fillna("NULL")
    df_imput = df_imput.groupby([var_id, var_imput])["cont"].sum().reset_index()
    num_nulls_initial = df[df[var_imput].isnull()].shape[0]

    df["total"] = 1
    ax_marca = df.groupby([var_id])["total"].sum().reset_index().sort_values("total", ascending=False)

    df_imput = pd.merge(df_imput, ax_marca, on = [var_id], how = "left")

    df_imput["%"] = round((df_imput["cont"] / df_imput["total"])* 100, 2)
    df_imput = df_imput.sort_values([var_id], ascending = True)

    df_imput = df_imput[df_imput[var_imput] != 'NULL']
    df_imput = df_imput.groupby(var_id).apply(lambda x: x if (x['%'] > umbral).any() else x.head(0)).reset_index(drop=True)
    df_imput = df_imput.sort_values([var_id, "%"], ascending = False).groupby(var_id).first().reset_index()
    df_imput["var_imput_temp"] = df_imput[var_imput]
    df_imput = df_imput[[var_id, "var_imput_temp"]]

    df_imput["is_imputate"] = 1
    df = pd.merge(df, df_imput, on = [var_id], how = "left")
    df[var_imput] = df[var_imput].fillna(df["var_imput_temp"])

    num_nulls_final = df[df[var_imput].isnull()].shape[0]
    print("Nulos", var_imput, ":", num_nulls_initial, 
            "| Imputados:", num_nulls_initial - num_nulls_final, 
                            f"({0 if num_nulls_initial == 0 else round(100-(num_nulls_final / num_nulls_initial)*100, 2)} %)")

    df_performance = df[df["is_imputate"] == 1].copy()
    df_performance = df[["tipo_de_carroceria", "var_imput_temp"]].dropna()
    num_concordances = df_performance[df_performance["tipo_de_carroceria"] == df_performance["tipo_de_carroceria"]].shape[0]
    print("Precision model:", round((num_concordances / df_performance.shape[0])*100, 2), "%")

    del df["cont"], df["total"], df['var_imput_temp'], df['is_imputate']
    return df


def ClassicationTreeGroup(
        df = pd.DataFrame(),
        y = str, 
        x = str, 
        max_leaf_nodes=[3, 5, 10], 
        random_state = 42,
        metric_imput = "median",
        prints = False
        ):
    """
    Agrupar mediante un árbol de regresión, una variable categorica en función de una 
    sola covariable continua

    Parameters
    ----------
    y: numeric
    x: categoric (to agroup)
    max_leaf_nodes: number of groups

    Example
    ----------
    df = models.ClassicationTreeGroup(df, y = "precio", x = "marca", max_leaf_nodes=[4, 5, 6, 7])
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    _x_name = x
    df[x] = df[x].fillna('NULL')
    print("Categories in", _x_name, ":", len(df[x].unique()))

    le = LabelEncoder()
    df['x_encoded'] = le.fit_transform(df[x])

    y_imputer = SimpleImputer(strategy=metric_imput)
    df['y_imputed'] = y_imputer.fit_transform(df[[y]])

    X = df[['x_encoded']]
    y = df['y_imputed']

    tree = DecisionTreeRegressor(random_state=random_state)
    param_grid = {
        'max_leaf_nodes': max_leaf_nodes,
        # 'max_depth': [2, 3, 4, 5, 6, 8],
        # 'min_samples_split': [2, 3, 4, 5, 6, 8],
        # 'min_samples_leaf': [2, 3, 4, 5, 6, 8]}
    }
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='r2', error_score=np.nan)
    grid_search.fit(X, y)
    best_tree = grid_search.best_estimator_
    df['terminal_node'] = best_tree.apply(X)
    print("Final groups:", len(df['terminal_node'].unique()))
    node_to_group = df.groupby('terminal_node')[x].apply(lambda x: ', '.join(x.unique()))
    _x_name_agrup = _x_name + '_name_agrup'
    df[_x_name_agrup] = df['terminal_node'].map(node_to_group)

    _df_groups = df[_x_name_agrup].value_counts().reset_index()
    _df_groups.columns = [_x_name_agrup, 'count']
    _df_groups["count%"] = round((_df_groups["count"] / _df_groups["count"].sum()) * 100, 2)
    del _df_groups["count"]

    _df_groups[_x_name + "_agrup"] = ["group_" + str(i) for i in range(1, _df_groups.shape[0] + 1)]
    if prints == True:
        print("-" * 30, "Groups", "-" * 30)
        print(_df_groups)

    df = pd.merge(df, _df_groups, on=[_x_name_agrup], how="left")
    df.drop(columns=["x_encoded", "terminal_node", "count%", "y_imputed"], inplace=True)
    df[_x_name] = df[_x_name].replace({'NULL':np.nan})
    return df

def TrainTestDummies(df, 
             y, 
             dummies = None, 
             test_size = 0.2, 
             random_state = 42
             ):
    """
    X_train, X_test, y_train, y_test = models.TrainTestDummies(df = df, y='precio', dummies = ["transmision", "tipo_de_combustible"])
    """
    from sklearn.model_selection import train_test_split
    
    if dummies is not None:
        df = pd.get_dummies(df, columns = dummies, drop_first=True)
        df = df.replace(True, 1).replace(False, 0)
    X = df.drop(y, axis=1)
    y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


class XGBoost:
    """
    Parameters
    ----------

    Example
    ----------
    model = XGBoost(df, y='precio', param_grid = param_grid).fit()

    param_grid = {
        'n_estimators': [10, 50, 100, 250, 500],
        'max_depth': [2, 4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.5],
        'min_child_weight': [2, 4, 6, 8],
        'booster': ['gbtree'],
        'eta': [0, 0.4, 0.8, 1],
        'gamma': [0, 0.4, 0.8, 1],
        'lambda': [0,  0.4, 0.8, 1],
        'alpha': [0, 0.4, 0.8, 1],
        'subsample': [0.5, 0.65, 0.8, 0.9],
        'colsample_bytree': [0.4, 0.6, 0.8, 1],
        'objective': ['reg:squarederror']
    }
    model = XGBoost(df, y='precio', param_grid = param_grid).fit()
    """
    def __init__(
            self, 
            y_train = pd.DataFrame(), 
            X_train = pd.DataFrame(), 
            y_test = pd.DataFrame(), 
            X_test = pd.DataFrame(),
            random_state=42, 
            param_grid=None,
            cv = 5,
            save = False):
        
        self.y_train = y_train
        self.X_train = X_train
        self.y_test = y_test
        self.X_test = X_test
        self.random_state = random_state
        self.param_grid = param_grid
        self.best_grid = None
        self.cv = cv
        self.save = save
        self.figsize = (7, 5)

    def _train(self):
        if self.param_grid is None:
            self.param_grid = {'booster': ['gbtree'],
                               'objective': ['reg:squarederror'],
                               "n_jobs":[-1]}

        xgb_model = xgb.XGBRegressor()
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=self.param_grid, cv=self.cv, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        self.best_grid = grid_search.best_estimator_

    def importance_(self):
        feature_importances_ = self.best_grid.feature_importances_
        vars = self.X_train.columns

        df_importances = pd.DataFrame({'Variable': vars, 'Importance': feature_importances_})
        df_importances = df_importances.sort_values(by='Importance', ascending=True).tail(15)

        plt.figure(figsize=self.figsize)
        plt.barh(df_importances['Variable'], df_importances['Importance'], color="#CD5C5C")
        plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Importance XGBoost')
        plt.show()

    def fit(self):
        self._train()
        
        predicts_test = self.best_grid.predict(self.X_test)
        predicts_train = self.best_grid.predict(self.X_train)

        print("Test Metrics:")
        metrics_test = MetricsRegression(self.y_test, self.X_test, predicts_test)

        print("Train Metrics:")
        metrics_train = MetricsRegression(self.y_train, self.X_train, predicts_train)

        self.importance_()

        import os
        if self.save == True:
            metrics_test = pd.DataFrame(metrics_test)
            metrics_train = pd.DataFrame(metrics_train)
            metrics_test["set"] = "test"; metrics_train["set"] = "train"
            df_metrics = pd.concat([metrics_train, metrics_test], axis = 0)
            df_metrics["dtm_train"] = func.GetActualDate()
            df_metrics.to_csv(f'{ph.metrics}/metrics_xgboost.txt', sep = "|", index = False)
        
        return self.best_grid, metrics_test



def predictNN(model, X):
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    X_new_scaled = scalerX.transform(X)
    y_pred_scaled = model.predict(X_new_scaled)
    y_pred = scalerY.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return y_pred

class NN:
    """
    Parameters
    ----------
    y_train : DataFrame
        Target variable for training.
    X_train : DataFrame
        Features for training.
    y_test : DataFrame
        Target variable for testing.
    X_test : DataFrame
        Features for testing.
    random_state : int
        Random seed for reproducibility.
    param_grid : dict
        Dictionary of parameters for hyperparameter tuning.
    cv : int
        Cross-validation splits.
    save : bool
        Flag to save metrics.

    Example
    ----------
    param_grid = {
                    'layers': [[64, 128]], 
                    'activation': ['relu'],      
                    'optimizer': ['adam'],         
                    'epochs': [100],                 
                    'batch_size': [10]
                }
    model, metrics_test = models.NN(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, param_grid = param_grid, cv = 2, save = True).fit()
    """

    def __init__(
            self, 
            y_train=pd.DataFrame(), 
            X_train=pd.DataFrame(), 
            y_test=pd.DataFrame(), 
            X_test=pd.DataFrame(),
            random_state=42, 
            param_grid=None,
            cv=5,
            save=False):
        
        self.y_train = y_train
        self.X_train = X_train
        self.y_test = y_test
        self.X_test = X_test
        self.random_state = random_state
        self.param_grid = param_grid
        self.best_grid = None
        self.cv = cv
        self.save = save
        self.figsize = (7, 5)
        self.scalerX = MinMaxScaler()
        self.scalerY = MinMaxScaler()

    def _train(self):
        if self.param_grid is None:
            self.param_grid = {
                'layers': [[12, 24]], 
                'activation': ['relu'],      
                'optimizer': ['adam'],         
                'epochs': [100],                 
                'batch_size': [10]
            }

        def _define_model(layers, activation='relu', optimizer='adam'):
            model = Sequential()
            for i, neurons in enumerate(layers):
                if i == 0: 
                    model.add(Dense(neurons, input_shape=(self.X_train.shape[1],), activation=activation))
                else:  
                    model.add(Dense(neurons, activation=activation))
            model.add(Dense(1, activation='linear'))  
            model.compile(optimizer=optimizer, loss='mse')
            return model
        
        X_train_scaled = self.scalerX.fit_transform(self.X_train)
        y_train = np.array(self.y_train).reshape(-1, 1)
        y_train_scaled = self.scalerY.fit_transform(y_train)

        model = KerasRegressor(build_fn=_define_model, verbose=0)
        
        grid = GridSearchCV(estimator=model, param_grid=self.param_grid, n_jobs=-1, cv=self.cv)
        grid_result = grid.fit(X_train_scaled, y_train_scaled)
        self.best_grid = grid_result.best_estimator_

        print("Best grid:")
        print(grid_result.best_params_)


    def fit(self):
        self._train()

        predicts_test = predictNN(self.best_grid, self.X_test)
        predicts_train = predictNN(self.best_grid, self.X_train)

        print("Test Metrics:")
        metrics_test = MetricsRegression(self.y_test, self.X_test, predicts_test)

        print("Train Metrics:")
        metrics_train = MetricsRegression(self.y_train, self.X_train, predicts_train)

        import os
        if self.save:
            metrics_test = pd.DataFrame(metrics_test)
            metrics_train = pd.DataFrame(metrics_train)
            metrics_test["set"] = "test"
            metrics_train["set"] = "train"
            df_metrics = pd.concat([metrics_train, metrics_test], axis=0)
            df_metrics["dtm_train"] = func.GetActualDate()
            df_metrics.to_csv(f'{ph.metrics}/metrics_nn.txt', sep="|", index=False)
        
        return self.best_grid, metrics_test


class LM:
    """
    import models
    X_train, X_test, y_train, y_test = models.TrainTestDummies(df = df, y='precio', dummies = ["transmision", "tipo_de_combustible"])

    lm = models.LM(y_train, X_train, y_test, X_test)
    lm.fit() 

    lm.results()

    lm.assumpts()

    lm.vif()
    """
    def __init__(self, y_train=pd.DataFrame(), X_train=pd.DataFrame(), y_test=pd.DataFrame(), X_test=pd.DataFrame()):
        self.y_train = y_train
        self.X_train = X_train
        self.y_test = y_test
        self.X_test = X_test
        self.X_train_orig = X_train
        self.figsize = (7, 5)
        self.model = None

    def _train(self):
        import statsmodels.api as sm
        import numpy as np

        self.X_train = sm.add_constant(self.X_train)
        self.y_train = np.asarray(self.y_train)
        self.X_train = np.asarray(self.X_train)
        self.model = sm.OLS(self.y_train, self.X_train).fit()
        return self.model
    
    def results(self):
        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
        results = pd.DataFrame({'variable': ["intercept"] + self.X_train_orig.columns.tolist(),
                                'beta': self.model.params,
                                'p-valor': self.model.pvalues})
        return results
    
    def vif(self):
        # NO FUNCIONAL
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        df_vif = self.X_train_orig
        df_vif = pd.DataFrame()
        df_vif['variable'] = df_vif.columns
        df_vif['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        return df_vif

    @staticmethod
    def assumpts(
        fittedvalues, 
        residuals):
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        standardized_residuals = (residuals - residuals.mean())/np.sqrt(residuals.var())

        # Histogram
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        sns.histplot(standardized_residuals, kde=True)
        plt.title('Histograma de Residuos Estandarizados'); plt.xlabel('Residuos Estandarizados'); plt.ylabel('Frecuencia')

        # Fitted vs residuals
        plt.subplot(1, 2, 2)
        plt.scatter(fittedvalues, standardized_residuals, alpha=1, s = 20, c = "black")
        plt.title('Valores Ajustados vs Residuos Estandarizados'); plt.xlabel('Valores Ajustados'); plt.ylabel('Residuos Estandarizados')
        plt.axhline(0, color='red', linestyle='--')
        plt.show(); plt.tight_layout()

    def fit(self):
        import statsmodels.api as sm
        self._train()
        
        if self.X_test is not None:
            self.X_test = sm.add_constant(self.X_test)
        
        predicts_test = self.model.predict(self.X_test)
        predicts_train = self.model.predict(self.X_train)

        print("Test Metrics:")
        MetricsRegression(self.y_test, self.X_test, predicts_test)

        print("Train Metrics:")
        MetricsRegression(self.y_train, self.X_train, predicts_train)

        return self.model
    

class GLM:
    """
    import models
    X_train, X_test, y_train, y_test = models.TrainTestDummies(df = df, y='precio', dummies = ["transmision", "tipo_de_combustible"])

    lm = models.GLM(y_train, X_train, y_test, X_test)
    lm.fit() 

    lm.results()

    lm.assumpts()

    lm.vif()
    """
    def __init__(self, y_train=pd.DataFrame(), X_train=pd.DataFrame(), y_test=pd.DataFrame(), X_test=pd.DataFrame()):
        self.y_train = y_train
        self.X_train = X_train
        self.y_test = y_test
        self.X_test = X_test
        self.X_train_orig = X_train
        self.figsize = (7, 5)
        self.model = None

    def _train(self):
        import statsmodels.api as sm
        import numpy as np

        self.X_train = sm.add_constant(self.X_train)
        self.y_train = np.asarray(self.y_train)
        self.X_train = np.asarray(self.X_train)
        self.model = sm.GLM(self.y_train, self.X_train, family=sm.families.Gamma(sm.families.links.log)).fit()
        return self.model
    
    def results(self):
        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
        results = pd.DataFrame({
            'variable': ["intercept"] + self.X_train_orig.columns.tolist(),
            'beta': self.model.params,
            'p-valor': self.model.pvalues
        })
        return results
    
    def vif(self):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        df_vif = pd.DataFrame()
        df_vif['variable'] = self.X_train_orig.columns
        df_vif['VIF'] = [variance_inflation_factor(self.X_train_orig.values, i) for i in range(self.X_train_orig.shape[1])]
        return df_vif

    def assumpts(self):
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        fitted_values = self.model.fittedvalues
        residuals = self.model.resid_response
        standardized_residuals = (residuals - residuals.mean())/np.sqrt(residuals.var())

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        sns.histplot(standardized_residuals, kde=True)
        plt.title('Histograma de Residuos Estandarizados'); plt.xlabel('Residuos Estandarizados'); plt.ylabel('Frecuencia')

        plt.subplot(1, 2, 2)
        plt.scatter(fitted_values, standardized_residuals, alpha=1, s=20, c="black")
        plt.title('Valores Ajustados vs Residuos Estandarizados'); plt.xlabel('Valores Ajustados'); plt.ylabel('Residuos Estandarizados')
        plt.axhline(0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()

    def fit(self):
        import statsmodels.api as sm
        self._train()
        
        if self.X_test is not None:
            self.X_test = sm.add_constant(self.X_test)
        
        predicts_test = self.model.predict(self.X_test)
        predicts_train = self.model.predict(self.X_train)

        print("Test Metrics:")
        MetricsRegression(self.y_test, self.X_test, predicts_test)

        print("Train Metrics:")
        MetricsRegression(self.y_train, self.X_train, predicts_train)

        return self.model
