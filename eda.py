import pandas as pd
import numpy as np
import re as re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, BinaryIO, Literal, TypeVar, List, Dict

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')).replace("\\", "/")+ "/src")
import functions as func

class Stats:
    """
    Examples
    ----------
    eda.EDA(df).eda()

    eda.EDA(df).nulls()

    eda.EDA(df).discretes(n=10)

    eda.EDA(df).continues()
    """
    figsize_main = (8,4)

    def __init__(self, df):
        self.df = df

    def dimension(self):
        valor = self.shape
        print(f"Dimensión: {valor}")
        
    def duplicates(self):
        valor = self.duplicated().sum()
        print(f"Duplicados: {valor}"); print("="*120)

    def nulls(self):
        print("\nNulos por columna:"); print("_"*50)
        ax = self.isnull().sum()
        ax = pd.DataFrame(ax).reset_index()
        ax.columns = ["column", "nulls"]
        ax["nulls%"] = round(ax["nulls"] / self.shape[0] *100, 2)
        print(ax); print("-"*60)

        ax_grave = ax[ax["nulls%"] >= 30]
        if ax_grave.shape[0] != 0:
            print("ADVERTENCIA: variables con más del 30% de nulos")
            print(ax_grave); print("="*120)

    def discretes(self, n=10):
        print("\nEstadísticas de variables categóricas:")
        frecuents = {}
        for col in self.columns:
            try:
                pd.to_numeric(self[col])
            except ValueError:
                ax = self[col].value_counts().head(n).reset_index()
                ax.columns = ["variable", "total"]
                ax["total%"] = round(ax["total"] / ax["total"].sum() *100, 2)
                frecuents[col] = ax
                print(f"\n{col}:")
                print(frecuents[col])
        print("="*120)

    def continues(self):
        print("\nEstadísticas de variables continuas:"); print("_"*50)
        stats = self.describe().T[['min', 'max', 'mean']]
        stats['median'] = self.median()
        stats['var'] = self.var()
        stats['sd'] = np.sqrt(self.var())
        stats['cv'] = (np.sqrt(self.var()) / self.mean()) * 100
        stats = round(stats, 2)
        print(stats); print("="*120)

    def eda(self):
        self.dimension()
        self.duplicates()
        self.nulls()
        self.discretes(n=10)
        self.continues()


def summary_numeric(variable):
    if isinstance(variable, (list, np.ndarray)):
        _min = round(np.min(variable), 2)
        _max = round(np.max(variable), 2)
        _mean = round(np.mean(variable), 4)
        _median = round(np.median(variable), 4)
        _sd = round(np.sqrt(np.var(variable)), 4)
        _cv = round(_sd / _mean, 4) * 100
    else:
        _min = round(variable.min(), 2)
        _max = round(variable.max(), 2)
        _mean = round(variable.mean(), 8)
        _median = round(variable.median(), 4)
        _sd = round(np.sqrt(variable.var()), 4)
        _cv = round(np.sqrt(variable.var()) / variable.mean(), 4) * 100

    print("min:", _min, "| max:", _max,  "| mean:", _mean, "| median:", _median,  "| sd:", _sd,  "| cv:", _cv, "%")

class Plot:
    """
    Examples
    ----------
    eda.Plot.hist("precio")

    eda.Plot.plot_serie("año","precio","median")

    outliers = eda.Plot.uni_outliers("precio", deviation = "extreme")

    eda.Plot.barplot("tipo_de_carroceria", horiz= True)

    eda.Plot.pairs()
    """
    figsize_main = (8,4)

    @staticmethod
    def hist(
            variable=str, 
            name_variable = "Histograma",
            bins = 20,
            figsize=(8,4)
        ):
        
        from scipy.stats import gaussian_kde
        plt.figure(figsize=figsize)
        plt.hist(variable, bins=bins, edgecolor='black', density=True)
        line_density = gaussian_kde(variable)
        x = np.linspace(min(variable), max(variable), 100)
        plt.plot(x, line_density(x), color='red')
        plt.title('Histograma de ' + name_variable if name_variable != "Histograma" else name_variable)
        plt.xlabel(name_variable)
        plt.ylabel('Frecuencia')
        plt.show()

        summary_numeric(variable)
        
    
    @staticmethod
    def barplot(
            df, 
            variable=str, 
            top_n=None, 
            horiz=False, 
            figsize=figsize_main
            ):
        
        plt.figure(figsize = figsize)
        if horiz == True:
            counts = df[variable].value_counts()
            if top_n is not None:
                counts = counts.head(top_n)
            counts = counts.sort_values(ascending=True)
            values = counts.values
            plt.barh(counts.index, values, edgecolor='black')
            for index, value in enumerate(values):
                plt.text(value + value*0.02, index, f'{value} ({value/counts.sum():.1%})', va='center', ha='left')
            plt.xlim(0, max(counts) + max(counts)*0.15)
            plt.ylabel(variable)
            plt.xlabel('Frecuencia')
        else:
            counts = df[variable].value_counts()
            if top_n is not None:
                counts = counts.head(top_n).sort_index()
            counts = counts.sort_values(ascending=False)
            plt.bar(counts.index, counts.values, edgecolor='black')
            for index, value in enumerate(counts.values):
                text = f'{value} ({value/counts.sum():.1%})'
                plt.text(counts.index[index], value + value*0.01, text, ha='center', va="bottom")
            plt.ylim(0, max(counts) + max(counts)*0.1)
            plt.xlabel(variable)
            plt.ylabel('Frecuencia')
        plt.title(f'Barplot de {variable} (Top {top_n})' if top_n else f'Barplot de {variable}')
        plt.show()

    @staticmethod
    def plot_serie(
            df, 
            x=str, 
            y=str, 
            agg = "mean", 
            figsize=figsize_main):
        
        plt.figure(figsize = figsize)
        df.groupby(x)[y].agg(agg).plot(kind='line')
        plt.show()


    @staticmethod
    def pairs(df, figsize=figsize_main):
        import seaborn as sns, matplotlib.pyplot as plt
        plt.figure(figsize = figsize)
        df_continous = df[func.detect_numeric(df)]
        pairplot = sns.pairplot(df_continous, kind='scatter', diag_kind='kde')
        def _calculate_corr(axes, df):
            variables = df.columns
            n = len(variables)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ro = df[[variables[j], variables[i]]].corr().iloc[0, 1]
                        axes[i, j].text(0.5, 0.9, f'ρ = {ro:.4f}', transform=axes[i, j].transAxes, horizontalalignment='center',
                                        fontsize=10, color='red')
        _calculate_corr(pairplot.axes, df_continous)
        plt.show()

    @staticmethod
    def uni_outliers(
            variable, 
            deviation="extreme", 
            figsize=figsize_main,
            title = "Histograma"):
        
        import seaborn as sns
        if deviation == "extreme":
            deviation = 3
        elif deviation == "severe":
            deviation = 1.5
        else:
            raise ValueError("Deviation values ​​not allowed: extreme, severe")

        plt.figure(figsize=figsize)
        sns.boxplot(x=variable, orient="h")

        Q1 = variable.quantile(0.25)
        Q3 = variable.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - deviation * IQR
        upper_bound = Q3 + deviation * IQR
        outliers = variable[(variable < lower_bound) | (variable > upper_bound)]

        plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower bound ({lower_bound:.2f})')
        plt.axvline(upper_bound, color='blue', linestyle='--', label=f'Upper bound ({upper_bound:.2f})')
        plt.xlabel(variable); plt.title("Histograma de " + variable if title != "Histograma" else title); plt.legend()
        plt.scatter(outliers, [0] * len(outliers), color='red', zorder=5, label='Outliers')
        plt.show()

        print("Deviation:", str(deviation) + " + IQR")
        print("Total outliers:", len(outliers))
        print(sorted(outliers))
        return outliers.tolist()

    @staticmethod
    def boxplot(
            variable,  
            figsize=(10, 6),
            title="Boxplot",
            orient="v",
            df = pd.DataFrame(),
            variable_cat=None,
            agg = "mean"):
        
        import seaborn as sns
        plt.figure(figsize=figsize)
        if variable_cat is None:
            sns.boxplot(x=variable, orient=orient)
            plt.title(title if title != "Boxplot" else title)
        else:
            order = df.groupby(variable_cat).agg({variable: agg}).sort_values(variable, ascending=False).index
            sns.boxplot(x=variable_cat, y=variable, orient=orient, data = df, order = order)
            plt.title(title if title != "Boxplot" else f"Boxplot de {variable} de acuerdo a {variable_cat}")
        plt.xlabel(variable_cat if variable_cat else "variable")
        plt.ylabel("variable" if variable_cat else "")
        plt.legend(); plt.show()

        summary_numeric(variable)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as SKPCA
from sklearn.cluster import KMeans
class PCA:
    def __init__(
        self, 
        df=pd.DataFrame(), 
        vars_numeric=None, 
        vars_cat=None, 
        n_components=2, 
        clusters=None,
        title="PCA", 
        figsize=(8, 5), 
        font_size=10,
        pch=15, 
        palette="plasma",
        xlim=None,
        ylim=None
    ):
        self.df = df
        self.vars_numeric = vars_numeric
        self.vars_cat = vars_cat if vars_cat is not None else []
        self.n_components = n_components
        self.clusters = clusters
        self.title = title
        self.figsize = figsize
        self.font_size = font_size
        self.pch = pch
        self.palette = palette
        self.xlim = xlim
        self.ylim = ylim
        self.pca = None
        self.components = None
        self.inertia = None
        self.transformed_columns = None

    def fit(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), self.vars_cat),
                ('cont', StandardScaler(), self.vars_numeric)
            ]
        )
        pipeline_pca = Pipeline(steps=[('preprocessor', preprocessor), ('pca', SKPCA(n_components=self.n_components))])
        self.df_pca = pipeline_pca.fit_transform(self.df)
        self.df_pca = pd.DataFrame(self.df_pca, columns=[f'PC{i+1}' for i in range(self.n_components)])

        self.pca = pipeline_pca.named_steps['pca']
        self.components = self.pca.components_
        self.inertia = [round(v * 100, 2) for v in self.pca.explained_variance_ratio_]

        if self.vars_cat:
            self.transformed_columns = (
                pipeline_pca.named_steps['preprocessor']
                .transformers_[0][1]
                .get_feature_names_out(self.vars_cat)
                .tolist()
            )
        else:
            self.transformed_columns = []
        self.transformed_columns.extend(self.vars_numeric)

    def plot(self):
        if self.pca is None or self.components is None:
            self.fit()

        plt.figure(figsize=self.figsize)
        if self.clusters is None:
            sns.scatterplot(x='PC1', y='PC2', data=self.df_pca, s=self.pch, color="gray")
        else:
            n_components_kmeans = len(self.vars_cat + self.vars_numeric)
            pipeline_kmeans = Pipeline(steps=[('preprocessor', ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first'), self.vars_cat),
                    ('cont', StandardScaler(), self.vars_numeric)
                ])), ('pca', SKPCA(n_components=n_components_kmeans))])
            
            df_kmeans = pipeline_kmeans.fit_transform(self.df)
            df_kmeans = pd.DataFrame(df_kmeans, columns=[f'PC{i+1}' for i in range(n_components_kmeans)])
            kmeans = KMeans(n_clusters=self.clusters, random_state=0)
            df_kmeans['cluster'] = kmeans.fit_predict(df_kmeans) + 1
            self.df_clusters = df_kmeans

            palette = sns.color_palette(self.palette, as_cmap=True)
            cluster_colors = {i: palette(i / self.clusters) for i in range(1, self.clusters + 1)}
            sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_kmeans, palette=cluster_colors, s=self.pch, alpha=0.7)
            for cluster in df_kmeans['cluster'].unique():
                cluster_data = df_kmeans[df_kmeans['cluster'] == cluster]
                sns.kdeplot(x=cluster_data['PC1'], y=cluster_data['PC2'], fill=False, color=cluster_colors[cluster], linestyle='--')

        for i, var in enumerate(self.transformed_columns):
            x2 = self.components[0, i]
            y2 = self.components[1, i]
            color_var = "blue" if any(cat_col in var for cat_col in self.vars_cat) else "black"
            plt.arrow(0, 0, x2 * 5.3, y2 * 5.3, head_width=0.15, head_length=0.15, color=color_var)
            plt.text(x2 * 5.6, y2 * 5.6, var, color=color_var, fontsize=self.font_size, fontweight='bold')

        plt.xlabel(f'PC1 ({self.inertia[0]}%)')
        plt.ylabel(f'PC2 ({self.inertia[1]}%)')
        plt.title(f'{self.title} ({round(sum(self.inertia), 2)}%)')
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        if self.clusters is not None:
            plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()



    def variables(self):
        if self.pca is None or self.components is None:
            self.fit()

        plt.figure(figsize=(5,5))
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        plt.plot(x_circle, y_circle, linestyle='-', color='black')

        k = 1.1
        for i, var in enumerate(self.transformed_columns):
            x2 = self.components[0, i]
            y2 = self.components[1, i]
            color_var = "blue" if any(cat_col in var for cat_col in self.vars_cat) else "black"
            if color_var == "black":
                plt.arrow(0, 0, x2, y2, head_width=0.03, head_length=0.03, color=color_var)
                plt.text(x2 * k, y2 * k, var, color=color_var, fontsize=self.font_size, fontweight='bold')

        plt.xlabel(f'PC1 ({self.inertia[0]}%)')
        plt.ylabel(f'PC2 ({self.inertia[1]}%)')
        plt.title(f'Variables - PCA ({round(sum(self.inertia), 2)}%)')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.show()
