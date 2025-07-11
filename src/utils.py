import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.metrics import fbeta_score, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb
from scipy.stats import chi2_contingency


def lgbm_feature_importance(
    df, target_var, max_depth = 5, min_child_samples = 1000, random_state = 123456
):

    # Select the target variable
    X = df.drop(target_var, axis = 1)
    y = df[target_var]

    # List to store categorical variables
    categorical_vars = [col for col in X.columns if X[col].dtype == "object"]

    # Convert categorical variables to 'category' type
    X[categorical_vars] = X[categorical_vars].astype("category")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    import lightgbm as lgb
    # Train the LightGBM model
    model = lgb.LGBMClassifier(
        max_depth = max_depth,
        min_child_samples = min_child_samples,
        random_state = random_state,
    )

    model.fit(X_train, y_train, categorical_feature=categorical_vars)
    'Mirar como crear la VAR de categorical_columns'
    # model.fit(X_train, y_train, categorical_feature=categorical_columns)

    # Make predictions on the test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_prob]

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)

    # Calculate the AUC
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(false_positive_rate, recall)
    print("AUC:", roc_auc)

    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred_binary)
    print("F1 score:", f1)

    # Visualizar la importancia de las características
    lgb.plot_importance(model, importance_type="gain")
    plt.show()

    # Obtain the top 30 features
    feature_importance = pd.DataFrame(
        sorted(zip(model.feature_importances_, X.columns)), columns=["Value", "Feature"]
    )
    feature_importance = feature_importance.sort_values(by="Value", ascending=False)
    top_30_features = feature_importance.head(25)

    # Obtain from df the columns that are in top_30_features
    df_selected = df[top_30_features["Feature"].tolist()]

    # Convert the selected categorical variables to 'category' type
    categorical_vars = [
        col for col in df_selected.columns if df_selected[col].dtype == "object"
    ]
    df_selected[categorical_vars] = df_selected[categorical_vars].astype("category")

    return top_30_features, df_selected


def decision_tree_model(
    target,
    df_selected,
    df,
    min_samples_leaf=15000,
    max_depth=5,
    random_state=123456,
    plot_tree=True,
    class_weight="balanced",
):
    """
    Esta función entrena un modelo de árbol de decisión con un dataframe dado
    y la variable objetivo especificada, realiza predicciones, imprime
    la precisión, AUC, F1 score, traza la matriz de confusión y el árbol de decisión.

    Parámetros:
    target (str): El nombre de la variable objetivo.
    df_selected (DataFrame): El DataFrame a usar para entrenar el modelo.
    df (DataFrame): El DataFrame a usar para obtener la varaible objetivo.

    Return:
    model: El modelo de árbol de decisión entrenado.
    """
    from sklearn.metrics import auc

    # Add target variable
    df_selected[target] = df[target]

    # Preparar los datos
    df_selected = df_selected.dropna(subset=[target])
    X = df_selected.drop(target, axis=1)
    y = df_selected[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar el modelo
    model = DecisionTreeClassifier(
        criterion="gini",
        min_weight_fraction_leaf=0,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    # Calcular y mostrar las métricas
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)

    predict_prob = model.predict_proba(X_test)[:, 1]
    false_positive_rate, recall, thresholds = roc_curve(y_test, predict_prob)
    roc_auc = auc(false_positive_rate, recall)
    print("AUC:", roc_auc)

    f1 = f1_score(y_test, y_pred_binary)
    print("F1 score:", f1)

    print("F2 score:", fbeta_score(y_test, y_pred_binary, average="macro", beta=2))

    # Dibujar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    feature_names = X_train.columns.tolist()
    # Dibujar el árbol de decisión
    plt.figure(figsize=(20, 20))
    tree.plot_tree(
        model, feature_names=feature_names, class_names=["0", "1"], filled=True
    )
    plt.show()

    # Print the decision tree in text format with the name of the features
    print(tree.export_text(model, feature_names=X_train.columns.tolist()))

    return model

def label_encode_variables(df, variables):
    """
    Apply label encoding to the specified variables in a DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to be processed.
    - variables (list): A list of variable names in the DataFrame to be label encoded.

    Returns:
    - df_encoded (pd.DataFrame): The DataFrame with label encoded variables.
    """
    df_encoded = df.copy()  # Create a copy of the original DataFrame

    for var in variables:
        label_encoder = LabelEncoder()
        encoded_var = label_encoder.fit_transform(df_encoded[var])
        df_encoded[var + '_LabelEncoded'] = encoded_var

    return df_encoded


def cramerV(label, x):
    confusion_matrix = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    try:
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider not using bias correction",
                RuntimeWarning)
            v = 0
            print("If condition Met: ", v)
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            print("Else condition Met: ", v)
    except:
        print("inside error")
        v = 0
    return v


def plot_cramer(df):
    cramer = pd.DataFrame(index=df.columns, columns=df.columns)
    for column_of_interest in df.columns:
        try:
            temp = {}

            columns = df.columns
            for j in range(0, len(columns)):
                v = cramerV(df[column_of_interest], df[columns[j]])
                cramer.loc[column_of_interest, columns[j]] = v
                if (column_of_interest == columns[j]):
                    pass
                else:
                    temp[columns[j]] = v
            cramer.fillna(value=np.nan, inplace=True)
        except:
            print('Dropping row:', column_of_interest)
            pass
    plt.figure(figsize=(12, 12))
    sns.heatmap(cramer, annot=True, fmt='.2f')

    plt.title("Cross Correlation plot on Dataframe with Cramer's Correlation Values")
    plt.show()

    return cramer


def plot_lgb_importances(model, plot=True):
    feat_imp = pd.DataFrame({'Features names': model.Feature,
                             'Importancia': model.Value}).sort_values('Importancia', ascending=False).round(1)
    if plot:
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1)
        ax = sns.barplot(x="Importancia", y="Features names", data=feat_imp[0:25], palette='hls')
        ax.bar_label(ax.containers[0])
        plt.title('Feature importance')
        plt.tight_layout()
        plt.gca().set_facecolor('white')
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.show()
    else:
        print(feat_imp.head(10))


def asignar_ccaa(provincia):
    provincias_y_comunidades = {"15":"pais_vasco",
                                "01":"castilla - la_mancha",
                                "02":"comunidad_valenciana",
                                "04":"andalucia",
                                "33":"asturias",
                                "05":"castilla_y_leon",
                                "06":"extremadura",
                                "08":"baleares",
                                "09":"cataluña",
                                "10":"castilla_y_leon",
                                "11":"extremadura",
                                "39":"andalucia",
                                "51":"cantabria",
                                "13":"comunidad_valenciana",
                                "14":"ceuta",
                                "16":"castilla - la_mancha",
                                "72":"andalucia",
                                "76":"castilla - la_mancha",
                                "19":"cataluña",
                                "21":"andalucia",
                                "22":"castilla - la_mancha",
                                "07":"pais_vasco",
                                "23":"andalucia",
                                "26":"aragon",
                                "35":"andalucia",
                                "24":"galicia",
                                "25":"la_rioja",
                                "27":"canarias",
                                "28":"castilla_y_leon",
                                "29":"cataluña",
                                "52":"galicia",
                                "30":"comunidad_de_madrid",
                                "31":"andalucia",
                                "32":"melilla",
                                "34":"region_de_murcia",
                                "36":"comunidad_foral_de_navarra",
                                "38":"galicia",
                                "37":"castilla_y_leon",
                                "40":"galicia",
                                "41":"castilla_y_leon",
                                "42":"canarias",
                                "43":"castilla_y_leon",
                                "44":"andalucia",
                                "45":"castilla_y_leon",
                                "47":"cataluña",
                                "49":"aragon",
                                "50":"castilla - la_mancha",
                                "48":"comunidad_valenciana",
                                "20":"castilla_y_leon",
                                "03":"pais_vasco",
                                "12":"castilla_y_leon",
                                "46":"aragon"
                                }
    return provincias_y_comunidades.get(provincia, 'Desconocida')

def asignar_CCAA(dataframe):
    dataframe['comunidad_autonoma'] = dataframe['provincia'].apply(asignar_ccaa)