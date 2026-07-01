import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneOut
from scipy import stats


class GestorDatos:
    """encargado de la 'percepción' del agente"""
    """osea, cargar, limpiar y preprocesar los datos"""

    def __init__(self, ruta_csv, separar_outliers=True):
        self.ruta_csv = ruta_csv
        self.separar_outliers = separar_outliers
        self.df = None
        self.df_normal = None
        self.df_outliers = None
        self.df_normal_scaled = None
        self.scaler = None
        self.le_region = None

    def cargar(self):
        self.df = pd.read_csv(self.ruta_csv, encoding="latin-1", sep=";")
        self.df = self.df.rename(columns={"GENERACION_DOM URBANA_TANIO": "GENERACION_DOM_URBANA_TANIO"})
        for col in ["DEPARTAMENTO", "PROVINCIA", "DISTRITO", "REGION_NATURAL", "TIPO_MUNICIPALIDAD"]:
            self.df[col] = self.df[col].str.strip().str.upper()
        return self.df

    @staticmethod
    def _limites_iqr(serie):
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    def detectar_outliers(self):
        lim_inf, lim_sup = self._limites_iqr(self.df["GENERACION_MUN_TANIO"])
        self.df["ES_OUTLIER_GENERACION"] = (
            (self.df["GENERACION_MUN_TANIO"] < lim_inf) | (self.df["GENERACION_MUN_TANIO"] > lim_sup)
        )
        if self.separar_outliers:
            self.df_normal = self.df[~self.df["ES_OUTLIER_GENERACION"]].copy()
            self.df_outliers = self.df[self.df["ES_OUTLIER_GENERACION"]].copy()
        else:
            self.df_normal = self.df.copy()
            self.df_outliers = self.df.iloc[0:0].copy()
        return self.df_normal, self.df_outliers

    def codificar_y_normalizar(self):
        df_encoded = self.df_normal.copy()
        self.le_region = LabelEncoder()
        df_encoded["REGION_NATURAL_COD"] = self.le_region.fit_transform(df_encoded["REGION_NATURAL"])
        df_encoded = pd.get_dummies(
            df_encoded, columns=["TIPO_MUNICIPALIDAD", "CLASIFICACION_MUNICIPAL_MEF"], prefix=["TIPO", "MEF"]
        )
        variables_numericas = ["GENERACION_MUN_TANIO", "GENERACION_PER_CAPITA_MUNICIPAL", "POB_TOTAL_INEI"]
        self.scaler = StandardScaler()
        df_encoded[variables_numericas] = self.scaler.fit_transform(df_encoded[variables_numericas])
        self.df_normal_scaled = df_encoded
        return self.df_normal_scaled


class ModeloClasificacion:
    """aplica k-means para clasificar distritos segun la criticidad ambiental"""

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.modelo = None
        self.mapa_criticidad = None

    def metodo_codo_y_silueta(self, X, rango_k=range(2, 11)):
        inercia, siluetas = [], []
        for k in rango_k:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inercia.append(km.inertia_)
            siluetas.append(silhouette_score(X, labels))
        return list(rango_k), inercia, siluetas

    def entrenar(self, X):
        self.modelo = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        return self.modelo.fit_predict(X)

    def interpretar(self, df_normal, columna_cluster="CLUSTER_CRITICIDAD", columna_ref="GENERACION_MUN_TANIO"):
        resumen = df_normal.groupby(columna_cluster)[columna_ref].mean()
        orden = resumen.sort_values().index.tolist()
        etiquetas = ["Baja", "Media", "Alta"]
        self.mapa_criticidad = dict(zip(orden, etiquetas))
        return self.mapa_criticidad

    @staticmethod
    def clasificar_outliers(df_outliers):
        df_outliers = df_outliers.copy()
        df_outliers["CRITICIDAD"] = "Alta"
        df_outliers["CLUSTER_CRITICIDAD"] = -1
        return df_outliers


class ModeloPrediccion:
    """esto entrena un modelo de regresion lineal por departamento con validacion leave one out
    y genera predicciones futuras con intervalos de confianza"""

    def __init__(self):
        self.resultados = []
        self.modelos = {}

    @staticmethod
    def _intervalo_confianza(X, y, modelo, x_nuevo, confianza=0.95):
        n = len(X)
        x_flat = X.flatten()
        y_pred_train = modelo.predict(X)
        sse = np.sum((y - y_pred_train) ** 2)
        gl = n - 2
        se = np.sqrt(sse / gl) if gl > 0 else np.nan
        x_media = x_flat.mean()
        sxx = np.sum((x_flat - x_media) ** 2)
        se_pred = se * np.sqrt(1 + 1 / n + (x_nuevo - x_media) ** 2 / sxx)
        t_valor = stats.t.ppf((1 + confianza) / 2, gl)
        return t_valor * se_pred

    def entrenar_por_departamento(self, df, columna_grupo="DEPARTAMENTO", columna_anio="ANIO",
                                   columna_objetivo="GENERACION_MUN_TANIO"):
        datos_agrupados = df.groupby([columna_grupo, columna_anio])[columna_objetivo].sum().reset_index()
        loo = LeaveOneOut()

        for grupo in datos_agrupados[columna_grupo].unique():
            datos = datos_agrupados[datos_agrupados[columna_grupo] == grupo].sort_values(columna_anio)
            X = datos[[columna_anio]].values.astype(float)
            y = datos[columna_objetivo].values

            y_real_loo, y_pred_loo = [], []
            for train_idx, test_idx in loo.split(X):
                m = LinearRegression().fit(X[train_idx], y[train_idx])
                y_real_loo.append(y[test_idx][0])
                y_pred_loo.append(m.predict(X[test_idx])[0])

            r2 = r2_score(y_real_loo, y_pred_loo)
            rmse = np.sqrt(mean_squared_error(y_real_loo, y_pred_loo))
            mae = mean_absolute_error(y_real_loo, y_pred_loo)

            modelo_final = LinearRegression().fit(X, y)
            pred_2025 = modelo_final.predict([[2025]])[0]
            pred_2026 = modelo_final.predict([[2026]])[0]
            ic_2025 = self._intervalo_confianza(X, y, modelo_final, 2025)
            ic_2026 = self._intervalo_confianza(X, y, modelo_final, 2026)

            self.resultados.append({
                columna_grupo: grupo, "R2_LOOCV": r2, "RMSE_LOOCV": rmse, "MAE_LOOCV": mae,
                f"{columna_objetivo}_2024": y[-1],
                "PREDICCION_2025": pred_2025, "IC95_2025": ic_2025,
                "PREDICCION_2026": pred_2026, "IC95_2026": ic_2026
            })
            self.modelos[grupo] = {"modelo": modelo_final, "X": X, "y": y}

        return pd.DataFrame(self.resultados).sort_values(f"{columna_objetivo}_2024", ascending=False).reset_index(drop=True)


class AgenteResiduos:
    """Agente inteligente que percibe datos de generacion de residuos
    solidos municipales, los procesa, clasifica distritos según su criticidad
    ambiental usando k means y predice la generación futura por departamento
    (usando regresión lineal con validación LOO-CV)"""

    def __init__(self, ruta_csv, separar_outliers=True):
        self.gestor_datos = GestorDatos(ruta_csv, separar_outliers)
        self.modelo_clasificacion = ModeloClasificacion(n_clusters=3)
        self.modelo_prediccion = ModeloPrediccion()
        self.df_clasificado = None
        self.df_resultados_prediccion = None

    def percibir(self):
        return self.gestor_datos.cargar()

    def procesar(self):
        self.gestor_datos.detectar_outliers()
        return self.gestor_datos.codificar_y_normalizar()

    def clasificar(self):
        X = self.gestor_datos.df_normal_scaled[
            ["POB_TOTAL_INEI", "GENERACION_MUN_TANIO", "GENERACION_PER_CAPITA_MUNICIPAL"]
        ]
        self.gestor_datos.df_normal["CLUSTER_CRITICIDAD"] = self.modelo_clasificacion.entrenar(X)
        self.modelo_clasificacion.interpretar(self.gestor_datos.df_normal)
        self.gestor_datos.df_normal["CRITICIDAD"] = self.gestor_datos.df_normal["CLUSTER_CRITICIDAD"].map(
            self.modelo_clasificacion.mapa_criticidad
        )
        df_outliers_clasif = self.modelo_clasificacion.clasificar_outliers(self.gestor_datos.df_outliers)
        self.df_clasificado = pd.concat([self.gestor_datos.df_normal, df_outliers_clasif], ignore_index=True)
        return self.df_clasificado

    def predecir(self):
        self.df_resultados_prediccion = self.modelo_prediccion.entrenar_por_departamento(self.gestor_datos.df)
        return self.df_resultados_prediccion