import io
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from itertools import product
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from typing import List
from collections import defaultdict

from data_viz.functions import render_mpl_table, plot_image, create_countplot
from preprocessing.image_processing import full_image_pipeline, crop_image_pipeline
from utils.config import MODEL_FILES, XGB_CONFIG, CLASSIFICATION_METRICS, CLASSIFICATION_DATA_AUGMENTATION_FUNCS, EXPERIMENT
from utils.functions import get_path, search_files, get_filename


sns.set(style='white')
sns.despine()


class DataVisualizer:

    metrics = [f.lower() if type(f) is str else f.__name__ for f in CLASSIFICATION_METRICS.values()] + ['accuracy', 'loss']

    def __init__(self, config: MODEL_FILES):
        self.conf = config

    @staticmethod
    def get_dataframe_from_logs(dirname: io, metrics: list) -> pd.DataFrame:
        """

        Función utilizada para crear gráficas a partir de los historiales generados por keras durante el entrenamiento.

        :param dirname: directorio de almacenado del archivo de salida
        :param metrics: lista de métricas a plotear. Estas métricas deben estar en los logs
        """

        # Lista para almacenar las métricas
        data_list = []

        # Se itera sobre los archivos almacenados en el directorio para almacenarlos en un dataframe. Estos archivos
        # se filtrarán mediante el nombre del test asignado durante el entrenamiento.
        for file in search_files(dirname, 'csv'):

            # Se lee el dataframe
            data = pd.read_csv(file, sep=';')

            # Se recupera el nombre del modelo a partir del nombre asignado al log.
            data.loc[:, ['Model', 'Phase']] = get_filename(file).split('_')
            data.loc[:, ['Weights', 'FrozenLayers']] = file.split(os.sep)[-3:-1]

            data.loc[:, 'epoch'] = data.index + 1

            # Se obtienen aquellas columnas que no contienen en el nombre alguna de las métricas definidas por el
            # usuario para evitar que se pivoten.
            melt_cols = [col for col in data.columns if not any(i in col for i in metrics)]

            # Se pivotan las métricas en una columna que se llamará Metrics_value. El nombre de cada métrica se
            # asignará a la columna Metrics_name.
            data_melt = data.melt(id_vars=melt_cols, var_name='Metrics_name', value_name='Metrics_value')

            # Se añade la columna modo a partir del nombre de la métrica. Por defecto, el callback de keras añade
            # la palabra val a cada una de las métricas, por lo que se utilizará dicha casuística para segregar en
            # entrenamiento y validación.
            data_melt.loc[:, 'Mode'] = np.where(data_melt.Metrics_name.str.contains('val', case=False), 'Val', 'Train')

            # Se añaden las métricas de cada modelo.
            data_list.append(data_melt)

        return pd.concat(data_list, ignore_index=True)

    @staticmethod
    def get_dataframe_from_preds(dirname: io) -> pd.DataFrame:

        l = []
        for file in search_files(dirname, 'csv'):

            df = pd.read_csv(file, sep=';')
            df.loc[:, ['Weight', 'Layer']] = file.split(os.sep)[-3:-1]
            df.rename(columns={'PREDICTION': get_filename(file)}, inplace=True)

            l.append(df)

        return pd.concat(l, ignore_index=True).groupby(['PROCESSED_IMG', 'Weight', 'Layer'], as_index=False).first()

    def get_dataframe_from_dataset_excel(self) -> pd.DataFrame:
        return pd.read_excel(self.conf.model_db_desc_csv, dtype=object, index_col=None)

    @staticmethod
    def plot_model_metrics(plot_params: List[dict], dirname: io = None, filename: io = None, plots_per_line: int = 2):
        """
        Función para representar gráficamente las métricas obtenidas especificadas mediante el parámetro plot_params
        :param plot_params: lista de diccionarios que contiene las especificaciones para generar un grafico
        :param dirname: nombre del directorio en el que se guardará la imagen
        :param filename: nombre del archivo con el que se guardará la imagen
        :param plots_per_line: determina el número de columnas de cada figura
        """

        # Se crea la figura y los subplots
        nrows = (len(plot_params) // plots_per_line) + 1
        figure = plt.Figure(figsize=(15, 10))

        # Se itera cada diccionario almacenado en plot params
        for i, plot_configuration in enumerate(plot_params, 1):

            # Se crea el subplot
            ax = figure.add_subplot(nrows, plots_per_line, i)

            assert 'plot' in plot_configuration, 'debe especificar el diccionario data en plot_params'

            # Se crea un gráfico lineal con las especificiaciones de plot
            sns.lineplot(**plot_configuration['plot'], ax=ax)

            # Configuración de los labels del gráfico
            if plot_configuration.get('labels', False):
                ax.set(xlabel=plot_configuration['labels'].get('xlabel', plot_configuration['plot']['x']),
                       ylabel=plot_configuration['labels'].get('ylabel',  plot_configuration['plot']['y']))

            # Configuración del título del gráfico
            if plot_configuration.get('title', False):
                ax.title.set(**plot_configuration['title'])

            # Configuración de la leyenda del gráfico
            if plot_configuration.get('legend', False):
                ax.legend(**plot_configuration['legend'])

            # Se elimina el grid del grafico
            ax.grid(False)
            sns.despine(ax=ax)

        # Ajuste de la figura y guardado de la imagen
        figure.tight_layout()
        figure.savefig(get_path(dirname, filename))

    def plot_confusion_matrix(self, df: pd.DataFrame, models: list) -> None:
        # En función del número de modelos, se generan n hileras para graficar los resultados. Cada hilera contendrá
        # dos modelos.
        nrows = (len(models) // 2) + 1

        # Se itera para cada fase de entrenamiento/validación de cada modelo.
        for mode, w, layer in product(df.TRAIN_VAL.unique(), df.Weight.unique(), df.Layer.unique()):

            plt_data = df[(df.TRAIN_VAL == mode) & (df.Weight == w) & (df.Layer == layer)].copy()

            if len(plt_data) > 0:
                # Se crea la figura de matplotlib
                fig = plt.figure(figsize=(15, 10))

                # Se iteran los modelos
                for i, col in enumerate(models, 1):

                    # Se crea un subplot.
                    ax = fig.add_subplot(nrows, 2, i)

                    # Se crea la tabla de contingencia a través de las clases verdaderas (columna true_label) y las
                    # predecidas (columna definida por el nombre del modelo)..
                    ct = pd.crosstab(plt_data.IMG_LABEL, plt_data[col], normalize=False)

                    # Se muestra la matriz de confusión.
                    sns.set(font_scale=1)
                    sns.heatmap(ct.reindex(sorted(ct.columns), axis=1), cmap="Blues", annot=True,
                                annot_kws={"size": 15}, fmt="d", cbar=False, ax=ax)

                    # título y eje x del gráfico
                    ax.set_title(f'Modelo: {col}\n{mode}', fontweight='bold', size=14)
                    ax.set(xlabel='Predictions')

                # Se ajustan los subplots
                fig.tight_layout()
                # Se almacena la figura.
                fig.savefig(get_path(self.conf.model_viz_results_confusion_matrix_dir, w, layer,
                                     f'{XGB_CONFIG}_{mode}.jpg'))

    def plot_metrics_table(self, df: pd.DataFrame, models: list, class_metric: bool = True) -> None:
        """
        Función utilizada para generar una imagen con una tabla que contiene las métricas de accuracy, precision,
        recall y f1_score para entrenamiento y validación. Las métricas se calculan a partir del log de predicciones
        generado por un modelo.

        :param input_file: path del log de predicciones
        :param dirname: carpeta en la que se guardarán las imagenes generadas
        :param out_file: nombre del archivo de imagen a generar
        :param class_metric: booleano que sirve para calcular las métricas de precision, recall, accuracy y f1 de cada
                              clase y modelo en caso de ser true. En caso contrario, se generarán las métricas de forma
                              global para cada modelo.
        """

        metrics = ['accuracy', 'precision', 'recall', 'f1']

        for w, l in product(df.Weight.unique(), df.Layer.unique()):

            df_ = df[(df.Weight == w) & (df.Layer == l)].copy()

            if len(df_) > 0:
                # Se crea un dataset que contendrá las métricas accuracy, precision, recall y f1 para train y validación a
                # nivel general. En caso de tener el parametro class_metrics a True, las columnas del dataset serán un
                # multiindice con modelos y clases; en caso contrario, únicamente contendrá el nombre de los modelos.
                metric_df = pd.DataFrame(
                    index=pd.MultiIndex.from_product([df.TRAIN_VAL.unique(), metrics]),
                    columns=pd.MultiIndex.from_product([models, df_.IMG_LABEL.unique().tolist()]) if class_metric
                    else models
                )

                # Se asigna el nombre del índice
                metric_df.index.set_names(['mode', 'metric'], inplace=True)

                for phase, model in product(df_.TRAIN_VAL.unique(), models):

                    df_2 = df_[df_.TRAIN_VAL == phase]

                    if class_metric:
                        # En caso de querer obtener las metricas de cada clase se itera sobre cada una de estas.
                        for class_label in df_.IMG_LABEL.unique():
                            # Para poder generar las métricas deseadas, se considerará cada clase como verdadera
                            # asignandole el valor 1, y el resto de clases con el valor 0. De esta forma, se evaluará
                            # para cada clase. (Técnica one vs all)
                            map_dict = defaultdict(lambda: 0, {class_label: 1})

                            # Creación del dataset de métricas
                            metric_df.loc[(phase,), (model, class_label)] = [
                                round(accuracy_score(df_2.IMG_LABEL.map(map_dict), df_2[model].map(map_dict)) * 100, 2),
                                round(precision_score(df_2.IMG_LABEL.map(map_dict), df_2[model].map(map_dict),
                                                      zero_division=0, average='weighted') * 100, 2),
                                round(recall_score(df_2.IMG_LABEL.map(map_dict), df_2[model].map(map_dict),
                                                   zero_division=0, average='weighted') * 100, 2),
                                round(f1_score(df_2.IMG_LABEL.map(map_dict), df_2[model].map(map_dict),
                                               zero_division=0, average='weighted') * 100, 2)]
                    else:
                        # Creación del dataset de métricas
                        metric_df.loc[(phase,), model] = [
                            round(accuracy_score(df_2.IMG_LABEL.tolist(), df_2[model].tolist()) * 100, 2),
                            round(precision_score(df_2.IMG_LABEL.tolist(), df_2[model].tolist(), zero_division=0,
                                                  average='weighted') * 100, 2),
                            round(recall_score(df_2.IMG_LABEL.tolist(), df_2[model].tolist(), zero_division=0,
                                               average='weighted') * 100, 2),
                            round(f1_score(df_2.IMG_LABEL.tolist(), df_2[model].tolist(), zero_division=0,
                                           average='weighted') * 100, 2)]

                # se resetea el índice para poder mostrar en la tabla si las métricas son de entrenamiento o de
                # validación
                metric_df.reset_index(inplace=True)

                merge_rows = [
                    [(i, 0) for i in range(l, l + len(metrics))] for l in
                    range(metric_df.columns.nlevels, (len(metrics) + 1) * df_.TRAIN_VAL.nunique(), len(metrics))
                ]

                merge_cols = [
                    [(0, i) for i in range(l, l + df_.IMG_LABEL.nunique())]
                    for l in range(2, (len(models) + 1) * df_.IMG_LABEL.nunique(), df_.IMG_LABEL.nunique())
                ]
                merge_cells = [[(0, 0), (1, 0)], [(0, 1), (1, 1)], *merge_rows, *merge_cols] if class_metric else None

                fig, _ = render_mpl_table(metric_df, merge_pos=merge_cells, header_rows=2)
                filename = f'{XGB_CONFIG}_model_metrics{"_marginal"  if class_metric else ""}.jpg'
                fig.savefig(get_path(self.conf.model_viz_results_metrics_dir, w, l, filename))

    def plot_accuracy_plots(self, df: pd.DataFrame, models: list, hue: str, title: str, img_name: str):

        df_grouped = df.groupby(['Weight', 'Layer', 'TRAIN_VAL'], as_index=False). \
            apply(lambda x: pd.Series({m: round(accuracy_score(x.IMG_LABEL, x[m]) * 100, 2) for m in models}))

        df_melt = pd.melt(df_grouped, id_vars=['Weight', 'Layer', 'TRAIN_VAL'], value_vars=models, var_name='model',
                          value_name='accuracy')

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))

        for mode, axes in zip(df_melt.TRAIN_VAL.unique(), ax.flatten()):

            sns.barplot(x='model', y='accuracy', hue=hue, data=df_melt[df_melt.TRAIN_VAL == mode], ax=axes)

            axes.set(xlabel=mode, ylim=[0, 100])

        fig.tight_layout()
        fig.suptitle(title, y=1.05, fontsize=14, fontweight='bold')
        fig.savefig(get_path(self.conf.model_viz_results_accuracy_dir, XGB_CONFIG, f'{img_name}.png'))

    def get_model_time_executions(self, summary_dir: io):

        data = pd.read_csv(search_files(summary_dir, 'csv', in_subdirs=False)[0], sep=';', decimal='.')

        fig, ax = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

        for weights, axes in zip(data.weights.unique(), ax.flatten()):

            data_filtered = data[data.weights == weights]

            data_plot = pd.concat(objs=[
                data_filtered[data_filtered.FT != 'ALL'].groupby(['cnn', 'FT'], as_index=False).time.sum(),
                data_filtered[data_filtered.FT == 'ALL']],
            )

            sns.barplot(x='FT', y='time', hue='cnn', data=data_plot, ax=axes)

            axes.set_title(f'Tiempo de entrenamiento con inicialización de pesos: {weights}')
            axes.set(ylabel='Tiempo (seg)', xlabel='Capas entrenables')

        fig.tight_layout()
        fig.savefig(get_path(self.conf.model_viz_results_dir, 'Comparación tiempos entrenamiento.jpg'))

    def get_model_logs_metrics(self, logs_dir: io):
        """
        Función para generar las métricas de accuracy, f1 score y loss a partir de los logs generados durante la fase
        de entrenamiento de un modelo
        :param logs_dir: directorio en el cual se almacenan los logs
        :param test_name: nombre de la prueba asignada a la fase de entrenamiento
        :param dirname: directorio para almacenar las gráficas
        :param out_filename: nombre del archivo para almacenar las gráficas
        :param train_phases: nombre de las fases de entrenamiento con las que se ha entrenado un modelo
        """

        # Se recupera un dataframe a partir del directorio de logs que contendrá las métricas
        data = self.get_dataframe_from_logs(logs_dir, self.metrics)

        # Se itera sobre las fases con las que se ha entrenado cada modelo
        for weights, layers, phase in product(data.Weights.unique().tolist(), data.FrozenLayers.unique().tolist(),
                                              data.Phase.unique().tolist()):
            # Filtrado de datos
            data_filtered = data[(data.Weights == weights) & (data.Phase == phase) & (data.FrozenLayers == layers)]

            if len(data_filtered) > 0:
                # Se crea la gráfica correspondiente
                self.plot_model_metrics(
                    dirname=self.conf.model_viz_results_model_history_dir,
                    filename=f'Model_history_train_{phase}_{weights}_{layers}.jpg',
                    plot_params=[
                        {
                            'plot': {
                                'data': data_filtered[
                                    data_filtered['Metrics_name'].str.contains(metric, case=False, na=False)],
                                'y': 'Metrics_value',
                                'x': 'epoch',
                                'hue': 'Model',
                                'style': 'Mode',
                            },
                            'labels': {
                                'xlabel': 'Epocas',
                                'ylabel': ylabel,
                            },
                            'title': {
                                'text': f'{title}',
                                'fontsize': 12,
                                'fontweight': 'bold'
                            },
                            'legend': {
                                'fontsize': 'x-small',
                                'frameon': False,
                                'framealpha': 1
                            }
                        } for title, metric, ylabel, in zip(
                            ['Exactitud Modelos', 'Pérdidas Modelos', 'F1_Score Modelos', 'AUC Modelos',
                             'Precisión Modelos', 'Recall Modelos'], self.metrics,
                            ['Exactitud', 'Pérdidas', 'F1', 'AUC', 'Precisión', 'Recall'])
                    ]
                )

    def get_model_predictions_metrics(self, predictions_dir: io):
        """
        Función que permite crear una matriz de confusión a partir de las predicciones generadas por el modelo en la
        fase de entrenamiento para graficarla en un archivo.

       :param dirname: directorio en el que se almacenará la imagen de salida.
       :param out_file: nombre del archivo de salida.
       :param input_file: nombre del archivo csv que contiene las predicciones.
       """

        # Lectura de los datos
        df = self.get_dataframe_from_preds(dirname=predictions_dir)
        models = [c for c in df.columns if c not in ['PROCESSED_IMG', 'IMG_LABEL', 'TRAIN_VAL', 'Weight', 'Layer']]
        self.plot_confusion_matrix(df, models)
        # self.plot_metrics_table(df, models, class_metric=True)
        self.plot_metrics_table(df, models, class_metric=False)
        self.plot_accuracy_plots(df[df.Layer == 'ALL'], models, hue='Weight', img_name='Weight Init Accuracy',
                                 title='Random Initialization vs Imagenet')
        self.plot_accuracy_plots(df[df.Weight == 'imagenet'], models, hue='Layer', img_name='Frozen Layers Accuracy',
                                 title='Impact of the fraction of convolutional blocks fine-tuned on CNN performance')

    def get_data_augmentation_examples(self) -> None:
        """
        Función que permite generar un ejemplo de cada tipo de data augmentation aplicado
        :param out_filepath: ruta del archivo de imagen a generar
        :param example_imag: nombre de una muestra de ejemplo sobre la que se aplicarán las transformaciones propias del
                             data augmentation
        """

        df = self.get_dataframe_from_dataset_excel()

        example_imag = df.loc[random.sample(df.index.tolist(), 1)[0], 'PROCESSED_IMG']

        # Se lee la imagen del path de ejemplo
        image = load_img(example_imag)
        # Se transforma la imagen a formato array
        image = img_to_array(image)
        # Se añade una dimensión para obtener el dato de forma (1, width, height, channels)
        image_ori = np.expand_dims(image, axis=0)

        # Figura y subplots de matplotlib. Debido a que existen 4 transformaciones de data augmentation, se creará un
        # grid con 5 columnas que contendrán cada ejemplo de transformación y la imagen original
        elements = len(CLASSIFICATION_DATA_AUGMENTATION_FUNCS.keys())
        cols = 3
        rows = elements // cols + elements % cols
        fig = plt.figure(figsize=(15, 4 * rows))

        # Se representa la imagen original en el primer subplot.
        plot_image(img=image_ori, title='Imagen Original', ax_=fig.add_subplot(rows, cols, 1))

        # Se iteran las transformaciones
        for i, (k, v) in enumerate(CLASSIFICATION_DATA_AUGMENTATION_FUNCS.items(), 2):

            # Se crea al datagenerator con exclusivamente la transformación a aplicar.
            datagen = ImageDataGenerator(**{k: v}, fill_mode='constant', cval=0)
            # Se recupera la imagen transformada mediante next() del método flow del objeto datagen
            plot_image(img=next(datagen.flow(image_ori)), title=k, ax_=fig.add_subplot(rows, cols, i))

        # Se ajusta la figura
        fig.tight_layout()

        # Se almacena la figura
        plt.savefig(get_path(self.conf.model_viz_data_augm_dir, f'{get_filename(example_imag)}.png'))

    def get_eda_from_df(self) -> None:
        """
        Función que permite representar graficamente el número de observaciones y la proporción de cada una de las
        clases presentes en un dataet. La clase de cada observción debe estar almacenada en una columna cuyo
        nombre sea "class".

        :param dirname: directorio en el que se almacenará la imagen.
        """

        df = self.get_dataframe_from_dataset_excel()

        print(f'{"-" * 75}\n\tGenerando análisis del dataset\n{"-" * 75}')
        title = 'Distribución clases según orígen'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='DATASET', hue='IMG_LABEL', data=df, title=title, file=file)

        title = 'Distribución clases'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='IMG_LABEL', data=df, title=title, file=file)

        title = 'Distribución clases segun train-val'
        file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        create_countplot(x='TRAIN_VAL', hue='IMG_LABEL', data=df, title=title, file=file, norm=True)

        # title = 'Distribución clases segun patología'
        # file = get_path(self.conf.model_viz_eda_dir, f'{title}.png')
        # create_countplot(x='ABNORMALITY_TYPE', hue='IMG_LABEL', data=df, title=title, file=file, norm=True)
        # print(f'{"-" * 75}\n\tAnálisis del dataset finalizado en {self.conf.model_viz_eda_dir}\n{"-" * 75}')

    def get_preprocessing_examples(self) -> None:

        df = self.get_dataframe_from_dataset_excel().assign(example_dir=None)
        photos = []
        for dataset in df.DATASET.unique():
            photos += random.sample(df[df.DATASET == dataset].index.tolist(), 5)

        df.loc[photos, 'example_dir'] = df.loc[photos, :].apply(
            lambda x: get_path(self.conf.model_viz_preprocesing_dir, x.DATASET, get_filename(x.PROCESSED_IMG),
                               f'{get_filename(x.PROCESSED_IMG)}.png'),
            axis=1
        )

        if EXPERIMENT == 'COMPLETE_IMAGE':
            for _, r in df[df.example_dir.notnull()].iterrows():
                full_image_pipeline([r.CONVERTED_IMG, r.example_dir, True])
        elif EXPERIMENT == 'PATCHES':
            for _, r in df[df.example_dir.notnull()].iterrows():
                crop_image_pipeline([r.CONVERTED_IMG, r.example_dir, r.X_MAX, r.Y_MAX, r.X_MIN, r.Y_MIN, True])
        elif EXPERIMENT == 'MASK':
            pass
        else:
            raise ValueError(f"Function {EXPERIMENT} doesn't defined")