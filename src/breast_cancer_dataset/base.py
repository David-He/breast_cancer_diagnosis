import os
import warnings

import tqdm
import cv2

import pandas as pd
import numpy as np

from typing import io, List
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from albumentations import Compose

from preprocessing.image_conversion import convert_img
from preprocessing.image_processing import full_image_pipeline
from utils.functions import search_files, get_filename, get_path


class GeneralDataBase:

    """
        Clase padre con las funcionalidades básicas que serán heredadas por los datasets de entrenamiento.
    """

    __name__ = 'GeneralDataBase'
    IMG_TYPE: str = 'FULL'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'IMG_TYPE','ABNORMALITY_TYPE', 'FILE_NAME', 'RAW_IMG', 'CONVERTED_IMG',
        'PROCESSED_IMG', 'CONVERTED_MASK', 'PROCESSED_MASK', 'IMG_LABEL',
    ]
    df_desc = pd.DataFrame(columns=DF_COLS, index=[0])

    def __init__(self, ori_dir: io, ori_extension: str, dest_extension: str, converted_dir: io, procesed_dir: io,
                 database_info_file_paths: List[io]):
        """
        :param ori_dir: directorio con las imagenes 原有图像目录
        :param ori_extension: extensión de las imagenes del dataset数据集图像扩展名
        :param dest_extension: extensión final de las imagenes convertidas 转换图像的扩展名
        :param converted_dir: directorio en el que se volcarán las imagenes convertidas 转换的图像将被转储到的目录
        :param procesed_dir: directorio en el que se volcarán las imagenes procesadas 处理的图像将被转储到的目录
        :param database_info_file_paths: lista con los ficheros que contienen información del dataset 包含数据集信息的文件列表
        """

        for p in database_info_file_paths:
            assert os.path.exists(p), f"Directory {p} doesn't exists."

        assert os.path.exists(ori_dir), f"Directory {ori_dir} doesn't exists."

        self.ori_extension = ori_extension
        self.dest_extension = dest_extension
        self.ori_dir = ori_dir
        self.conversion_dir = converted_dir
        self.procesed_dir = procesed_dir
        self.database_info_file_paths = database_info_file_paths

    def get_df_from_info_files(self) -> pd.DataFrame:
        """
        Función que procesa los archivos de entrada para generar el dataframe de cada base. Esta función, deberá
        ser sobreesctira por las clases hija debido a que cada set de datos tendrá sus peculiaridades.
        :return: dataframe con la información procesada.
        """
        pass

    def add_dataset_columns(self, df: pd.DataFrame) -> None:
        """
        Función que añade dos columnas fijas para el dataframe de la clase que contiene los datos con información
        referente al dataset utilizado.
        :param df: dataframe al que se añadirán las colmnas
        """
        # Se crea una columna con información acerca de qué dataset se trata.
        df.loc[:, 'DATASET'] = self.__name__

        # Se crea la columna IMG_TYPE que indicará si se trata de una imagen completa (FULL) o bien de una imagen
        # recortada (CROP)
        df.loc[:, 'IMG_TYPE'] = self.IMG_TYPE

    @staticmethod
    def add_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Función que será sobreescrita por cada clase hija con el objetivo de añadir aquellas columnas que sean
        necesarias.
        :param df: dataframe al que se añadirán las columns
        :return: pandas dataframe con las columnas añadidas.
        """
        return df

    def get_raw_files(self, df: pd.DataFrame, get_id_func: callable = lambda x: get_filename(x)) -> pd.DataFrame:
        """
        Función genérica para recuperar las rutas dónde se almacenan las imágenes de cada set de datos
        检索每个数据集的图像存储路径的通用函数。
        :param df: pandas dataframe con la  'FILE_NAME' utilizada para unir los datos originales con el path de
                  almacenado de cada imagen.
                  带有 "FILE_NAME "的pandas数据帧，用于连接原始数据和每个图像的存储路径。
        :param get_id_func: callable utilizado para crear un identificador único que permita unir la ruta de cada imagen
                            con la columna FILE_NAME de def.
                            回调方法，用于创建一个唯一的标识符，将每个图像的路径与def的FILE_NAME列联系起来。

        :return: Pandas dataframe con la columna RAW_IMG especificando la ruta de almacenado de cada imagen
                返回带有RAW_IMG列的Pandas数据框架，指定每张图片的存储路径。
        """

        # Se recuperan los paths de las imagenes almacenadas en cada dataset (self.ori_dir) con el formato específico
        # de cada base de datos (self.ori_extension)
        db_files_df = pd.DataFrame(data=search_files(self.ori_dir, self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder unir cada path con los datos del excel.
        db_files_df.loc[:, 'FILE_NAME'] = db_files_df.RAW_IMG.apply(get_id_func).astype(str)

        # Se une la información del dataset original con las rutas almacenadas
        df_def = pd.merge(left=df, right=db_files_df, on='FILE_NAME', how='left')

        return df_def

    def add_img_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Función para añadir los paths de las imágenes que serán convertidas y procesadas
        :param df: pandas dataframe al que se añadiran las columnas
        :return: dataframe con las columas PROCESSED_IMG (ruta destino imagen procesada); CONVERTED_IMG (ruta destino
                 imagen convertida a formato png)
        """

        # Se crea la clumna PROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df.loc[:, 'PROCESSED_IMG'] = df.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_TYPE, f'{x.ID}.{self.dest_extension}'), axis=1
        )

        # Se crea la columna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df.loc[:, 'CONVERTED_IMG'] = df.apply(
            lambda x: get_path(self.conversion_dir, 'FULL', f'{x.FILE_NAME}.{self.dest_extension}'), axis=1
        )

        return df

    def add_mask_files(self, df: pd.DataFrame):
        """
        Función para añadir los paths de las mascaras que serán convertidas y procesadas
        :param df: pandas dataframe al que se añadiran las columnas
        :return: dataframe con las columas CONVERTED_MASK (ruta destino a mascara convertida) y PROCESSED_MASK
                 (ruta destino de la mascara procesada)
        """
        # Se crea la columna CONVERTED_MASK en la que se volcarán las mascaras convertidas de formato
        df.loc[:, 'CONVERTED_MASK'] = df.apply(
            lambda x: get_path(self.conversion_dir, f'MASK', f'{x.ID}.png'), axis=1
        )
        # Se crea la columna PROCESSED_MASK en la que se volcarán las mascaras procesadas
        df.loc[:, 'PROCESSED_MASK'] = df.apply(
            lambda x: get_path(self.procesed_dir, f'MASK', f'{x.FILE_NAME}.png'), axis=1
        )

    def clean_dataframe(self):
        """
        Función para limpiar la base de datos de duplicados y patologias ambiguas.
        """

        # Se descartarán aquellas imagenes completas que presenten más de una tipología. (por ejemplo, el seno presenta
        # una zona benigna y otra maligna).
        duplicated_tags = self.df_desc.groupby('ID').IMG_LABEL.nunique()
        print(f'\tExcluding {len(duplicated_tags[duplicated_tags > 1]) * 2} samples for ambiguous pathologys')
        self.df_desc.drop(
            index=self.df_desc[self.df_desc.ID.isin(duplicated_tags[duplicated_tags > 1].index.tolist())].index,
            inplace=True
        )

        # Se descartan id's duplicados
        print(f'\tExcluding {len(self.df_desc[self.df_desc.ID.duplicated()])} samples duplicated pathologys')
        self.df_desc.drop(index=self.df_desc[self.df_desc.ID.duplicated()].index, inplace=True)

    def get_image_mask(self, func: callable = None, args: List = None):
        """
        Función para procesar las máscaras de cada instancia mediante el uso de paralelismos
        :param func: función de preprocesado de las mascasas
        :param args: argumentos de la función de preprocesado
        """

        print(f'{"-" * 75}\n\tGetting masks: {self.df_desc.CONVERTED_MASK.nunique()} existing files.')
        # Se crea un pool de multihilos para realizar la tarea de forma paralelizada.
        # 多线程池被创建，以并行地执行该任务
        #with ThreadPool(1) as pool:
        with ThreadPool(processes=cpu_count() - 2) as pool:        
            results = tqdm(pool.imap(func, args), total=len(args), desc='getting mask files')
            tuple(results)

        # Se recupera información del número de mascaras procesadas
        completed = list(search_files(
            file=f'{self.conversion_dir}{os.sep}**{os.sep}MASK', ext='png', in_subdirs=False))
        print(f"\tGenerated {len(completed)} masks.\n{'-' * 75}")

    def convert_images(self, func: callable = convert_img, args: List = None) -> None:
        """
        Función para convertir las imagenes de cada instancia mediante el uso de paralelismos
        :param func: función de conversión de las imagenes
        :param args: argumentos de la función de conversion
        """

        print(f'{"-" * 75}\n\tStarting conversion: {self.df_desc.CONVERTED_IMG.nunique()} {self.ori_extension} files.')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        if args is None:
            args = list(set([(os.path.abspath(row.RAW_IMG), os.path.abspath(row.CONVERTED_IMG)) for _, row in self.df_desc.iterrows()]))

        # Se crea un pool de multihilos para realizar la tarea de conversión de forma paralelizada.
        # 多线程池被创建，以并行地执行转换任务。
        #with ThreadPool(1) as pool:
        with ThreadPool(processes=cpu_count() - 2) as pool:        
            results = tqdm(pool.imap(func, args), total=len(args), desc='converting images')
            tuple(results)

        # Se recuperan las imagenes modificadas para informar al usuario
        completed = list(search_files(
            file=f'{self.conversion_dir}{os.sep}**{os.sep}FULL', ext=self.dest_extension, in_subdirs=False))
        print(f"\tConverted {len(completed)} images to {self.dest_extension} format.\n{'-' * 75}")

    def preproces_images(self, args: list = None, func: callable = full_image_pipeline) -> None:
        """
        Función para procesar las imagenes de cada instancia mediante el uso de paralelismos
        通过使用并行性来处理每个实例的图像的函数。
        :param func: función de preprocesado de las imagenes 图像预处理功能
        :param args: argumentos de la función de preprocesado 预处理函数的参数
        """
        print(f'{"-" * 75}\n\tStarting preprocessing of {self.df_desc.PROCESSED_IMG.nunique()} images')

        # Se crea el iterador con los argumentos necesarios para realizar la función a través de un multiproceso.
        if args is None:
            args = list(set([
                (x.CONVERTED_IMG, x.PROCESSED_IMG, False, x.PROCESSED_MASK, x.CONVERTED_MASK) for _, x in
                self.df_desc.iterrows()
            ]))

        # Se crea un pool de multihilos para realizar la tarea de procesado de forma paralelizada.
        #with ThreadPool(processes=cpu_count() - 2) as pool:
        #with ThreadPool(1) as pool:
        with ThreadPool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(func, args), total=len(args), desc='preprocessing full images')
            tuple(results)

        # Se recuperan las imagenes procesadas para informar al usuario
        completed = list(search_files(
            file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension, in_subdirs=False))
        print(f'\tProcessed {len(completed)} images.\n{"-" * 75}')

    def get_roi_imgs(self):
        """

        Función para unir el set de datos final con la ruta de los rois de cada imagen creando un identificador unico

        """
        # Se recuperan los rois generados
        croped_imgs = pd.DataFrame(
            data=search_files(get_path(self.procesed_dir, self.IMG_TYPE), ext=self.dest_extension, in_subdirs=False),
            columns=['FILE']
        )
        # Para cada roi se recupera el nombre de archivo original, el número de recorte generado y si se trata de
        # un roi de background o un roi de lesión.
        croped_imgs.loc[:, 'FILE_NAME'] = croped_imgs.FILE.apply(lambda x: "_".join(get_filename(x).split('_')[1:-1]))
        croped_imgs.loc[:, 'N_CROP'] = croped_imgs.FILE.apply(lambda x: get_filename(x).split('_')[-1])
        croped_imgs.loc[:, 'LABEL_BACKGROUND_CROP'] = croped_imgs.FILE.apply(lambda x: get_filename(x).split('_')[0])

        # Se une la información original con cada roi
        self.df_desc = pd.merge(left=self.df_desc, right=croped_imgs, on=['FILE_NAME'], how='left')

        # Se suprimen los casos que no contienen ningún recorte
        print(f'\tDeleting {len(self.df_desc[self.df_desc.FILE.isnull()])} samples without cropped regions')
        self.df_desc.drop(index=self.df_desc[self.df_desc.FILE.isnull()].index, inplace=True)

        # Se modifica el identificador único de la imagen
        self.df_desc.loc[:, 'ID'] = self.df_desc.ID + '_' + self.df_desc.N_CROP

        # Se modifica la columna PROCESSED_IMG
        self.df_desc.loc[:, 'PROCESSED_IMG'] = self.df_desc.FILE

        # Se modifica la patología dependiendo de si es background o no
        self.df_desc.loc[:, 'IMG_LABEL'] = self.df_desc.IMG_LABEL.where(
            self.df_desc.LABEL_BACKGROUND_CROP == 'roi', 'BACKGROUND'
        )

    def delete_observations(self):
        """
        Función para eliminar imagenes una vez ser termine el pipeline de procesado.
        Se eliminarán aquellas imagenes cuyo procesado haya fallado juntamente con aquellas imagenes cuya máscara
        haya fallado el procesado.
        :return:
        """

        # Se eliminan los id's con imagenes no procesadas
        proc_imgs = list(search_files(get_path(self.procesed_dir, 'FULL'), ext=self.dest_extension, in_subdirs=False))
        print(f'\tFailed processing of {len(self.df_desc[~self.df_desc.PROCESSED_IMG.isin(proc_imgs)])} images\n'
              f'{"-" * 75}')
        self.df_desc.drop(index=self.df_desc[~self.df_desc.PROCESSED_IMG.isin(proc_imgs)].index, inplace=True)

        # Se eliminan los id's con mascaras no procesadas
        proc_mask = list(search_files(get_path(self.procesed_dir, 'MASK'), ext=self.dest_extension, in_subdirs=False))
        print(f'\tFailed processing of {len(self.df_desc[~self.df_desc.PROCESSED_MASK.isin(proc_mask)])} masks\n'
              f'{"-" * 75}')
        self.df_desc.drop(index=self.df_desc[~self.df_desc.PROCESSED_MASK.isin(proc_mask)].index, inplace=True)

    def start_pipeline(self):
        """
            PIPELINE PARA LA CREACION DE CADA DATASET
        """

        # Funciones para obtener el dataframe de los ficheros planos
        df = self.get_df_from_info_files()

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        # Se suprimen los casos cuya patología no sea masas 病理不是肿块的病例被删除
        # 暂时保留所有类型
        #print(f'\tExcluding {len(df[df.ABNORMALITY_TYPE != "MASS"].index.drop_duplicates())} non mass pathologies.')
        #df.drop(index=df[df.ABNORMALITY_TYPE != 'MASS'].index, inplace=True)

        # Se añaden columnas informativas sobre la base de datos utilizada根据使用的数据库添加信息列
        self.add_dataset_columns(df)

        # Se realiza la búsqueda de las imagenes crudas
        # 搜索原始图像
        df_with_raw_img = self.get_raw_files(df)

        # Se añaden columnas adicionales para completar las columnas del dataframe
        # 添加扩展列
        df_def = self.add_extra_columns(df_with_raw_img)

        # Se añaden las columnas con el destino de las máscaras.
        # 添加掩码图片的目的地址列
        self.add_mask_files(df_def)

        # Se añaden las columnas con el destino de las imagenes
        # 添加图像的目的地址列
        self.df_desc = self.add_img_files(df_def)

        # Se limpia el dataframe de posibles duplicidades
        # 从数据中清理可能重复的部分        
        self.clean_dataframe()

        # Se realiza la conversión de las imagenes
        # 以下执行图像转换
        
        self.convert_images()

        # Se obtienen las mascaras
        # 获得Mask文件
        self.get_image_mask()

        # Se preprocesan las imagenes.
        self.preproces_images()

        # Se eliminan las imagenes que no hayan podido ser procesadas y se obtiene un único registro a nivel de label
        # e imagen. 无法处理的图像被删除，并在标签和图像级别获得单个记录。
        self.delete_observations()


class SegmentationDataset:

    """
        Clase para generar un iterador que aplique las mismas transformaciones de data augmentation a las imagenes
        y a las máscaras
    """

    def __init__(self, df: pd.DataFrame, img_col: str, mask_col: str, augmentation: Compose = None):

        assert img_col in df.columns, f'{img_col} not in df'
        assert mask_col in df.columns, f'{mask_col} not in df'

        self.df = self._filter_valid_filepaths(df.copy(), img_col)
        self.img_col = img_col
        self.mask_col = mask_col
        self.augmentation = augmentation

    def __getitem__(self, i: int):

        """
        Función get item. Se leeran las imagenes y las máscaras para aplicarles la función de data augmentation
        :param i: número de índice de la imagen
        :return: tupla con la imagen y la mascara transforamdas
        """

        # read data
        image = cv2.cvtColor(cv2.imread(self.df.iloc[i][self.img_col]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.df.iloc[i][self.mask_col], cv2.IMREAD_GRAYSCALE)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _filter_valid_filepaths(df: pd.DataFrame, x_col: str):
        """
        Se filtran únicamente los paths validos

        :param df: Pandas dataframe con los filepaths en una columna
        :param x_col: columna del dataframe que contiene los filepaths
        :return: paths absolutos de las imagenes eliminando los paths invalidos o inexistentes
        """

        mask = df[x_col].apply(lambda x: os.path.isfile(x))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]


class ClassificationDataset:
    """

        Clase para generar un iterador que aplique transformaciones de data augmentation realizadas mediante la libreria
        albumentations

    """
    def __init__(self, df: pd.DataFrame, x_col: str, y_col: str = None, augmentation: Compose = None,
                 class_mode: str = 'categorical', classes: list = None):

        assert x_col in df.columns, f'{x_col} not in df'

        self.df = self._filter_valid_filepaths(df.copy(), x_col)
        self.x_col = x_col
        self.augmentation = augmentation
        self.filenames = self.df[x_col].values.tolist()

        if class_mode not in ['categorical', 'binary', 'sparse']:
            raise NotImplementedError(
                f'{class_mode} not implemented. Available class_mode params: categorical, binary, sparse'
            )

        if y_col:
            assert y_col in df.columns, f'{y_col} not in df'

            if (class_mode == 'binary') & (df[y_col].nunique() > 2):
                raise ValueError(f'Incorrect number of classes for binary mode')

            if classes is None:
                classes = sorted(df[y_col].unique().tolist())

            self.y_col = y_col
            self.class_indices = {lbl: indx for indx, lbl in enumerate(classes)}
            self.classes = self.df[y_col].values.tolist()
            self.class_list = classes
            self.class_mode = class_mode

            self.get_class()
        else:
            self.y_col = None
            self.class_indices = None
            self.classes = None
            self.class_list = None
            self.class_mode = None

    def get_class(self):
        """
        Función para generar las etiquetas de clase en función del tipo indicado por el usuario.
        """
        if self.class_mode == 'binary' or self.class_mode == 'sparse':
            self.df.loc[:, self.y_col] = self.df[self.y_col].map(self.class_indices).astype(np.float32)
        elif self.class_mode == 'categorical':
            self.df.loc[:, self.y_col] = pd.get_dummies(
                self.df, columns=[self.y_col], dtype=np.float32, prefix='dummy_'
            )[[f'dummy__{c}' for c in self.class_list]].apply(lambda x: list(x), axis=1)

    def __getitem__(self, i):
        """
        Función get item. Se leeran las imagenes para aplicarles la función de data augmentation
        :param i: número de índice de la imagen
        :return: timagen transformada juntamente con su etiqueta de clase en caso de existir
        """

        # read data
        image = cv2.cvtColor(cv2.imread(self.df.iloc[i][self.x_col]), cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.y_col:
            return image, np.float32(self.df.iloc[i][self.y_col])
        else:
            return image

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _filter_valid_filepaths(df, x_col):
        """
        Se filtran únicamente los paths validos

        :param df: Pandas dataframe con los filepaths en una columna
        :param x_col: columna del dataframe que contiene los filepaths
        :return: paths absolutos de las imagenes eliminando los paths invalidos o inexistentes
        """

        mask = df[x_col].apply(lambda x: os.path.isfile(x))
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        return df[mask]


class Dataloder(Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, seed=0):
        self.dataset = dataset
        self.filenames = dataset.filenames
        self.class_indices = dataset.class_indices
        self.indexes = np.arange(len(dataset))

        if dataset.classes:
            self.classes = dataset.classes

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        data = []
        for j in range(i * self.batch_size, (i + 1) * self.batch_size):
            data.append(self.dataset[self.indexes[j]])

        # transpose list of lists
        if self.dataset.y_col:
            batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        else:
            batch = [np.stack(samples, axis=0) for samples in zip(data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.RandomState(seed=self.seed).permutation(self.indexes)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
