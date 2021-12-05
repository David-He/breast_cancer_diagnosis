import os
import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.breast_cancer_dataset.base import GeneralDataBase
from src.preprocessing.image_processing import crop_image_pipeline
from src.utils.config import (
    INBREAST_DB_PATH, INBREAST_CONVERTED_DATA_PATH, INBREAST_PREPROCESSED_DATA_PATH, INBREAST_CASE_DESC
)
from src.utils.functions import search_files, get_filename, get_path


class DatasetINBreast(GeneralDataBase):

    __name__ = 'INBreast'

    def __init__(self):
        super().__init__(
            ori_dir=INBREAST_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=INBREAST_CONVERTED_DATA_PATH, procesed_dir=INBREAST_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[INBREAST_CASE_DESC]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificaros
        print(f'{"-" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"-" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_excel(path, skipfooter=2))
        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGNA' y 'MALIGNA' en función de la puntición
        # asignada por BIRADS. Los codigos 2 se asignan a la clase benigna; los 4b, 4c, 5 y 6 a la clase maligna.
        # Se excluyen los códigos 0 (Incompletos), 3 (Probablemente benigno), 4a (Probablemente maligno (2-9%).
        # noinspection PyTypeChecker
        df.loc[:, 'IMG_LABEL'] = np.where(
            df['Bi-Rads'].astype(str).isin(['2']), 'BENIGN',
            np.where(df['Bi-Rads'].astype(str).isin(['4b', '4c', '5', '6']), 'MALIGNANT', None)
        )

        # Se suprimen los casos que no contienen ninguna patología
        print(f'\tExcluding {len(df[df.IMG_LABEL.isnull()].index.drop_duplicates())} samples without pathologies.')
        df.drop(index=df[df.IMG_LABEL.isnull()].index, inplace=True)

        return df

    def add_extra_columns(self, df: pd.DataFrame):
        # Se crea la columna Breast que indicará si se trata de una imagen del seno derecho (Right) o izquierdo (Left).
        df.loc[:, 'BREAST'] = np.where(df.Laterality == 'R', 'RIGHT', 'LEFT')

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df.View

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        # noinspection PyTypeChecker
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['Mass '] == 'X', 'MASS', np.where(df.Micros == 'X', 'CALC', None))

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.ACR

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        :return:
        """

        # Se crea la columna ID para poder linkar la información del excel con la información de las imagenes
        # almacenadas en la carpeta RAW. En este caso, se utilizará el campo File Name
        df.loc[:, 'ID'] = df['File Name'].astype(str)

        # Se recuperan los paths de las imagenes almacenadas con el formato específico (por defecto dcm) en la carpeta
        # de origen (por defecto INBREAST_DB_PATH)
        db_files_df = pd.DataFrame(data=search_files(self.ori_dir, self.ori_extension), columns=['RAW_IMG'])

        # Se procesa la columna ori path para poder lincar cada path con los datos del excel. Para ello, se separa
        # los nombres de cara archivo a partir del símbolo _ y se obtiene la primera posición.
        db_files_df.loc[:, 'ID'] = db_files_df.RAW_IMG.apply(lambda x: get_filename(x).split('_')[0])

        # Se crea la columna RAW_IMG con el path de la imagen original
        df_def = pd.merge(left=df, right=db_files_df, on='ID', how='left')

        print(f'\t{len(df_def.RAW_IMG.unique())} image paths available in database')

        # Se crea la clumna PREPROCESSED_IMG en la que se volcarán las imagenes preprocesadas
        df_def.loc[:, 'PREPROCESSED_IMG'] = df_def.apply(
            lambda x: get_path(self.procesed_dir, x.IMG_LABEL, x.IMG_TYPE,
                               f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        # Se crea la clumna CONVERTED_IMG en la que se volcarán las imagenes convertidas de formato
        df_def.loc[:, 'CONVERTED_IMG'] = df_def.apply(
            lambda x: get_path(self.conversion_dir, x.IMG_LABEL, x.IMG_TYPE,
                               f'{get_filename(x.RAW_IMG)}.{self.dest_extension}'), axis=1
        )

        return df_def[self.DF_COLS]


class DatasetINBreastCROP(DatasetINBreast):

    IMG_TYPE: str = 'CROP'
    DF_COLS = [
        'ID', 'DATASET', 'BREAST', 'BREAST_VIEW', 'BREAST_DENSITY', 'ABNORMALITY_TYPE', 'IMG_TYPE', 'RAW_IMG',
        'CONVERTED_IMG', 'PREPROCESSED_IMG', 'X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN', 'IMG_LABEL'
    ]

    def add_extra_columns(self, df: pd.DataFrame):
        super(DatasetINBreastCROP, self).add_extra_columns(df)
        for col in ['X_MAX', 'Y_MAX', 'X_MIN', 'Y_MIN']:
            df.loc[:, col] = None

    def clean_dataframe(self):
        super(DatasetINBreastCROP, self).clean_dataframe()
        self.df_desc = self.df_desc.groupby('CONVERTED_IMG', as_index=False).first()

    def preproces_images(self, show_example: bool = False) -> None:
        """
        Función utilizara para realizar el preprocesado de las imagenes completas.

        :param show_example: booleano para almacenar 5 ejemplos aleatorios en la carpeta de resultados para la
        prueba realizada.
        """

        preprocessed_imgs = pd.DataFrame(
            data=search_files(file=f'{self.conversion_dir}{os.sep}**{os.sep}{self.IMG_TYPE}',
                              ext=self.dest_extension),
            columns=['CONVERTED_IMG']
        )
        print(f'{"-" * 75}\n\tStarting preprocessing of {len(preprocessed_imgs)} images')

        args = [(row.CONVERTED_IMG, row.PREPROCESSED_IMG, row.X_MAX, row.Y_MAX, row.X_MIN, row.Y_MIN) for _, row in
                self.df_desc.iterrows()]

        with Pool(processes=cpu_count() - 2) as pool:
            results = tqdm(pool.imap(crop_image_pipeline, args), total=len(args), desc='preprocessing crop images')
            tuple(results)

        # Se recuperan las imagenes modificadas y se crea un dataframe
        proc_imgs = list(
            search_files(file=f'{self.procesed_dir}{os.sep}**{os.sep}{self.IMG_TYPE}', ext=self.dest_extension)
        )
        print(f'\tProcessed {len(proc_imgs)} images.\n{"-" * 75}')