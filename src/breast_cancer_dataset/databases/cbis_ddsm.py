import os
from typing import List

import numpy as np
import pandas as pd


from breast_cancer_dataset.base import GeneralDataBase
from preprocessing.image_processing import crop_image_pipeline
from preprocessing.mask_generator import get_cbis_roi_mask
from utils.config import (
    CBIS_DDSM_DB_PATH, CBIS_DDSM_CONVERTED_DATA_PATH, CBIS_DDSM_PREPROCESSED_DATA_PATH, CBIS_DDSM_MASS_CASE_DESC_TRAIN,
    CBIS_DDSM_MASS_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TEST, CBIS_DDSM_CALC_CASE_DESC_TRAIN, CROP_CONFIG,
    CROP_PARAMS,
)
from utils.functions import get_dirname, get_path


class DatasetCBISDDSM(GeneralDataBase):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos CBISDDSM
    """

    __name__ = 'CBIS-DDSM'

    def __init__(self):
        super(DatasetCBISDDSM, self).__init__(
            ori_dir=CBIS_DDSM_DB_PATH, ori_extension='dcm', dest_extension='png',
            converted_dir=CBIS_DDSM_CONVERTED_DATA_PATH, procesed_dir=CBIS_DDSM_PREPROCESSED_DATA_PATH,
            database_info_file_paths=[CBIS_DDSM_MASS_CASE_DESC_TRAIN, CBIS_DDSM_MASS_CASE_DESC_TEST,
                                      CBIS_DDSM_CALC_CASE_DESC_TRAIN, CBIS_DDSM_CALC_CASE_DESC_TEST]
        )

    def get_df_from_info_files(self) -> pd.DataFrame:
        """
        Función que creará un dataframe con información del dataset CBIS-DDSM. Para cada imagen, se pretende recuperar
        la tipología (IMG_LABEL) detectada, el path de origen de la imagen y su nombre (nombre del estudio que contiene
        el identificador del paciente, el seno sobre el cual se ha realizado la mamografía y el tipo de mamografía).
        该函数将创建一个包含CBIS-DDSM数据集信息的DataFrame。对于每一张图像,其目的是恢复检测到的类型、图像的来源路径及其名称(包含患者标识符的研究名称、进行乳房X光检查的乳房和乳房X光检查的类型)。
        
        :return: pandas dataframe con la base de datos procesada. 返回 处理数据库的Pandas Dataframe
        """

        # Se crea una lista que contendrá la información de los archivos csv del set de datos
        l = []

        # Se iteran los csv con información del set de datos para unificarlos
        # 用数据集信息迭代CSV以统一它们
        print(f'{"=" * 70}\n\tGetting information from database {self.__name__} ({self.IMG_TYPE})\n{"=" * 70}')
        for path in self.database_info_file_paths:
            l.append(pd.read_csv(path))

        df = pd.concat(objs=l, ignore_index=True)

        # Se crea la columna IMG_LABEL que contendrá las tipologías 'BENIGN' y 'MALIGNANT'. Se excluyen los casos de
        # patologias 'benign without callback'.
        #创建包含“良性”和“恶性”类型的img_label列。不包括“良性无回调”疾病的病例。
        df.loc[:, 'IMG_LABEL'] = df.pathology.where(df.pathology != 'BENIGN_WITHOUT_CALLBACK', np.nan)

        # Se crea la columna ABNORMALITY_TYPE que indicará si se trata de una calcificación o de una masa.
        #创建异常类型列，该列将指示这是钙化还是肿块
        df.loc[:, 'ABNORMALITY_TYPE'] = np.where(df['abnormality type'] == 'calcification', 'CALC', 'MASS')

        # El campo que contiene información de la localización de las imagenes sin convertir es 'image file path'.
        # Se suprimen posibles duplicidades en este campo para realizar el procesado una única vez.
        # 包含未转换图像位置信息的字段是“图像文件路径”。删除该字段中可能的重复，以便只执行一次处理。
        df.drop_duplicates(subset=['image file path'], inplace=True)

        # Se obtiene si se trata del seno derecho o izquierdo
        df.loc[:, 'BREAST'] = df['left or right breast']

        # Se crea la columna BREAST_VIEW que indicará si se trata de una imagen CC o MLO
        df.loc[:, 'BREAST_VIEW'] = df['image view']

        # Se crea la columna BREAST_DENSITY que indicará la densidad del seno
        df.loc[:, 'BREAST_DENSITY'] = df.breast_density

        # Se crea la columna filename a partir de la carpeta que contiene cada tipología. Esta columna
        # servirá para unir las imagenes originales con su información.
        df.loc[:, 'FILE_NAME'] = df['image file path'].apply(lambda x: get_dirname(x).split("/")[0])

        # Se crea una columna identificadora de cada registro.
        df.loc[:, 'ID'] = df['image file path'].apply(lambda x: get_dirname(x).split("/")[0])

        return df

    def get_raw_files(self, df: pd.DataFrame, f: callable = lambda x: get_dirname(x).split(os.sep)[-3]) -> pd.DataFrame:
        """
        Función que une los paths de las imagenes del dataset con el dataframe utilizado por la clase.

        :param df: pandas dataframe con la información del dataset de la clase.
        :param f: función utilizada para obtener el campo identificador de cada imagen que permitirá unir cada
                     filepath con su correspondiente hilera en el dataframe.
        :return: Pandas dataframe con la columna que contiene los filepaths de las imagenes.
        """
        return super(DatasetCBISDDSM, self).get_raw_files(df=df, get_id_func=f)

    def get_image_mask(self, func: callable = get_cbis_roi_mask, args: List = None) -> None:
        """
        Función que genera las máscaras del set de datos y une los paths de las imagenes generadas con el dataframe del
        set de datos
        :param func: función para generar las máscaras
        :param args: parámetros a introducir en la función 'func' de los argumentos.
        """
        super(DatasetCBISDDSM, self).get_image_mask(
            func=func, args=[(x.ID, os.path.abspath(x.CONVERTED_MASK),os.path.abspath(CBIS_DDSM_DB_PATH)) for _, x in self.df_desc.iterrows()]
        )


class DatasetCBISDDSMCrop(DatasetCBISDDSM):

    """
        Clase cuyo objetivo consiste en preprocesar los datos de la base de datos CBISDDSM y generar imagenes
        con las regiones de interes del dataset.
        该类的目的是对CBISDDSM数据库的数据进行预处理，并生成带有数据集感兴趣区域的图像。
    """

    IMG_TYPE: str = get_path('CROP', CROP_CONFIG, create=False)

    def preproces_images(self, args: list = None, func: callable = crop_image_pipeline) -> None:
        """
        Función utilizada para realizar el preprocesado de las imagenes recortadas.
        用于对裁剪的图像进行预处理的函数。

        :param func: función utilizada para generar los rois del set de datos
                    用来生成数据集的rois的函数
        :param args: lista de argumentos a introducir en la función 'func'.
                    要在函数中输入的参数列表
        """

        # Se recupera la configuración para realizar el preprocesado de las imagenes.
        p = CROP_PARAMS[CROP_CONFIG]

        # Se crea la lista de argumentos para la función func. En concreto se recupera el path de origen
        # (imagen completa convertida), el path de destino, el path de la máscara, el número de muestras de
        # background, el numero de rois a obtener, el overlap de los rois y el margen de cada roi (zona de roi +
        # background).
        args = list(set([
            (os.path.abspath(x.CONVERTED_IMG), os.path.abspath(x.PROCESSED_IMG), os.path.abspath(x.CONVERTED_MASK), p['N_BACKGROUND'], p['N_ROI'], p['OVERLAP'],
             p['MARGIN']) for _, x in self.df_desc.iterrows()
        ]))

        # Se llama a la función que procesa las imagenes para obtener los rois 调用处理图像以获得ROI的函数
        super(DatasetCBISDDSMCrop, self).preproces_images(args=args, func=func)

        # Se llama a la función que une el path de cada roi con cada imagen original del dataset y con su
        # correspondiente hilera del dataset 调用将每个ROI的路径与数据集的每个原始图像及其相应的数据集行连接起来的函数
        super(DatasetCBISDDSMCrop, self).get_roi_imgs()

    def delete_observations(self) -> None:
        """
        Función para eliminar los registros del dataset al realizar las validaciones.
        """
        pass
