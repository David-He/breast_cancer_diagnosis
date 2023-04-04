import os
import sys

import pydicom
import numpy as np

from pathlib import Path
from typing import io
from PIL import Image

from utils.config import LOGGING_DATA_PATH
from utils.functions import get_path, get_filename, get_dirname, get_value_from_args_if_exists


def convert_img(args) -> None:
    """
    Función encargada de convertir las imagenes del formato recibido al formato explícito.

    :param args: Los argumentos deberán ser:
        - Posición 0: (Obligatorio) Ruta la imagen a transformar.
        - Posición 1: (Obligatorio) Ruta de la imagen transformada.
    """
    try:
        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        error_path: io = get_value_from_args_if_exists(args, 3, LOGGING_DATA_PATH, IndexError, KeyError)

        if not (len(args) >= 2):
            raise ValueError('Not enough arguments for convert_dcm_img function. Minimum required arguments: 3')

        img_path: io = args[0]
        dest_path: io = args[1]

        # Para las imagenes dicom este valor permite recuperar las máscaras
        # 对于dicom图像，该值允许检索掩码。
        filter_binary: bool = get_value_from_args_if_exists(args, 2, False, IndexError, TypeError)

        #current_path = os.path.dirname( os.path.abspath(__file__))
        #print(f'{"-" * 75}\n current path: {current_path} \n{"-" * 75}')
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        # 验证转换格式是否正确，要转换的图像是否存在。
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"{img_path} doesn't exists.")
        if os.path.splitext(img_path)[1] not in ['.pgm', '.dcm']:
            raise ValueError('Conversion only available for: pgm, dcm')

        assert not os.path.isfile(dest_path), f'Image converted {dest_path} currently exists'

        # En función del formato de origen se realiza una conversión u otra
        # 根据源格式的不同，会进行一种或另一种转换。
        if os.path.splitext(img_path)[1] == '.dcm':
            convert_dcm_imgs(ori_path=img_path, dest_path=dest_path, filter_binary=filter_binary)
        elif os.path.splitext(img_path)[1] == '.pgm':
            convert_pgm_imgs(ori_path=img_path, dest_path=dest_path)
        else:
            raise KeyError(f'Conversion function for {os.path.splitext(img_path)} not implemented')

    except AssertionError as err:
        if not getattr(sys, 'frozen', False):
            with open(get_path(error_path, f'Conversion Errors (Assertions).txt'), 'a') as f:
                f.write(f'{"=" * 100}\nAssertion Error in convert_img\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(error_path, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_path)}\n{err}\n{"=" * 100}')


def convert_dcm_imgs(ori_path: io, dest_path: io, filter_binary: bool) -> None:
    """
    Función encargada de leer imagenes en formato dcm y convertirlas al formato especificado por el usuario.
    :param ori_path: ruta de origen de la imagen
    :param dest_path: ruta de destino de la imgen
    :param filter_binary: en caso de imagenes dicom, se permite recuperar una máscara para su conversion
    """
    try:
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        if os.path.splitext(dest_path)[1] not in ['.png', '.jpg']:
            raise ValueError('Conversion only available for: png, jpg')

        # se crea el directorio y sus subdirectorios en caso de no existir
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato dcm
        img = pydicom.dcmread(ori_path)

        # Se convierte las imagenes a formato de array
        img_array = img.pixel_array.astype(float)

        # Si se desean obtener las máscaras se contabilizan el número de píxeles únicos de cada imagen.
        if filter_binary:
            assert len(np.unique(img_array)) == 2, f'{ori_path} excluded. Not binary Image'
        else:
            assert len(np.unique(img_array)) > 2, f'{ori_path} excluded. Binary Image.'

        # Se realiza un reescalado de la imagen para obtener los valores entre 0 y 255
        # 图像被重新缩放以获得0到255之间的数值。
        #rescaled_image = (np.maximum(img_array, 0) / max(img_array)) * 255
        rescaled_image = (np.maximum(img_array, 0) / img_array.max()) * 255

        # Se convierte la imagen al ipode datos unsigned de 8 bytes
        # 图像被转换为8字节的无符号ipode数据
        final_image = np.uint8(rescaled_image)

        # Se almacena la imagen
        # 存储图像
        Image.fromarray(final_image).save(dest_path)

    except AssertionError:
        pass


def convert_pgm_imgs(ori_path: io, dest_path: io) -> None:
    """
    Función encargada de leer imagenes en formato pgm y convertirlas al formato especificado por el usuario.
    :param ori_path: ruta de origen de la imagen
    :param dest_path: ruta de destino de la imgen
    """
    # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
    if os.path.splitext(dest_path)[1] not in ['.png', '.jpg']:
        raise ValueError('Conversion only available for: png, jpg')

    # se crea el directorio y sus subdirectorios en caso de no existir
    Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

    # Se lee la información de las imagenes en formato pgm y se almacena en el formato deseado
    img = Image.open(ori_path)
    img.save(dest_path)
