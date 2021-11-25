from pathlib import Path

import os
import pydicom
import numpy as np


from typing import io
from PIL import Image

from utils.config import LOGGING_DATA_PATH
from utils.functions import get_path, get_filename, get_dirname


def convert_img(args) -> None:
    """
    Función encargada de convertir las imagenes del formato recibido al formato explícito.

    :param args: Los argumentos deberán ser:
        - Posición 0: (Obligatorio) Ruta la imagen a transformar.
        - Posición 1: (Obligatorio) Ruta de la imagen transformada.
    """
    try:
        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        assert len(args) >= 2, 'Not enough arguments for convert_dcm_img function. Minimum required arguments: 3'
        img_path: io = args[0]
        dest_path: io = args[1]

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.isfile(img_path), f"{img_path} doesn't exists."
        assert os.path.splitext(img_path)[1] in ['.pgm', '.dcm'], f'Conversion only available for: png, jpg'
        assert not os.path.isfile(dest_path), f'Image converted {dest_path} currently exists'

        if os.path.splitext(img_path)[1] == '.dcm':
            convert_dcm_imgs(ori_path=img_path, dest_path=dest_path)
        elif os.path.splitext(img_path)[1] == '.pgm':
            convert_pgm_imgs(ori_path=img_path, dest_path=dest_path)
        else:
            raise KeyError(f'Conversion function for {os.path.splitext(img_path)} not implemented')

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in convert_img\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_path)}\n{err}\n{"=" * 100}')


def convert_dcm_imgs(ori_path: io, dest_path: io) -> None:
    """
    Función encargada de leer imagenes en formato dcm y convertirlas al formato especificado por el usuario.
    """
    try:
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.splitext(dest_path)[1] in ['.png', '.jpg'], f'Conversion only available for: png, jpg'

        # se crea el directorio y sus subdirectorios en caso de no existir
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato dcm
        img = pydicom.dcmread(ori_path)

        # Se convierte las imagenes a formato de array
        img_array = img.pixel_array.astype(float)

        # Se realiza un reescalado de la imagen para obtener los valores entre 0 y 255
        rescaled_image = (np.maximum(img_array, 0) / img_array.max()) * 255

        # Se convierte la imagen al ipode datos unsigned de 8 bytes
        final_image = np.uint8(rescaled_image)

        # Se almacena la imagen
        Image.fromarray(final_image).save(dest_path)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in convert_dcm_imgs\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(ori_path)}\n{err}\n{"=" * 100}')


def convert_pgm_imgs(ori_path: io, dest_path: io) -> None:
    """
    Función encargada de leer imagenes en formato pgm y convertirlas al formato especificado por el usuario.
    """
    try:
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        assert os.path.splitext(dest_path)[1] in ['.png', '.jpg'], f'Conversion only available for: png, jpg'

        # se crea el directorio y sus subdirectorios en caso de no existir
        Path(get_dirname(dest_path)).mkdir(parents=True, exist_ok=True)

        # Se lee la información de las imagenes en formato pgm y se almacena en el formato deseado
        img = Image.open(ori_path)
        img.save(dest_path)

    except AssertionError as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nAssertion Error in convert_pgm_imgs\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(LOGGING_DATA_PATH, f'Conversion Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(ori_path)}\n{err}\n{"=" * 100}')