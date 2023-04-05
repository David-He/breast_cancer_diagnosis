import numpy as np
import cv2
import os
import sys

from itertools import repeat
from typing import io
from PIL import Image

from preprocessing.functions import (
    apply_clahe_transform, remove_artifacts, remove_noise, crop_borders, pad_image_into_square, resize_img,
    normalize_breast, flip_breast, correct_axis
)
from utils.config import LOGGING_DATA_PATH, PREPROCESSING_FUNCS, PREPROCESSING_CONFIG
from utils.functions import (
    get_filename, save_img, get_path, get_value_from_args_if_exists, get_dirname, get_contours,search_files2
)


def full_image_pipeline(args) -> None:
    """
    Función utilizada para realizar el preprocesado de las mamografías. Este preprocesado consiste en:
        执行乳房X光片预处理的函数。该预处理包括
        1 - Recortar los bordes de las imagenes.剪裁图像的边缘
        2 - Eliminar el ruido 消除噪音
        3 - Quitar anotaciones realizadas sobre las imagenes.删除对图像的注释
        4 - Realizar una normalización min-max para estandarizar las imagenes a 8 bits.执行Min-Max标准化以标准化8位图像
        5 - Aplicar una ecualización de las imagenes para mejorar el contrastre 应用图像均衡以提高对比度
        6 - Relizar un flip horizontal para estandarizar la orientacion de los senos.重新设计水平翻转以标准化乳房的方向
        7 - Realizar el padding de las imagenes al aspect ratio deseado 按所需的纵横比填充图像
        8 - Resize de las imagenes重新调整图像
    En caso de existir una máscara, se aplicarán las siguientes funciones de la zona de interes (funcionalidad para
    segmentación de datos). 如果存在遮罩，将应用感兴趣区域的以下功能（用于数据分割）
        1 - Recortar los bordes de las imagenes.剪裁图像的边缘。
        2 - Relizar un flip horizontal para estandarizar la orientacion de los senos.重新设计水平翻转，以规范乳房的方向
        3 - Realizar el padding de las imagenes al aspect ratio deseado按所需的纵横比填充图像
        4 - Resize de las imagenes 重新调整图像

    :param args: listado de argumentos cuya posición debe ser:
        1 - path de la imagen sin procesar.未处理图像的路径
        2 - path de destino de la imagen procesada.处理图像的目标路径
        3 - Booleano para almacenar los pasos intermedios para representarlos graficamente布尔存储中间步骤以图形方式表示它们
        4 - path de destino de la máscara procesada con la zona de interés用感兴趣的区域处理掩码的目标路径
        5 - path de origen de la máscara con la zona de interés de cada imagen带有每个图像感兴趣区域的遮罩源路径
    """

    error_path: io = get_value_from_args_if_exists(args, 5, LOGGING_DATA_PATH, IndexError, KeyError)

    try:

        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        if not (len(args) >= 2):
            raise IndexError('Not enough arguments for convert_dcm_img function. Minimum required arguments: 2')

        img_filepath: io = args[0]
        dest_dirpath: io = args[1]

        save_intermediate_steps = get_value_from_args_if_exists(args, 2, False, IndexError, TypeError)
        img_mask_out_path = get_value_from_args_if_exists(args, 3, None, IndexError, TypeError)
        img_mask_filepath = get_value_from_args_if_exists(args, 4, None, IndexError, TypeError)

        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        if not os.path.isfile(img_filepath):
            raise FileNotFoundError(f'The image {img_filepath} does not exists.')
        if os.path.splitext(dest_dirpath)[1] not in ['.png', '.jpg']:
            raise ValueError(f'Conversion only available for: png, jpg')

        assert not os.path.isfile(dest_dirpath), f'Processing file exists: {dest_dirpath}'

        # Se almacena la configuración del preprocesado
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # Se lee la imagen original sin procesar.
        img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2GRAY)

        # Se lee la mascara
        if img_mask_filepath is None:
            img_mask = np.ones(shape=img.shape, dtype=np.uint8)
        else:
            if not os.path.isfile(img_mask_filepath):
                raise FileNotFoundError(f'The mask {img_mask_filepath} does not exists.')
            img_mask = cv2.cvtColor(cv2.imread(img_mask_filepath), cv2.COLOR_BGR2GRAY)

        images = {'ORIGINAL': img}

        # Primero se realiza un crop de las imagenes en el caso de que sean imagenes completas
        images['CROPPING 1'] = crop_borders(images[list(images.keys())[-1]].copy(), **prep_dict.get('CROPPING_1', {}))

        # Se aplica el mismo procesado a la mascara
        img_mask = crop_borders(img_mask, **prep_dict.get('CROPPING_1', {}))

        # A posteriori se quita el ruido de las imagenes utilizando un filtro medio
        images['REMOVE NOISE'] = remove_noise(
            img=images[list(images.keys())[-1]].copy(), **prep_dict.get('REMOVE_NOISE', {}))

        # El siguiente paso consiste en eliminar los artefactos de la imagen.
        images['REMOVE ARTIFACTS'], _, mask, img_mask = remove_artifacts(
            img=images[list(images.keys())[-1]].copy(), mask=img_mask, **prep_dict.get('REMOVE_ARTIFACTS', {})
        )

        # Una vez eliminados los artefactos, se realiza una normalización de la zona del pecho
        images['IMAGE NORMALIZED'] = \
            normalize_breast(images[list(images.keys())[-1]].copy(), mask, **prep_dict.get('NORMALIZE_BREAST', {}))

        # A continuación se realiza aplica un conjunto de ecualizaciones la imagen. El número máximo de ecualizaciones
        # a aplicar son 3 y serán representadas enc ada canal
        ecual_imgs = []
        img_to_ecualize = images[list(images.keys())[-1]].copy()
        assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, 'Número de ecualizaciones incorrecto'
        for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):

            if 'CLAHE' in ecual_func.upper():
                images[ecual_func.upper()] = apply_clahe_transform(img=img_to_ecualize, mask=mask, **ecual_kwargs)
                ecual_imgs.append(images[list(images.keys())[-1]].copy())

            elif 'GCN' in ecual_func.upper():
                pass

        if len(prep_dict['ECUALIZATION'].keys()) == 2:
            images['IMAGES SYNTHESIZED'] = cv2.merge((img_to_ecualize, *ecual_imgs))
        elif len(prep_dict['ECUALIZATION'].keys()) == 3:
            images['IMAGES SYNTHESIZED'] = cv2.merge(tuple(ecual_imgs))

        # Se realiza el flip de la imagen en caso de ser necesario:
        images['IMG_FLIP'], flip = flip_breast(images[list(images.keys())[-1]].copy(), **prep_dict.get('FLIP_IMG', {}))

        if flip:
            img_mask = cv2.flip(src=img_mask, flipCode=1)

        # Se aplica el ultimo crop de la parte izquierda
        images['CROPPING LEFT'] = crop_borders(images[list(images.keys())[-1]].copy(),
                                               **prep_dict.get('CROPPING_2', {}))
        img_mask = crop_borders(img=img_mask,  **prep_dict.get('CROPPING_2', {}))

        # Se aplica el padding de las imagenes para convertirlas en imagenes con el aspect ratio deseado
        if prep_dict.get('RATIO_PAD', False):
            images['IMAGE RATIO PADDED'] = \
                pad_image_into_square(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RATIO_PAD', {}))
            img_mask = pad_image_into_square(img=img_mask, **prep_dict.get('RATIO_PAD', {}))

        # Se aplica el resize de la imagen:
        if prep_dict.get('RESIZING', False):
            images['IMAGE RESIZED'] = \
                resize_img(img=images[list(images.keys())[-1]].copy(), **prep_dict.get('RESIZING', {}))
            img_mask = resize_img(img=img_mask, **prep_dict.get('RESIZING', {}), interpolation=cv2.INTER_NEAREST)

        if save_intermediate_steps:
            for i, (name, imag) in enumerate(images.items()):
                save_img(imag, get_dirname(dest_dirpath), f'{i}. {name}')

        if img_mask_out_path and len(get_contours(img_mask)) > 0:
            Image.fromarray(np.uint8(img_mask)).save(img_mask_out_path)

        # Se almacena la imagen definitiva
        Image.fromarray(np.uint8(images[list(images.keys())[-1]].copy())).save(dest_dirpath)

    except AssertionError as err:
        if not getattr(sys, 'frozen', False):
            with open(get_path(error_path, f'Preprocessing Errors (Assertions).txt'), 'a') as f:
                f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except IndexError as err:
        with open(get_path(error_path, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\nError calling function convert_dcm_img pipeline\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(error_path, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')


def crop_image_pipeline(args) -> None:
    """
    Función utilizada para realizar el preprocesado de las zonas de interés de las mamografias. Este preprocesado
    consiste en:
        1 - Recortar los bordes de las imagenes completas para poder suprimir bordes de las zonas de interes
        2 - Eliminar el ruido
        3 - Quitar anotaciones realizadas sobre las imagenes completas para obtener la zona del seon
        4 - Realizar los recortes de cada zona de interés a partir de la máscara del seno completo
        5 - Aplicar una ecualización CLAHE para mejorar el contraste
    :param args: listado de argumentos cuya posición debe ser:
        1 - path de la imagen sin procesar.
        2 - path de destino de la imagen procesada.
        3 - path de origen de la máscara con la zona de itnerés
        --parámetros opcionales--
        4 - número de imagenes de background (zonas sin lesión)
        5 - número de imagenes del roi
        6 - overlap para realziar distintas imagenes de un roi
        7 - margen adicional para recortar cada roi
        8 - booleano para recuperar los pasos intermedios
    """

    error_path: io = get_value_from_args_if_exists(args, 8, LOGGING_DATA_PATH, IndexError, KeyError)

    try:
        # Se recuperan los valores de arg. Deben de existir los 3 argumentos obligatorios.
        if not (len(args) >= 3):
            raise ValueError('Not enough arguments for convert_dcm_img function. Minimum required arguments: 5')

        img_filepath: io = args[0]
        out_filepath: io = args[1]
        extension: str = os.path.splitext(out_filepath)[1]
        roi_mask_path: io = args[2]

        # n_background_imgs: int = get_value_from_args_if_exists(args, 3, 0, IndexError, TypeError)
        # n_roi_imgs: int = get_value_from_args_if_exists(args, 4, 1, IndexError, TypeError)
        # overlap_roi: float = get_value_from_args_if_exists(args, 5, 1.0, IndexError, TypeError)
        margin_roi: float = get_value_from_args_if_exists(args, 6, 1.0, IndexError, TypeError)
        # save_intermediate_steps: bool = get_value_from_args_if_exists(args, 7, False, IndexError, TypeError)

        #如果输出文件已经存在，则退出
        
        files = list(search_files2(get_path(get_dirname(out_filepath), f'*{get_filename(out_filepath)}*{extension}',create=False)))
        if len(files) > 0:
            return
        
        # Se valida que el formato de conversión sea el correcto y se valida que existe la imagen a transformar
        #验证转换格式是否正确，并验证是否存在要转换的图像
        
        if not os.path.isfile(img_filepath):
            raise FileNotFoundError(f'The image {img_filepath} does not exists.')
        if not os.path.isfile(roi_mask_path):
            raise FileNotFoundError(f'The image {roi_mask_path} does not exists.')
        if extension not in ['.png', '.jpg']:
            raise ValueError(f'Conversion only available for: png, jpg')

        # Se almacena la configuración del preprocesado
        prep_dict = PREPROCESSING_FUNCS[PREPROCESSING_CONFIG]

        # Se lee la imagen original sin procesar.读取未处理的原始图像。
        img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2GRAY)

        # Se lee la mascara读取Mask
        mask = cv2.cvtColor(cv2.imread(roi_mask_path), cv2.COLOR_BGR2GRAY)

        # Primero se realiza un crop de las imagenes en el caso de que sean imagenes completas首先对图像进行裁剪，以防它们是完整的图像
        crop_img = crop_borders(img, **prep_dict.get('CROPPING_1', {}))

        # Se aplica el mismo procesado a la mascara将相同的处理应用于Mask
        img_mask = crop_borders(mask, **prep_dict.get('CROPPING_1', {}))

        # A posteriori se quita el ruido de las imagenes utilizando un filtro medio使用平均滤波器从图像中去除后验噪声
        img_denoised = remove_noise(crop_img, **prep_dict.get('REMOVE_NOISE', {}))

        # Se obtienen las zonas de patologia de la mascara juntamente con las
        # El siguiente paso consiste en eliminar los artefactos de la imagen. Solo aplica a imagenes completas下一步是消除图像中的伪影，同时获得口罩的病理区域。仅适用于完整图像
        _, _, breast_mask, mask = remove_artifacts(img_denoised, img_mask, False,
                                                   **prep_dict.get('REMOVE_ARTIFACTS', {}))

        # Se obtienen los parches de las imagenes con patologías.
        roi_zones = []
        mask_zones = []
        breast_zone = breast_mask.copy()
        for contour in get_contours(img=mask):
            x, y, w, h = cv2.boundingRect(contour)

            if (h > 15) & (w > 15):
                center = (y + h // 2, x + w // 2)
                y_min, x_min = int(center[0] - h * margin_roi // 2), int(center[1] - w * margin_roi // 2)
                y_max, x_max = int(center[0] + h * margin_roi // 2), int(center[1] + w * margin_roi // 2)
                x_max, x_min, y_max, y_min = correct_axis(img_denoised.shape, x_max, x_min, y_max, y_min)
                roi_zones.append(img_denoised[y_min:y_max, x_min:x_max])
                mask_zones.append(breast_zone[y_min:y_max, x_min:x_max])

                # Se suprimen las zonas de la patología para posteriormente obtener la zona del background删除病理区域以随后获得背景区域
                cv2.rectangle(breast_mask, (x_min, y_min), (x_max, y_max), color=(0, 0, 0), thickness=-1)

        # Se procesan las zonas de interes recortadas处理切割的感兴趣区域
        for idx, (roi, roi_mask, tipo) in enumerate(zip(roi_zones, mask_zones, repeat('roi', len(roi_zones)))):

            roi_norm = normalize_breast(roi, roi_mask, **prep_dict.get('NORMALIZE_BREAST', {}))

            # A continuación se realiza aplica un conjunto de ecualizaciones la imagen. El número máximo de
            # ecualizaciones a aplicar son 3 y serán representadas enc ada canal然后对图像进行一组均衡。要应用的最大均衡数为3，将表示为Encada信道
            ecual_imgs = []
            img_to_ecualize = roi_norm.copy()
            assert 0 < len(prep_dict['ECUALIZATION'].keys()) < 4, 'Número de ecualizaciones incorrecto'
            for i, (ecual_func, ecual_kwargs) in enumerate(prep_dict['ECUALIZATION'].items(), 1):

                if 'CLAHE' in ecual_func.upper():
                    ecual_imgs.append(apply_clahe_transform(img_to_ecualize, roi_mask, **ecual_kwargs))

                elif 'GCN' in ecual_func.upper():
                    pass

            if len(prep_dict['ECUALIZATION'].keys()) == 2:
                roi_synthetized = cv2.merge((img_to_ecualize, *ecual_imgs))
            elif len(prep_dict['ECUALIZATION'].keys()) == 3:
                roi_synthetized = cv2.merge(tuple(ecual_imgs))
            else:
                roi_synthetized = ecual_imgs[-1]

            # Se almacena la imagen definitiva
            path = get_path(get_dirname(out_filepath), f'{tipo}_{get_filename(out_filepath)}_{idx}{extension}')
            Image.fromarray(np.uint8(roi_synthetized)).save(path)

    except AssertionError as err:
        if not getattr(sys, 'frozen', False):
            with open(get_path(error_path, f'Preprocessing Errors (Assertions).txt'), 'a') as f:
                f.write(f'{"=" * 100}\nAssertion Error in image processing\n{err}\n{"=" * 100}')

    except Exception as err:
        with open(get_path(error_path, f'Preprocessing Errors.txt'), 'a') as f:
            f.write(f'{"=" * 100}\n{get_filename(img_filepath)}\n{err}\n{"=" * 100}')
