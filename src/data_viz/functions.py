from itertools import product
from tensorflow.keras.preprocessing.image import array_to_img
from pandas import DataFrame
from typing import io, List

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_countplot(data: DataFrame, file: io, x: str, hue: str = None, title: str = '', annotate: bool = True,
                     norm: bool = False):
    """
    Función utilizada para crear un countplot de seaborn
    用于创建海产计数图的函数
    :param data: pandas dataframe con los datos a crear el countplot用数据创建计数图的pandas数据框架
    :param file: filepath donde guardar la imagen保存图像的文件路径
    :param x: nombre de columna del data que será la X del countplot将成为计数图中X的数据的列名
    :param hue: nombre de columna de data que servirá para segregar clases en el countplot数据列的名称，用于在计数图中分隔类别。
    :param title: titulo de gráfico图表标题
    :param annotate: booleano para anotar los valores de cada barra del countplot 布尔值，用于注释计数图的每个条形图的值。
    :param norm: booleano para realizar una normalización ed los valores individuales de cada X en función de hue
                 布尔值，根据色调对每个X的单个值进行归一化处理。
    """

    # Figura de matplotlib para almacenar el gráfico
    plt.figure(figsize=(15, 5))

    # Gráfico de frecuencias
    ax = sns.countplot(x=x, data=data, hue=hue, palette=sns.light_palette((210, 90, 60), input='husl', reverse=True))

    # Se realizan las anotaciones de los valores de frecuencia y frecuencia normalizda para cada valor de la
    # variable objetivo.
    if annotate:
        ax_ = list(ax.patches)
        ax_.sort(key=lambda annot: annot.get_x())
        for p, (l, _) in zip(ax_, product(ax.xaxis.get_ticklabels(), [*ax.get_legend_handles_labels()[1],
                                                                      *ax.xaxis.get_ticklabels()][:data[x].nunique()])):
            txt = '{a:.0f} ({b:.2f} %)'.format(
                a=p.get_height(),
                b=(p.get_height() / (len(data[data[x] == l.get_text()]) if norm else len(data))) * 100
            )
            ax.annotate(txt, xy=(p.get_x() + p.get_width() * 0.5, p.get_height()), va='center', ha='center',
                        clip_on=True, xycoords='data', xytext=(0, 7), textcoords='offset points')

    # Título del gráfico
    if title:
        ax.set_title(title, fontweight='bold', size=14)

    # Se elimina el label del eje y.
    ax.set(ylabel='')
    ax.yaxis.grid(True)

    sns.despine(ax=ax, left=True)

    # Se almacena la figura
    plt.savefig(file)


def merge_cells(table: plt.table, cells: List[tuple]):
    """
    función para unir las celdas de una tabla de matplotlib (https://stackoverflow.com/a/53819765/12684122)

    :param table: tabla de matplotlib a unir las celdas
    :param cells: lista con las celdas a unir

    """

    # Se crea un array con las celdas a unir tanto verticales como horizontales
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i + 1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i + 1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # Si se realiza un merge horizontal todos los valores de h serán 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda vert: vert[1]))
        edges = ['BTL'] + ['BT'] * (len(cells) - 2) + ['BTR']
    # Si se realiza un merge en vertical todos los valores de v seran 0
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda hor: hor[0]))
        edges = ['TRL'] + ['RL'] * (len(cells) - 2) + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    # Se iteran las celdas eliminando la visibilidad de los eejes
    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e

    for cell in cells[1:]:
        table[cell[0], cell[1]].get_text().set_visible(False)


def render_mpl_table(data, font_size=14, merge_pos: List[List[tuple]] = None, header_rows: int = 1):
    """
    Función utilizada para renderizar un dataframe en una tabla de matplotlib. Función recuperada de
    https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure
    """

    col_width = 3
    row_height = 0.625
    row_colors = ['#f1f1f2', 'w']

    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    header_columns = data.columns.nlevels

    table = ax.table(
        cellText=np.vstack([*[data.columns.get_level_values(i) for i in range(0, header_columns)], data.values]),
        bbox=[0, 0, 1, 1], cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for k, cell in table._cells.items():
        if k[0] < header_columns or k[1] < header_rows:
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if merge_pos is not None:
        for cells in merge_pos:
            merge_cells(table, cells)

    return ax.get_figure(), ax


def plot_image(img: np.ndarray, title: str, ax_: plt.axes):
    """
    Función que permite representar una imagen en un axes de matplotlib suprimiendole el grid y los ejes.

    :param img: imagen en formato array y de dimensiones (n, width, height, channels)
    :param title: título del axes
    :param ax_: axes subplot

    """

    # Se representa la imagen
    ax_.imshow(array_to_img(img[0]))

    # Se eliminan ejes y grid
    ax_.axes.grid(False)
    ax_.axes.set_xticklabels([])
    ax_.axes.set_yticklabels([])

    # Título del gráfico en el eje de las x.
    ax_.axes.set(xlabel=title)
