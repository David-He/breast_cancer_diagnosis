# Algoritmo de clasificación de cancer de lesiones en exámenes mamográficos.
# 乳房造影检查中病变的癌症分类算法

Este proyecto persigue el objetivo de crear una herramienta que permita clasificar las lesiones 
presentes en imágenes mamográficas como benignas o malignas.
该项目旨在创建一个工具，将乳腺图像上的病变分类为良性或恶性。

Para la realización de esta aplicación se han utilizado 4 arquitecturas que componen el estado 
del arte en tareas de clasificación de imagen como son: _VGG16_, _ResNet50_, _DenseNet121_ y _InceptionV3_.
Las predicciones realizadas por cada arquitectura son combinadas en un _Random Forest_ para obtener la
predicción final de cada instancia (probabilidad de que un cáncer sea maligno)
为了实现这一应用，我们使用了4个架构，它们构成了图像分类领域的最新技术。
如：_VGG16_、_ResNet50_、_DenseNet121_和_InceptionV3_.

Las bases de datos utilizadas para entrenar cada modelo son:
用于训练每个模型的数据库包括：
- CBIS-DDDSM: Disponible en https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
- INBreast: Disponible en https://www.kaggle.com/martholi/inbreast?select=inbreast.tgz
- MIAS: Disponible en https://www.kaggle.com/kmader/mias-mammography/version/3?select=Info.txt

**Este repositorio pretende servir de base para que otras personas puedan aprovechar el trabajo realizado
y unirse en la causa para la lucha contra el cáncer de seno.**
****该知识库旨在为其他人提供基础，使他们能够利用所做的工作，团结起来抗击乳腺癌。**
 De este modo, a continuación se detallará la estructura
del repositorio así como los objetivos de cada módulo o paquete.
程序结构以及每个模块或包的目标将在下面详细说明。

- `bin`: Contiene los archivos necesarios para crear una interfaz gráfica a partir de la librería `Pyqt5`. Entre estos
destaca la carpeta `hooks` con las dependencias necesarias a instalar en la aplicación.
   包含从库创建图形界面所需的文件。其中最主要的是包含在.pyqt5hooks应用程序中安装所需依赖项的文件夹

- `notebooks`: Contiene algunos análisis _adhoc_ para la realización de la herramienta (procesado de imágenes o creación
de la combinación secuencial de clasificadores).
   它包含一些用于实现该工具的附加分析（图像处理或创建分类器的顺序组合）。

- `src`: Contiene los paquetes y los _scripts_ principales de ejecución de código. Este paquete se divide en:
    包含包和主要代码执行脚本。包括：
    - `algoriths`: Módulo utilizado para crear las redes neuronales de clasificación y de segmentación (_on going_).
    **En este módulo se deberían de añadir todas aquellas arquitecturas de red nuevas a incluir**. Por otra parte,
    también existen scripts para la creación secuencial de clasificadores a partir de un _Random Forest_. La generación
    de nuevos algorítmos podría introducirse en este script. 
	用于创建分类和分割神经网络的模块（正在进行）。所有要包括的新的网络架构都应该被添加到这个模块中。
	另一方面，也有一些脚本用于从随机森林中依次创建分类器。新算法的生成可以在这个脚本中引入。
    
    - `breast_cancer_dataset`: Módulo que contiene los scripts utilizados para realizar el procesado de datos de 
    cada set de datos individual (CBIS, MIAS e INBreast). Estos scripts se encuentran en el paquete `databases` del 
    módulo, de modo que **para cualquier base de datos que se desee añadir, será necesario introducir su procesado en este
    paquete**. Por otra parte, el script _database_generator.py_ crea el set de datos utilizado por los algorítmos de 
    _deep learning_ utilizados uniendo cada base de datos individual contenida en el paquete `databases`. 
    Asimismo, se aplican técnicas de _data augmentation_ y se realiza el split de datos en entrenamiento y validación.
	包含用于执行每个单独数据集（CBIS、MIAS和INBREAST）的数据处理的脚本的模块。这些脚本位于模块包中，因此对于要添加的任何数据库，
	都需要在该包中输入其处理过程。
	另一方面，database_generator.py脚本通过绑定包中包含的每个单独数据库来创建所使用的深度学习算法所使用的数据集。
	此外，还应用了数据增强技术，并在训练和验证中进行了数据分割。DatabaseSDatabase
    
     - `data_viz`: módulo utilizado para generar visualizaciones de los resultados obtenidos por las redes.
	 用于生成网络获得的结果的可视化的模块。
     
     - `preprocessing`: módulo que contiene las funciones de preprocesado genéricas aplicadas a todos los conjuntos de 
     datos. Además, contiene las funcionalidades necesarias para estandarizar las imágenes a formato _png_ o _jpg_.
     **Cualquier procesado nuevo a añadir, deberá hacerse en este módulo**.
	 包含应用于所有数据集的通用预处理函数的模块。此外，它还包含将图像标准化为PNG或JPG格式所需的功能。任何要添加的新处理都必须在此模块中进行。
     
     - `static`: módulo que contiene los archivos estáticos utilizados para la creación de la interfaz gráfica del 
     programa como ficheros _.css_, _.html_ e imágenes.
	 包含用于创建程序图形界面的静态文件的模块，如.css、.html和图像。
     
     - `user_interace`:  módulo utilizado para crear la aplicación `Pyqt5` de clasificación de imágenes de seno.
	 用于创建sine.pyqt5图像分类应用程序的模块
     
     - `utils`: módulo genérico en el cual configurar las rutas de las bases de datos dentro del entorno local desde 
     dónde se esté ejecutando el aplicativo, así como la configuración de los hiperparámetros de las redes neuronales. 
	 通用模块，用于在运行应用程序的本地环境中配置数据库路径，以及配置神经网络的超参数。
     
     - `main_train.py`: script utilizado para realizar generar el pipeline de entrenamiento, desde la obtención de datos
     hasta la creación y el entrenamiento de cada modelo.
	 用于生成训练管道的脚本，从获取数据到创建和训练每个模型。
     
     - `main.py`: script utilizado para lanzar la aplicación final realizada.
	 用于启动执行的最终应用程序的脚本。
     
Juntamente con los módulos contenidos en esta descripción, se crearán un conjunto de carpetas adicionales. Estas carpetas
no están contenidas en el repositorio por motivos de capacidad de almacenaje. A continuación se detallan los módulos y 
sus objetivos:
与本说明中包含的模块一起，将创建一组额外的文件夹。由于存储容量的原因，这些文件夹不包含在存储库中。模块及其目标如下：

- `logging`: Carpeta que contendrá los logs de ejecuciones del programa, como por ejemplo los errores producidos durante
el procesado de las imagenes.
包含程序执行日志的文件夹，例如在处理图像期间产生的错误。

- `models`: Carpeta que contendrá los modelos almacenados juntamente con las predicciones realizadas durante el entrenamiento. 
文件夹，其中将包含与训练期间进行的预测一起存储的模型。

- `data`: Carpeta que contendrá las imagenes de cada set de datos convertidas (sub-directorio _01_CONVERTED_) y 
procesadas (sub-directorio _02_PROCESED_). Esta carpeta tiene el objetivo de reiterar el proceso de procesado de imagenes
una vez realizado.
包含每个已转换（01_converted子目录）和已处理（02_processed子目录）数据集图像的文件夹。该文件夹的目的是在执行后重复图像处理过程。