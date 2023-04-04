import tensorflow
import warnings

from itertools import repeat
from multiprocessing import Queue, Process

from breast_cancer_dataset.database_generator import BreastCancerDataset
from algorithms.classification import VGG16Model, InceptionV3Model, DenseNetModel, ResNet50Model
from algorithms.segmentation import UnetVGG16Model, UnetDenseNetModel, UnetInceptionV3Model, UnetResnet50Model
from algorithms.model_ensambling import RandomForest
from algorithms.utils import training_pipe
from data_viz.visualizacion_resultados import DataVisualizer

from utils.config import MODEL_FILES, ENSEMBLER_CONFIG
from utils.functions import bulk_data, get_path

warnings.filterwarnings('ignore')


if __name__ == '__main__':

    # Se chequea la existencia de GPU's activas
    #检查活动的GPU
    print("TF version   : ", tensorflow.__version__)
    # we'll need GPU!
    print("GPU available: ", tensorflow.config.list_physical_devices('GPU'))

    # Parámetros de entrada que serán sustituidos por las variables del usuario
    #将由用户变量替代的输入参数

    # Los valores posibles son segmentation, classification
    # 可能的值是分割、分类
    task_type = 'classification'
    # Los valores disponibles son PATCHES, COMPLETE_IMAGE
    #可用的值是PATCHES, COMPLETE_IMAGE
    experiment = 'PATCHES'
    # El tipo de arquitectura escogida: valores 'simple', 'complex'
    #选择的架构类型：值 "简单"，"复杂"，
    fc = 'complex'

    # Nombre del experimento
    # 实验名称
    experiment_name = 'EJEC_ROI_TEST_FC_COMPLEX'

    available_models = {
        'classification': [InceptionV3Model, DenseNetModel, ResNet50Model, VGG16Model],
        'segmentation': [UnetVGG16Model, UnetDenseNetModel, UnetInceptionV3Model, UnetResnet50Model]
    }

    # Se setean las carpetas para almacenar las variables del modelo en función del experimento.
    # 根据实验，设置了文件夹来存储模型的变量。
    model_config = MODEL_FILES
    model_config.set_model_name(name=experiment_name)

    # Se inicializa el procesado de las imagenes para los distintos datasets.
    # 不同数据集的图像处理被初始化。
    db = BreastCancerDataset(
        xlsx_io=model_config.model_db_desc_csv, img_type=experiment, task_type=task_type, test_dataset=['MIAS']
    )

    # Se generarán algunos ejemplos de la base de datos
    # 将会产生一些数据库的例子。
    data_viz = DataVisualizer(config=model_config, img_type=experiment)

    print(f'{"-" * 75}\nGenerando ejemplos de Data Augmentation del set de datos.\n{"-" * 75}')
    print(f'{"-" * 75}\n从数据集中生成数据增强的例子。\n{"-" * 75}')
    data_viz.get_data_augmentation_examples()

    print(f'{"-" * 75}\nGenerando análisis EDA del set de datos.\n{"-" * 75}')
    print(f'{"-" * 75}\n生成数据集的EDA分析。\n{"-" * 75}')
    data_viz.get_eda_from_df()

    print(f'{"-" * 75}\nGenerando imagenes de ejemplo de preprocesado del set de datos.\n{"-" * 75}')
    print(f'{"-" * 75}\n生成样本数据集预处理图像\n{"-" * 75}')
    # data_viz.get_preprocessing_examples()

    # Debido a que tensorflow no libera el espacio de GPU hasta finalizar un proceso, cada modelo se entrenará en
    # un subproceso daemonico para evitar la sobrecarga de memoria.
    #由于tensorflow在进程结束前不会释放GPU空间，因此每个模型将在一个守护线程中进行训练，以避免内存开销。
    for weight_init, froz_layer in zip([*repeat('imagenet', 6), 'random'], ['ALL', '0FT', '1FT', '2FT', '3FT', '4FT',
                                                                            'ALL']):
        for cnn in available_models[task_type]:
            q = Queue()

            # Se rea el proceso
            # 该过程是在以下情况下进行的
            p = Process(target=training_pipe, args=(cnn, db, q, model_config, task_type, fc, weight_init, froz_layer))

            # Se lanza el proceso
            # 该进程启动了
            p.start()

            # Se recuperan los resultados. El metodo get es bloqueante hasta que se obtiene un resultado.
            # 结果被检索出来。get方法是阻塞的，直到获得一个结果。
            predictions = q.get()

            if task_type == 'classification':

                # Se almacenan los resultados de cada modelo.
                # 每个模型的结果都被储存起来。
                path = get_path(model_config.model_predictions_cnn_dir, weight_init, froz_layer,
                                f'{cnn.__name__.replace("Model", "")}.csv')
                bulk_data(path, **predictions.to_dict())

    if task_type == 'classification':
        # Se crea el random forest
        print(f'{"-" * 75}\nGeneradando combinación secuencial de clasificadores.\n{"-" * 75}')
        print(f'{"-" * 75}\n生成分类器的顺序组合.\n{"-" * 75}')
        
        ensambler = RandomForest(db=db.df)
        ensambler.train_model(
            cnn_predictions_dir=get_path(model_config.model_predictions_cnn_dir),
            save_model_dir=get_path(model_config.model_store_ensembler_dir, ENSEMBLER_CONFIG),
            out_predictions_dir=get_path(model_config.model_predictions_ensembler_dir, ENSEMBLER_CONFIG)
        )

        print(f'{"-" * 50}\nProceso de entrenamiento finalizado\n{"-" * 50}')
        print(f'{"-" * 50}\n训练过程完成\n{"-" * 50}')

    # Visualización de los resultados
    # 结果的可视化
    print(f'{"="* 75}\nGeneradando visualización de resultados.\n{"="* 75}')
    print(f'{"="* 75}\n生成结果的可视化.\n{"="* 75}')

    print(f'{"-" * 75}\nRepresentando métricas del entrenamiento.\n{"-" * 75}')
    print(f'{"-" * 75}\n显示训练指标.\n{"-" * 75}')
    data_viz.get_model_logs_metrics(logs_dir=model_config.model_log_dir)

    print(f'{"-" * 75}\nRepresentando tiempos de entrenamiento.\n{"-" * 75}')
    print(f'{"-" * 75}\n显示训练时间.\n{"-" * 75}')
    data_viz.get_model_time_executions(summary_dir=model_config.model_summary_dir)

    print(f'{"-" * 75}\nGenerando matrices de confusión de los modelos.\n{"-" * 75}')
    print(f'{"-" * 75}\n从模型中生成混淆矩阵.\n{"-" * 75}')
    data_viz.get_model_predictions_metrics(
        cnn_predictions_dir=model_config.model_predictions_cnn_dir,
        ensembler_predictions_dir=model_config.model_predictions_ensembler_dir
    )
