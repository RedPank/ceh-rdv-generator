import os
import io
from jinja2 import Environment
from pandas import DataFrame

from core.exceptions import IncorrectMappingException
from core.exporters import MartPackExporter
from core.mapping import MappingMeta, MartMapping
from core.stream_header_data import StreamHeaderData
import logging
from core.config import Config as Conf
import re


def mapping_generator(
        file_path: str,
        out_path: str,
        # load_mode: str,
        env: Environment,
        author: str
) -> None:
    """Функция генератора маппинга, вызывает функционал по генерации
       файлов

    Args:
        file_path (str): Полный путь к файлу маппинга РДВ
        out_path (str): Каталог, в котором будут сформированы подкаталоги с описанием потоков
        env (Environment): Окружение шаблонов jinja2
        author (str): Наименование автора потоков для заполнения в шаблоне
    """

    Conf.is_warning = False
    Conf.is_error = False

    logging.info(f"file_path: {file_path}")
    logging.info(f"out_path: {out_path}")
    logging.info(f"author: {author}")

    logging.info(f'Чтение данных из файла "{file_path}"')

    # Чтение данных их EXCEL
    try:
        with open(file_path, 'rb') as f:
            byte_data = io.BytesIO(f.read())

    except Exception:
        msg = f"Ошибка чтения данных из файла {file_path}"
        logging.exception(msg)
        raise IncorrectMappingException(msg)

    # Данные EXCEL
    mapping_meta = MappingMeta(byte_data)

    # Список целевых таблиц
    map_objects: list[str] = mapping_meta.get_tgt_tables_list()

    # Список шаблонов имен потоков и/или имен потоков, которые будут обработаны
    wf_templates_list = Conf.config.get('wf_templates_list', list('.+'))

    # Цикл по списку целевых таблиц
    for tbl_index, tgt_table in enumerate(map_objects):
        if tbl_index > 0:
            logging.info('')

        logging.info('>>>>> Begin >>>>>')
        logging.info(f"{tbl_index} {tgt_table}")

        # Проверяем соответствие названия целевой таблицы шаблону
        pattern: str = Conf.get_regexp('tgt_table_name_regexp')
        if not re.match(pattern, tgt_table):
            logging.error(f'Имя целевой таблицы "{tgt_table}" на листе "Перечень загрузок Src-RDV" '
                          f'не соответствует шаблону "{pattern}"')
            Conf.is_error = True
            continue

        # Данные "Перечень загрузок Src-RDV" листа для таблицы
        stream_header_data: StreamHeaderData = StreamHeaderData(df=mapping_meta.mapping_list, tgt_table=tgt_table)

        if not stream_header_data.base_flow_name:
            logging.error(f'Для таблицы {tgt_table} неверно задано/не задано поле "Название потока"/Flow_name ')
            raise IncorrectMappingException("Имя потока не определено")
        logging.info(f'flow_name = {stream_header_data.flow_name}')

        # Фильтр по шаблону имени потока из файла конфигурации
        if not [True for pattern in wf_templates_list if re.match(pattern, stream_header_data.flow_name)]:
            logging.info(f'Поток "{stream_header_data.flow_name}" обрабатываться не будет, т.к. не соответствует ни '
                         f'одному из шаблонов в файле конфигурации')
            continue

        if stream_header_data.version_end:
            logging.info(f'Версия "{stream_header_data.version_end}" потока "{stream_header_data.flow_name}" '
                         f'обрабатываться не будет')
            continue

        # Проверяем таблицу-источник
        # src_table: str | None = stream_header_data.src_table
        pattern: str = Conf.get_regexp('src_table_name_regexp')
        if not re.match(pattern, stream_header_data.src_table):
            logging.error(f'Имя таблицы-источника "{stream_header_data.src_table}" на листе "Перечень загрузок Src-RDV" '
                          f'не соответствует шаблону "{pattern}"')
            # raise IncorrectMappingException("Имя таблицы-источника не определено")
            Conf.is_error = True
            continue

        logging.info(f'src_table = {stream_header_data.src_table}')

        # Данные для заданной целевой таблицы
        mapping: DataFrame = mapping_meta.get_mapping_by_table(tgt_table)

        # Возвращает наименование (логическое) "источника" для заданной целевой таблицы - поле src_sd
        src_cd: str | None = mapping_meta.get_src_cd_by_table(tgt_table)
        if not src_cd:
            logging.error(f'Для целевой таблицы "{tgt_table}" неверно задано/не задано имя источника')
            logging.error('Имя источника задается в колонке "Expression" для поля "src_cd"')
            # raise IncorrectMappingException("Имя источника не определено")
            Conf.is_error = True
            continue

        logging.info(f'src_cd = {src_cd}')

        # Имя источника - Source_name
        if not stream_header_data.source_system:
            logging.error(f'Для таблицы {tgt_table} неверно задано/не задано поле "Источник данных'
                          f' (транспорт)"/Source_name')
            Conf.is_error = True
            continue
        logging.info(f'source_name = {stream_header_data.source_system}')

        # Алгоритм - Algorithm_UID
        algorithm_uid: str = stream_header_data.algorithm_uid
        if not algorithm_uid:
            logging.error(f'Для таблицы {tgt_table} неверно задано/не задано поле "UID алгоритма"/"Algorithm_UID"')
            # raise IncorrectMappingException("Поле 'Algorithm_UID' не определено")
            Conf.is_error = True
            continue
        logging.info(f'algorithm_uid = {algorithm_uid}')

        # Название схемы таблицы (берется из названия таблицы src_table)
        source_system_schema: str | None = stream_header_data.src_table.split('.')[0]
        if not source_system_schema:
            logging.error(f'На листе "Перечень загрузок Src-RDV" '
                          f'неверно задано/не задано имя схемы источника в имени таблицы "{stream_header_data.src_table}"')
            # raise IncorrectMappingException("Имя схемы источника не определено")
            Conf.is_error = True
            continue

        # Подготовка данных для файлов для одной таблицы
        exp_obj = MartMapping(
            mart_name=tgt_table,
            mart_mapping=mapping,
            src_cd=src_cd,
            data_capture_mode='increment',
            stream_header_data=stream_header_data,
            source_system_schema=source_system_schema
        )

        # Каталог для файлов
        out_path_tbl = os.path.join(out_path, stream_header_data.flow_name)
        logging.info(f'Каталог потока {stream_header_data.base_flow_name}: {out_path_tbl}')

        # Объект для формирования данных для вывода в файлы
        mp_exporter = MartPackExporter(
            exp_obj=exp_obj,
            path=out_path_tbl,
            env=env,
            author=author)

        # Вывод данных в файлы
        mp_exporter.load()
        logging.info(f'Файлы потока {stream_header_data.base_flow_name} сформированы')

    if Conf.is_error:
        logging.error(f'Один или более потоков не были сформированы из-за обнаруженных ошибок')

    logging.info('')
