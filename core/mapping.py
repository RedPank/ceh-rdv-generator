import random
import string

import pandas
import pandas as pd
import re
from dataclasses import dataclass

from pandas import DataFrame
import logging

from core.config import Config
from .context import (SourceContext, TargetContext, MappingContext, UniContext,
                      HubFieldContext)
from .exceptions import IncorrectMappingException


class StreamData:
    """
    Класс представляет данные с листа 'Перечень загрузок Src-RDV' для указанной целевой таблицы.
    Названия полей соответствуют названиям колонок EXCEL
    """
    row: pd.DataFrame
    row_dict: dict

    # mapping_ver: str
    # mapping_ver_to: str
    algorithm_uid: str
    subalgorithm_uid: str
    flow_name: str
    tgt_table: str
    target_rdv_object_type: str
    src_table: str
    source_name: str
    scd_type: str
    # algo_name: str
    # data_filtering: str
    # distribution_field: str
    # comment: str

    def __init__(self, df: pd.DataFrame, tgt_table: str):

        self.row = df.query(f'tgt_table == "{tgt_table}"')
        if len(self.row) == 0:
            logging.error(f"Не найдено имя целевой таблицы '{tgt_table}' "
                          f"на листе 'Перечень загрузок Src-RDV'")
            raise IncorrectMappingException("Не найдено имя целевой таблицы")

        if len(self.row) > 1:
            logging.error("Найдено несколько строк для целевой таблицы '{table_name}' на листе "
                          "'Перечень загрузок Src-RDV'")
            raise IncorrectMappingException("Найдено несколько строк для целевой таблицы")

        self.row_dict = self.row.to_dict('records')[0]
        # self.mapping_ver = self.row_dict["mapping_ver"]
        # self.mapping_ver_to = self.row_dict["mapping_ver_to"]
        self.algorithm_uid = re.sub(r"\s", '', self.row_dict["algorithm_uid"])
        self.subalgorithm_uid = self.row_dict["subalgorithm_uid"]
        self.flow_name = re.sub(r"\s", '', self.row_dict["flow_name"])
        self.tgt_table = re.sub(r"\s", '', self.row_dict["tgt_table"])
        self.target_rdv_object_type = re.sub(r"\s", '', self.row_dict["target_rdv_object_type"])
        self.src_table = re.sub(r"\s", '', self.row_dict["src_table"])
        self.source_name = re.sub(r"\s", '', self.row_dict["source_name"])
        self.scd_type = re.sub(r"\s", '', self.row_dict["scd_type"])
        # self.algo_name = self.row_dict["algo_name"]
        # self.data_filtering = self.row_dict["data_filtering"]
        # self.distribution_field = self.row_dict["distribution_field"]
        # self.comment = self.row_dict["comment"]


def _generate_mapping_df(file_data: bytes, sheet_name: str):
    """
    Трансформирует полученные данные EXCEL в тип DataFrame.
    Обрабатываются только данные листа из sheet_name.
    Проверяет в данных наличие колонок из списка.

    Параметры:
        file_data: bytes
            Данные в виде "строки", считанные их EXCEL-файла
        sheet_name: str
            Название листа в книге EXCEL

    Возвращаемое значение:
        Объект с типом DataFrame.
    """

    # Список имен колонок в программе
    columns = Config.excel_data_definition.get('columns', dict())
    columns_list: list[str] = [col_name.lower().strip() for col_name in columns[sheet_name]]

    # Список "псевдонимов" названий колонок
    col_aliases = Config.excel_data_definition.get('col_aliases', dict())
    # "Название колонки в программе": "Название колонки на листе"
    aliases_list: dict = {}
    for key, val in col_aliases[sheet_name].items():
        aliases_list[key.lower().strip()] = val.lower().strip()

    # Преобразование данных в DataFrame
    try:
        mapping: DataFrame = pd.read_excel(file_data, sheet_name=sheet_name, header=1)
    except Exception:
        logging.exception("Ошибка преобразования данных в DataFrame")
        raise

    # Переводим названия колонок в нижний регистр
    # rename_list - словарь {старое_название: новое_название}
    rename_list = {col: col.lower().strip() for col in mapping.columns}
    mapping = mapping.rename(columns=rename_list)

    # Проверка полученных данных
    error: bool = False
    # Находим соответствие между "Названием колонки в программе" и "Названием колонки на листе"
    for col_name in columns_list:
        if not (col_name in mapping.columns.values):

            alias: str | None = aliases_list.get(col_name, None)
            if alias and alias in mapping.columns.values:
                logging.info(f"Иия колонки '{alias}' на листе '{sheet_name}' заменено на '{col_name}'")
                mapping = mapping.rename(columns={alias: col_name})

            else:
                logging.error(f"Колонка '{col_name}' не найдена на листе '{sheet_name}'")
                logging.error("Список допустимых имен колонок:")
                logging.error(columns_list)
                logging.error("Список колонок на листе EXCEL:")
                logging.error(mapping.columns.values)
                error = True

    if error:
        raise IncorrectMappingException("Ошибка в структуре данных EXCEL")

    # Трансформация данных: оставляем в наборе только колонки из списка и не пустые строки
    mapping = mapping[columns_list].dropna(how='all')

    return mapping


def _is_duplicate(df: pd.DataFrame, field_name: str) -> bool:
    """
    Проверяет колонку DataFrame на наличие не пустых дубликатов

    Args:
        df: DataFrame в котором выполняется проверка
        field_name: Имя поля в DataFrame, для которого выполняется проверка

    Returns:
        object: True-Дубликаты найдены
    """
    return True in df[field_name.lower()].dropna(how='all').duplicated()


def _get_duplicate_list(df: pd.DataFrame, field_name: str) -> list | None:
    """
    Проверяет колонку DataFrame на наличие не пустых дубликатов

    Args:
        df: Объект DataFrame, в котором выполняется поиск дубликатов.
        field_name: Имя поля в DataFrame, для которого выполняется поиск.

    Returns:
        object: None, если нет дубликатов или список дубликатов
    """

    dupl = df[~df[field_name.lower()].dropna().duplicated()].tolist()

    if len(dupl) > 0:
        return dupl

    return None


class MappingMeta:
    # Данные листа 'Детали загрузок Src-RDV'
    mapping_df: pd.DataFrame
    # Данные листа 'Перечень загрузок Src-RDV'
    mapping_list: pd.DataFrame

    def __init__(self, byte_data):

        is_error: bool = False
        tgt_pk: set = {'pk', 'bk', 'rk'}

        # Ф-ия для проверки "состава" поля 'tgt_pk'
        def test_tgt_pk(a) -> bool:
            if str is type(a):
                if not a:
                    return True
                else:
                    return len(set(a.split(',')).difference(tgt_pk)) == 0
            else:
                return False

        # Проверка, очистка данных, преобразование в DataFrame
        # Детали загрузок Src-RDV
        self.mapping_df = _generate_mapping_df(file_data=byte_data, sheet_name='Детали загрузок Src-RDV')

        # Оставляем только строки, в которых заполнено поле 'Tgt_table'
        self.mapping_df = self.mapping_df.dropna(subset=['tgt_table'])

        # Преобразуем значения в "нужный" регистр
        self.mapping_df['src_attr'] = self.mapping_df['src_attr'].str.lower()
        self.mapping_df['src_attr'] = self.mapping_df['src_attr'].str.strip()

        self.mapping_df['src_attr_datatype'] = self.mapping_df['src_attr_datatype'].str.lower()
        self.mapping_df['src_attr_datatype'] = self.mapping_df['src_attr_datatype'].str.strip()

        self.mapping_df['tgt_attribute'] = self.mapping_df['tgt_attribute'].str.lower()
        self.mapping_df['tgt_attribute'] = self.mapping_df['tgt_attribute'].str.strip()

        self.mapping_df['tgt_attr_datatype'] = self.mapping_df['tgt_attr_datatype'].str.lower()
        self.mapping_df['tgt_attr_datatype'] = self.mapping_df['tgt_attr_datatype'].str.strip()

        self.mapping_df['tgt_pk'] = self.mapping_df['tgt_pk'].str.lower()
        self.mapping_df['tgt_pk'] = self.mapping_df['tgt_pk'].str.strip()

        # Заменяем значения NaN на пустые строки, что-бы дальше "не мучится"
        self.mapping_df['tgt_pk'] = self.mapping_df['tgt_pk'].fillna(value="")

        # Проверяем состав поля 'tgt_pk'
        err_rows: pd.DataFrame = self.mapping_df[~self.mapping_df['tgt_pk'].apply(test_tgt_pk)]
        if len(err_rows) > 0:
            logging.error(f"Неверно указаны значения в поле 'tgt_pk'")
            for line in str(err_rows).splitlines():
                logging.error(line)
            logging.error(f'Допустимые значения: {tgt_pk}')
            is_error = True

        # "Разворачиваем" колонку Tgt_PK в отдельные признаки
        self.mapping_df = self.mapping_df.assign(_pk=lambda _df: _df['tgt_pk'].str.
                                                 extract('(^|,)(?P<_pk>pk)(,|$)')['_pk'])

        # Признак формирования значения hub из поля _rk/_id
        self.mapping_df = self.mapping_df.assign(_rk=lambda _df: _df['tgt_pk'].str.
                                                 extract(r'(^|,)(?P<_rk>rk|bk)(,|$)')['_rk'])

        # Перечень загрузок Src-RDV ------------------------------------------------------------------------------------
        self.mapping_list = _generate_mapping_df(file_data=byte_data, sheet_name='Перечень загрузок Src-RDV')

        # Список целевых таблиц. Проверяем наличие дубликатов в списке
        self._tgt_tables_list: list[str] = self.mapping_list['tgt_table'].dropna().tolist()
        visited: set = set()
        for tbl in self._tgt_tables_list:
            if tbl in visited:
                logging.error(f"На листе 'Перечень загрузок Src-RDV' "
                              f"присутствуют повторяющиеся названия таблиц: {tbl}")
                is_error: bool = True
            else:
                visited.add(tbl)

        # Проверка на наличие дубликатов на листе 'Перечень загрузок Src-RDV'
        for field_name in ['Algorithm_UID', 'Flow_name', 'Tgt_table']:
            if _is_duplicate(df=self.mapping_list, field_name=field_name):
                logging.error(f"На листе 'Перечень загрузок Src-RDV' найдены дубликаты в колонке '{field_name}'")
                is_error: bool = True

        if is_error:
            raise IncorrectMappingException("Ошибка в структуре данных")

        # Заменяем значения NaN на пустые строки, что-бы дальше "не мучиться"
        self.mapping_list['scd_type'] = self.mapping_list['scd_type'].fillna(value="")

    def get_tgt_tables_list(self) -> list[str]:
        """
        Возвращает список целевых таблиц (из колонки 'tgt_table')
        """
        return self._tgt_tables_list

    def get_mapping_by_table(self, tgt_table: str) -> pd.DataFrame:
        """
        Возвращает список (DataFrame) строк для заданной целевой таблицы
        """
        df: DataFrame = self.mapping_df[self.mapping_df['tgt_table'] == tgt_table].dropna(how="all")
        return df

    def get_src_cd_by_table(self, tgt_table: str) -> str | None:
        """
        Возвращает наименование источника для заданной целевой таблицы. Если None, то источник не найден
        """
        src_cd_obj = self.mapping_df.query(f'tgt_table == "{tgt_table}" and tgt_attribute == "src_cd"')['expression']
        if len(src_cd_obj) == 0:
            logging.error(f"Не найдено поле 'src_cd' в таблице '{tgt_table}'")
            return None

        if len(src_cd_obj) > 1:
            logging.error(f"Найдено несколько описаний для поля 'src_cd' в таблице '{tgt_table}'")
            return None

        src_cd: str = src_cd_obj.to_numpy()[0]
        # Удаляем пробельные символы
        src_cd = re.sub(r"\s", '', src_cd)
        # Выделяем имя источника
        pattern: str = Config.get_regexp('src_cd_regexp')
        result = re.match(pattern, src_cd)

        if result is None:
            logging.error(f"Не найдено имя источника для таблицы '{tgt_table}' по шаблону '{pattern}'")
            logging.error(f"Найденное значение: {src_cd}")
            return None

        src_cd = result.groups()[0]
        return src_cd


@dataclass
class MartMapping:
    """
    Переменные класса напрямую используются при заполнении шаблона
    """
    mart_name: str
    mart_mapping: pd.DataFrame
    # Код источника
    src_cd: str
    # Режим захвата данных: snapshot/increment
    data_capture_mode: str
    # Имя источника (для uni-провайдера).
    source_system: str
    # Имя схемы источника (для uni-провайдера)
    source_system_schema: str
    # Имя потока без префиксов wf_/cf_
    work_flow_name: str
    # Имя схемы источника (для uni-провайдера)

    src_ctx: SourceContext | None = None
    tgt_ctx: TargetContext | None = None
    mapping_ctx: MappingContext | None = None
    uni_ctx: UniContext | None = None
    # Режим загрузки "дельты" - значение параметра в файле описания рабочего потока
    delta_mode: str = 'new'
    algorithm_UID: str | None = None

    # Инициализация данных
    def __post_init__(self):
        # Подготовка контекста источника
        self._src_ctx_post_init()

        # Подготовка контекста целевых таблиц
        self._tgt_ctx_post_init()

        # Подготовка контекста
        self._map_ctx_post_init()

        self.delta_mode = 'new'

        hdp_processed: str = Config.setting_up_field_lists.get('hdp_processed', 'hdp_processed')
        hdp_processed_conversion = Config.setting_up_field_lists.get('hdp_processed_conversion', 'second')
        tgt_history_field = Config.setting_up_field_lists.get('tgt_history_field', '')

        # Формирование контекста для шаблона uni_res
        # Сделано отдельно от src_ctx, что-бы не "ломать" мозги
        self.uni_ctx = UniContext(source=self.source_system,
                                  schema=self.source_system_schema,
                                  table_name=self.src_ctx.name,
                                  src_cd=self.src_cd,
                                  hdp_processed=hdp_processed,
                                  hdp_processed_conversion=hdp_processed_conversion,
                                  tgt_history_field=tgt_history_field)

    def _get_tgt_table_fields(self) -> list:
        """
        Возвращает список полей целевой таблицы с типами данных и признаком "null"/"not null"
        Выполняет проверку типов и обязательных полей
        """

        corresp_datatype = Config.field_type_list.get("corresp_datatype", None)
        if not corresp_datatype:
            corresp_datatype = {
                'string': ['text'],
                'timestamp': ['timestamp'],
                'bigint': ['bigint'],
                'decimal': ['decimal'],
                'date': ['date']
            }

        def check_datatypes(row) -> bool:
            if type(row['src_attr_datatype']) is not str or type(row['tgt_attr_datatype']) is not str:
                return False

            # Проверка _id / _rk полей
            if (row['src_attr'].removesuffix('_id') and row['tgt_attribute'].removesuffix('_rk') and
                    row['src_attr_datatype'] == 'string' and row['tgt_attr_datatype'] == 'bigint'):
                return True

            if row['src_attr_datatype'] not in corresp_datatype.keys():
                logging.error(f"Тип данных источника '{row['src_attr_datatype']}' отсутствует в 'corresp_datatype'")
                return False

            return row['tgt_attr_datatype'] in corresp_datatype[row['src_attr_datatype']]

        is_error: bool = False

        src = self.mart_mapping[['src_table', 'src_attr', 'src_attr_datatype']].dropna(how='all')
        tgt = self.mart_mapping[['tgt_attribute', 'tgt_attr_datatype', 'tgt_attr_mandatory', 'tgt_pk', 'comment',
                                 '_pk', '_rk']].dropna(subset=['tgt_attribute', 'tgt_attr_datatype'])

        # Проверяем типы данных, заданные для источника. Читаем данные из настроек программы
        src_attr_datatype: dict = Config.field_type_list.get('src_attr_datatype', dict())
        err_rows = src[~src['src_attr_datatype'].isin(src_attr_datatype)][['src_attr', 'src_attr_datatype']]
        if len(err_rows) > 0:
            logging.error(f"Неверно указаны типы данных источника '{src.iloc[0].at['src_table']}':")
            for line in str(err_rows).splitlines():
                logging.error(line)
            logging.error(f'Допустимые типы данных: {src_attr_datatype}')
            is_error = True

        # Проверяем типы данных для целевой таблицы. Читаем данные из настроек программы
        tgt_attr_datatype: dict = Config.field_type_list.get('tgt_attr_datatype', dict())
        err_rows = tgt[~tgt['tgt_attr_datatype'].isin(tgt_attr_datatype)][['tgt_attribute', 'tgt_attr_datatype']]
        if len(err_rows) > 0:
            logging.error(f"Неверно указаны типы данных целевой таблицы '{self.mart_name}':")
            for line in str(err_rows).splitlines():
                logging.error(line)

            logging.error(f'Допустимые типы данных: {tgt_attr_datatype}')
            is_error = True

        # Если типы данных прошли проверку,
        # то выполняем "мягкую" проверку соответствия типа источника типу целевой таблицы
        if not is_error:
            # Выделяем колонки с типами данных.
            # Удаляем строки для которых не заполнены поля источника и/или целевой таблицы.
            data_types = self.mart_mapping[['src_attr', 'src_attr_datatype',
                                            'tgt_attribute', 'tgt_attr_datatype']].dropna(how='any')
            # Проверяем соответствие типов данных источника и целевой таблицы.
            tmp_df = data_types.apply(func=check_datatypes, axis=1, result_type='reduce')
            err_rows = data_types[~tmp_df]
            if len(err_rows) > 0:
                Config.is_warning = True
                logging.warning(f"Типы данных полей источника и целевой таблицы различаются")
                logging.warning(f"Проверьте корректность заполнения атрибутов")
                for line in str(err_rows).splitlines():
                    logging.warning(line)

        # Заполняем признак 'Tgt_attr_mandatory'.
        # При чтении данных Панда заменяет строку 'null' на значение 'nan'
        # Поэтому производим "обратную" замену ...
        # Заменяем "\xa0" на "null"

        # Как эта конструкция работает ...
        tgt.fillna({'tgt_attr_mandatory': "null"}, inplace=True)
        tgt.replace({'tgt_attr_mandatory': "\xa0"}, value="null",  inplace=True)

        err_rows = tgt[~tgt['tgt_attr_mandatory'].isin(['null', 'not null'])][['tgt_attribute', 'tgt_attr_mandatory']]
        if len(err_rows) > 0:
            logging.error(f"Неверно указан признак null/not null для целевой таблицы '{self.mart_name}':")
            for line in str(err_rows).splitlines():
                logging.error(line)
            is_error = True

        # Правим пустые комментарии: меняем NaN "null" на пустую строку
        tgt.fillna({'comment': ""}, inplace=True)

        # Проверка: Поля 'pk' должны быть "not null"
        err_rows = tgt.query('_pk in ["pk"] and tgt_attr_mandatory != "not null"')
        if len(err_rows) > 0:
            logging.error(f"Неверно указан признак 'Tgt_attr_mandatory' для целевой таблицы '{self.mart_name}':")
            logging.error("Поля отмеченные как 'pk' должны быть 'not null'")
            for line in str(err_rows).splitlines():
                logging.error(line)
            is_error = True

        # Проверка полей, тип которых фиксирован
        tgt_attr_predefined_datatype: dict = Config.field_type_list.get('tgt_attr_predefined_datatype', dict())
        for fld_name in tgt_attr_predefined_datatype.keys():
            err_rows = tgt.query(f"tgt_attribute == '{fld_name}'")
            if len(err_rows) == 0:
                logging.error(f"Не найден обязательный атрибут '{fld_name}' для целевой таблицы '{self.mart_name}'")
                is_error = True

            elif len(err_rows) > 1:
                logging.error(f"Обязательный атрибут '{fld_name}' для целевой таблицы '{self.mart_name}'"
                              f" указан более одного раза")
                for line in str(err_rows).splitlines():
                    logging.error(line)
                is_error = True

            else:
                if (err_rows.iloc[0]['tgt_attr_datatype'] != tgt_attr_predefined_datatype[fld_name][0] or
                        err_rows.iloc[0]['tgt_attr_mandatory'] != tgt_attr_predefined_datatype[fld_name][1]):
                    logging.error(
                        f"Параметры обязательного атрибута '{fld_name}' для целевой таблицы '{self.mart_name}'"
                        f" указаны неверно")
                    for line in str(err_rows).splitlines():
                        logging.error(line)
                    is_error = True

        # Проверяем соответствие названия полей целевой таблицы шаблону
        pattern: str = Config.get_regexp('tgt_attr_name_regexp')
        err_rows = tgt[~tgt.tgt_attribute.str.match(pattern).fillna(True)]
        if len(err_rows) > 0:
            logging.error(f"Названия полей целевой таблицы '{self.mart_name}' не соответствуют шаблону '{pattern}'")
            for index, fld_name in err_rows['tgt_attribute'].items():
                logging.error(fld_name)
            is_error = True

        if is_error:
            raise IncorrectMappingException(f"Неверно указаны атрибуты для целевой таблицы '{self.mart_name}'")

        return tgt.to_numpy().tolist()

    def _get_tgt_hub_fields(self) -> list:
        """
        Возвращает список атрибутов для hub-таблицы
        ["Имя_поля_в_источнике", "Имя_BK_Schema", "Имя_hub_таблицы",
         "признак_null", "поле_в_источнике",
         "имя_схемы", "имя_hub_таблицы_без_схемы", "short_name",
         "имя_поля_в_hub"]
        """

        hub: pd.DataFrame = self.mart_mapping[self.mart_mapping['attr:conversion_type'] == 'hub']
        hub = hub[['tgt_attribute', 'attr:bk_schema', 'attr:bk_object', 'attr:nulldefault', 'src_attr',
                   'expression', 'tgt_pk', 'tgt_attr_datatype', '_pk', 'src_attr_datatype', 'tgt_attr_mandatory']]
        hub_list = hub.to_numpy().tolist()

        # Проверяем корректность имен
        # Шаблон формирования short_name в wf.yaml
        # pattern = r"^[a-z][a-z0-9_]{2,22}$"
        # bk_object_pattern = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
        bk_object_regexp = Config.get_regexp('bk_object_regexp')
        bk_schema_regexp = Config.get_regexp('bk_schema_regexp')

        ret_list: list = list()
        for hh in hub_list:

            if hh[8] not in ['pk', 'bk']:
                logging.error(f"Значение поля 'tgt_pk' ('{hh[8]}') для хабов должно содержать значение 'pk' или 'bk'")
                logging.error(hh)
                raise IncorrectMappingException("Ошибка в данных EXCEL")

            # Проверяем соответствие названия БК-схемы шаблону
            if not re.match(bk_schema_regexp, hh[1]):
                logging.error(
                    f"Названия БК-схемы '{hh[1]}' не соответствуют шаблону '{bk_schema_regexp}'")
                raise IncorrectMappingException("Ошибка в данных EXCEL")

            # Значение NaN, которое так "любит" pandas "плохо" воспринимается другими библиотеками
            src_attr: str | None = hh[4] if not pandas.isna(hh[4]) else None
            expression: str | None = None
            if not pandas.isna(hh[5]):
                expression = hh[5]
                # Удаляем знак "="
                expression = expression.strip().removeprefix('=')

            if hh[10] != 'not null':
                logging.warning(f"Поле ссылки на хаб-таблицу '{hh[0]}' должно иметь атрибут 'not null'")
                Config.is_warning = True

            hub_ctx: HubFieldContext = HubFieldContext(name=hh[0],
                                                       bk_schema_name=hh[1],
                                                       hub_name=hh[2],
                                                       on_full_null=hh[3],
                                                       src_attr=src_attr,
                                                       expression=expression,
                                                       hub_schema=None,
                                                       hub_name_only=None,
                                                       hub_short_name=None,
                                                       hub_field=None,
                                                       is_bk='true' if hh[8] == 'bk' else 'false',
                                                       tgt_type=hh[7],
                                                       src_type=hh[9])

            h_schema: str
            h_name: str

            if not re.match(bk_object_regexp, hh[2]):
                logging.error(f"Значение hub_name '{hh[2]}' в поле 'attr:bk_object' не соответствует шаблону "
                              f"'{bk_object_regexp}'")
                raise IncorrectMappingException("Ошибка в структуре данных EXCEL")
            else:
                h_schema, h_name = hh[2].split('.')

            # Длина short_name должна быть от 2 до 22 символов
            if not re.match(bk_schema_regexp, h_name):
                h_short_name = ('hub_' + h_name.removeprefix('hub_')[0:12] + '_' +
                                ''.join(random.choice(string.ascii_lowercase) for i in range(5)))
            else:
                h_short_name = h_name

            hub_ctx.hub_schema = h_schema
            hub_ctx.hub_name_only = h_name
            hub_ctx.hub_short_name = h_short_name
            name_rk: str = hh[0]
            if name_rk.endswith('_rk'):
                name_id = re.sub(r'_rk$', r'_id', name_rk)
            else:
                name_id = name_rk + '_id'

            hub_ctx.hub_field = name_id

            ret_list.append(hub_ctx)

        ret_list.sort(key=lambda x: x.name)
        return ret_list

    def _get_src_table_fields(self) -> list:
        """
        Возвращает список полей источника с типами данных
        """
        src_attr: DataFrame = self.mart_mapping[['src_attr', 'src_attr_datatype', 'src_table']] \
            .dropna(how="any")

        src_tbl_name: str = src_attr.iloc[0]['src_table']

        # Удаление дубликатов в списке полей
        src_attr = src_attr.drop_duplicates(subset=['src_attr'])

        is_error: bool = False
        # Проверяем соответствие названия полей источника шаблону
        pattern: str = Config.get_regexp('src_attr_name_regexp')
        err_rows = src_attr[~src_attr.src_attr.str.match(pattern)]
        if len(err_rows) > 0:
            logging.error(f"Названия полей в таблице - источнике '{src_tbl_name}' не соответствуют шаблону '{pattern}'")
            for index, fld_name in err_rows['src_attr'].items():
                logging.error(fld_name)
            is_error = True

        # Проверяем обязательные поля таблицы - источника
        src_attr_predefined_datatype: dict = Config.field_type_list.get('src_attr_predefined_datatype', dict())
        for fld_name in src_attr_predefined_datatype.keys():
            err_rows = src_attr.query(f"src_attr == '{fld_name}'")
            if len(err_rows) == 0:
                logging.error(f"Не найден обязательный атрибут '{fld_name}' таблицы - источника '{src_tbl_name}'")
                is_error = True

            elif len(err_rows) > 1:
                logging.error(f"Обязательный атрибут '{fld_name}' для таблицы - источника '{src_tbl_name}'"
                              f" указан более одного раза")
                for line in str(err_rows).splitlines():
                    logging.error(line)
                is_error = True

            else:
                if err_rows.iloc[0]['src_attr_datatype'] != src_attr_predefined_datatype[fld_name][0]:
                    logging.error(
                        f"Параметры обязательного атрибута '{fld_name}' для таблицы - источника '{src_tbl_name}'"
                        f" указаны неверно")
                    for line in str(err_rows).splitlines():
                        logging.error(line)
                    is_error = True

        if is_error:
            raise IncorrectMappingException(f"Неверно указаны атрибуты таблицы - источника '{src_tbl_name}'")

        # Преобразуем к виду python list
        return src_attr.to_numpy().tolist()

    def _get_field_map(self) -> list:
        """
        Returns: Список атрибутов для заполнения секции field_map в шаблоне wf.yaml
        """
        mart_map = self.mart_mapping.where(self.mart_mapping['attr:conversion_type'] != 'hub')[
            ['src_attr', 'tgt_attribute', 'tgt_attr_datatype', 'src_attr_datatype']
        ].dropna().to_numpy().tolist()

        # Добавляем пустое поле
        for lst in mart_map:
            lst.append(None)

        # Список полей, которые рассчитываются
        mart_map_exp = self.mart_mapping.where(self.mart_mapping['attr:conversion_type'] != 'hub')[
            ['expression', 'tgt_attribute', 'tgt_attr_datatype']
        ].dropna().to_numpy().tolist()

        for lst in mart_map_exp:
            if not lst[0].startswith('='):
                logging.warning(f'Значение в колонке "Expression" должно начинаться со знака равно. src_attr={lst[1]}')
            else:
                mart_map.append([None, lst[1], lst[2], None, lst[0]])

        return mart_map

    def _src_ctx_post_init(self):
        # Имя таблицы-источника
        src_table_name = self.mart_mapping['src_table'].dropna().unique()[0].lower()
        # Список полей таблицы - источника
        src_field_ctx = self._get_src_table_fields()

        self.src_ctx = SourceContext(
            name=src_table_name,
            src_cd=self.src_cd,
            field_context=src_field_ctx,
            data_capture_mode=self.data_capture_mode
        )

        self.src_ctx.schema = self.source_system_schema

    def _tgt_ctx_post_init(self):
        tgt_field_ctx = self._get_tgt_table_fields()
        tgt_hub_field_ctx: list = self._get_tgt_hub_fields()
        self.tgt_ctx = TargetContext(
            name=self.mart_name,
            src_cd=self.src_cd,
            field_context=tgt_field_ctx,
            hub_context=tgt_hub_field_ctx
        )

    def _map_ctx_post_init(self):
        fld_map_ctx = self._get_field_map()
        # Код алгоритма
        algo = self.mart_mapping['algorithm_uid'].unique()[0]
        algo_sub = self.mart_mapping['subalgorithm_uid'].unique()[0]
        self.mapping_ctx = MappingContext(
            field_map_context=fld_map_ctx,
            src_cd=self.src_cd,
            src_name=self.src_ctx.name,
            src_schema=self.src_ctx.schema,
            tgt_name=self.tgt_ctx.name,
            algo=algo,
            algo_sub=algo_sub,
            data_capture_mode=self.data_capture_mode,
            work_flow_name=self.work_flow_name,
            hub_ctx_list=self.tgt_ctx.hub_ctx_list,
            source_system=self.source_system,

        )
