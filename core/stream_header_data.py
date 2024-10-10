import logging
import re

import pandas as pd

from core.exceptions import IncorrectMappingException


class StreamHeaderData:
    """
    Класс представляет данные с листа 'Перечень загрузок Src-RDV' для указанной целевой таблицы.
    Названия полей соответствуют названиям колонок EXCEL
    """
    row: pd.DataFrame
    row_dict: dict

    version: str
    version_end: str
    algorithm_uid: str
    subalgorithm_uid: str
    flow_name: str
    base_flow_name: str
    tgt_table: str
    target_rdv_object_type: str
    src_table: str
    # Колонка source_name
    source_system: str
    scd_type: str
    # algo_name: str
    # data_filtering: str
    distribution_field: str
    distribution_field_list: list
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
        self.version = self.row_dict["version"]
        self.version_end = self.row_dict["version_end"]
        self.algorithm_uid = re.sub(r"\s", '', self.row_dict["algorithm_uid"])
        self.subalgorithm_uid = self.row_dict["subalgorithm_uid"]
        self.flow_name = re.sub(r"\s", '', self.row_dict["flow_name"])
        self.tgt_table = re.sub(r"\s", '', self.row_dict["tgt_table"])
        self.target_rdv_object_type = re.sub(r"\s", '', self.row_dict["target_rdv_object_type"])
        self.src_table = re.sub(r"\s", '', self.row_dict["src_table"])
        self.source_system = re.sub(r"\s", '', self.row_dict["source_name"]).upper()
        self.scd_type = re.sub(r"\s", '', self.row_dict["scd_type"])
        # self.algo_name = self.row_dict["algo_name"]
        # self.data_filtering = self.row_dict["data_filtering"]
        self.distribution_field = self.row_dict["distribution_field"]
        self.distribution_field = self.distribution_field.lower()
        self.distribution_field = re.sub(r"\s", "", self.distribution_field)
        self.distribution_field_list = self.distribution_field.split(',')
        self.distribution_field_list.sort()
        # self.comment = self.row_dict["comment"]
        # Имя потока без wf_/cf_
        self.base_flow_name = self.flow_name.removeprefix('wf_')
