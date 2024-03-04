import os
import shutil
from jinja2 import Environment, Template

from .mapping import MartMapping
from .context import SourceContext, TargetContext, MappingContext, UniContext

import yaml
from core.config import Config as Conf


def _get_tags_list(tags_tmpl, tags_val) -> list:
    """
    Args:
        tags_tmpl: Шаблон для формирования результата
        tags_val: Данные для формирования результата

    Returns: Список строк для вставки в секцию tags
    """

    ret_tags: list

    # Делаем шаблон из секции tags
    tags_template = Template(str(tags_tmpl))
    # Формируем исходные данные
    tag_list = yaml.safe_load(tags_template.render(tags=tags_val))
    # Формируем список строк
    ret_tags = list()
    for tag in tag_list:
        if type(tag) is str:
            ret_tags.append(f"\"{tag}\"")
        elif type(tag) is dict:
            key = list(tag.keys())[0]
            val = list(tag.values())[0]
            ret_tags.append(f"\"{key}:{val}\"")

    return ret_tags


class SourceObjectExporter:
    src_ctx: SourceContext
    uni_ctx: UniContext
    env: Environment

    # Название шаблона
    template_name: str = 'db_table.yaml'        # Описание таблицы-источника
    template_uni_json: str = 'uni_res.json'     # Название шаблона ресурса UNI

    def __init__(self, env, ctx, uni_ctx):
        self.env = env
        self.src_ctx = ctx
        self.uni_ctx = uni_ctx

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        template = self.env.get_template(self.template_name)
        output = template.render(ctx=self.src_ctx)
        file_name: str = os.path.join(path, self.src_ctx.name + '.yaml')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    def export_uni_resource(self, path):
        """
        Функция формирования файла с описанием uni-ресурса
        Args:
            path: Каталог для формирования файла

        Returns: None
        """
        file_path = os.path.join(path, self.uni_ctx.source, self.uni_ctx.schema)
        os.makedirs(file_path, exist_ok=True)

        template = self.env.get_template(self.template_uni_json)
        output = template.render(ctx=self.uni_ctx)

        file_name = '.'.join([self.uni_ctx.source, self.uni_ctx.schema, self.uni_ctx.table_name, "json"])
        file_name = os.path.join(file_path, file_name)
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)


class TargetObjectExporter:
    tgt_ctx: TargetContext
    uni_ctx: UniContext
    env: Environment

    template_name_yaml: str = 'mart.yaml'               # Название шаблона описания март-таблицы
    template_hub_yaml: str = 'hub.yaml'                 # Название шаблона описания хаб-таблицы
    template_hub_create_sql: str = 'hub_create.sql'     # Название шаблона создания/заполнения хаб-таблицы
    template_name_sql: str = 'mart_ddl.sql'             # Название шаблона скрипта создания март-таблицы
    template_name_json: str = 'ceh_res.json'            # Название шаблона ресурса CEH
    template_hub_json: str = 'ceh.hub_table.json '      # Название шаблона ресурса хаб-таблицы
    template_bk_json: str = 'ceh_bk_schema.json'        # Название шаблона ресурса БК-схемы хаб-таблицы

    def __init__(self, env, ctx, uni_ctx):
        self.env = env
        self.tgt_ctx = ctx
        self.uni_ctx = uni_ctx

    def export_yaml(self, path):
        """
        Создание файлов описания март и хеш таблиц
        Args:
            path: Каталог, в котором будут создаваться файлы

        Returns: None
        """
        values: dict = {'src_cd': self.tgt_ctx.src_cd}

        os.makedirs(path, exist_ok=True)

        # Файл описания март-таблицы
        template = self.env.get_template(self.template_name_yaml)
        output = template.render(ctx=self.tgt_ctx, uni_ctx=self.uni_ctx)
        file_name: str = os.path.join(path, self.tgt_ctx.name + '.yaml')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

        # Файлы описания хаб-таблиц
        for hub in self.tgt_ctx.hub_ctx_list:
            template = self.env.get_template(self.template_hub_yaml)
            output = template.render(hub=hub, values=values)
            file_name: str = os.path.join(path, f'{hub.hub_name_only}.yaml')
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(output)

    def export_hub_sql(self, path):
        os.makedirs(path, exist_ok=True)
        # Файлы описания хаб-таблиц
        for hub in self.tgt_ctx.hub_ctx_list:
            # Создание/заполнение хаб-таблиц
            template = self.env.get_template(self.template_hub_create_sql)
            output = template.render(hub=hub)
            file_name: str = os.path.join(path, f'{hub.hub_name_only}.sql')
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(output)

    def export_sql(self, path):
        os.makedirs(path, exist_ok=True)

        template = self.env.get_template(self.template_name_sql)
        output = template.render(ctx=self.tgt_ctx)
        file_name: str = os.path.join(path, '01-' + self.tgt_ctx.name + '.sql')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

        file_name: str = os.path.join(path, '02-gen_access_view.sql')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write('-- Скрипт формирования акцессоров должен быть здесь!')

    def export_sql_view(self, path):
        """
        Формирует скрипт, для создания views на целевую таблицу (акцессоров)
        Args:
            path: Каталог, в котором будет сформирован файл

        Returns: None
        """
        os.makedirs(path, exist_ok=True)
        template = self.env.get_template('f_gen_access_view.sql')
        output = template.render(ctx=self.tgt_ctx)
        file_name: str = os.path.join(path, 'f_gen_access_view.sql')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    def export_ceh_resource(self, path):
        os.makedirs(path, exist_ok=True)

        # Данные для формирования секции "tags""
        tags_val = {}
        tags: list = _get_tags_list(tags_tmpl=Conf.resource_tags, tags_val=tags_val)

        actual_dttm: str = f"{self.tgt_ctx.src_cd}_actual_dttm".lower()
        # Словарь с доп. параметрами для шаблона
        values: dict = {"actual_dttm_name": actual_dttm}

        # Ресурс целевой mart - таблицы
        template = self.env.get_template(self.template_name_json)
        output = template.render(ctx=self.tgt_ctx, uni_ctx=self.uni_ctx, values=values, tags=tags)
        file_name: str = os.path.join(path, f'ceh.{self.tgt_ctx.schema}.{self.tgt_ctx.name}.json')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

        for hub in self.tgt_ctx.hub_ctx_list:
            # Ресурс хаб-таблицы
            template = self.env.get_template(self.template_hub_json)
            output = template.render(hub=hub, values=values, tags=tags)
            file_name: str = os.path.join(path, f'ceh.{hub.hub_name}.json')
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(output)

            # Ресурс БК-схемы
            template = self.env.get_template(self.template_bk_json)
            output = template.render(hub=hub, values=values, tags=tags)
            file_name: str = os.path.join(path, f'ceh.{hub.hub_name}.{hub.bk_schema_name}.json')
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(output)


class MappingObjectExporter:
    map_ctx: MappingContext
    uni_ctx: UniContext
    env: Environment
    wf_file: str
    cf_file: str
    author_name: str

    template_wf_name: str = 'wf.yaml'  # Название шаблона WF
    template_cf_name: str = 'cf.yaml'  # Название шаблона CF
    template_py_name: str = 'wf.py'    # Название PY файла рабочего потока

    def __init__(self, env, ctx, author, uni_ctx, tags):
        self.env = env
        self.map_ctx = ctx
        self.author_name = author
        self.wf_file = f"wf_{self.map_ctx.work_flow_name}"
        self.cf_file = f"cf_{self.map_ctx.work_flow_name}"
        self.uni_ctx = uni_ctx
        self.tags = tags

    def _get_filled_cf_mapping(self):
        template = self.env.get_template(self.template_cf_name)
        return template.render(
            ctx=self.map_ctx,
            wf_file=self.wf_file,
            cf_file=self.cf_file,
            author=self.author_name,
            uni_ctx=self.uni_ctx,
            tags=self.tags
        )

    def export_wf(self, path):
        os.makedirs(path, exist_ok=True)

        template = self.env.get_template(self.template_wf_name)
        output = template.render(ctx=self.map_ctx, wf_file=self.wf_file, uni_ctx=self.uni_ctx, tags=self.tags)

        file_name: str = os.path.join(path, self.wf_file + '.yaml')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    def export_cf(self, path):
        os.makedirs(path, exist_ok=True)

        output = self._get_filled_cf_mapping()

        file_name: str = os.path.join(path, self.cf_file + '.yaml')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    def export_py(self, path):
        os.makedirs(path, exist_ok=True)

        file_name: str = os.path.join(path, self.wf_file + '.py')
        shutil.copy(os.path.join(Conf.templates_path, 'wf.py'), file_name)


class MartPackExporter:
    exp_obj: MartMapping
    path: str

    _src_exporter: SourceObjectExporter
    _tgt_exporter: TargetObjectExporter
    _mapping_exporter: MappingObjectExporter

    def __init__(self, exp_obj, path, env, author):
        self.exp_obj = exp_obj
        self.path = path

        # Данные для формирования секции "tags""
        tags_val = {'src_cd': self.exp_obj.mapping_ctx.src_cd, 'src_tbl': self.exp_obj.src_ctx.name,
                    'prv': self.exp_obj.mapping_ctx.source_system,
                    'tgt': self.exp_obj.tgt_ctx.schema, 'tgt_tbl': self.exp_obj.tgt_ctx.name,
                    'cf_flow': 'cf_' + self.exp_obj.mapping_ctx.work_flow_name,
                    'wf_flow': 'wf_' + self.exp_obj.mapping_ctx.work_flow_name, 'alg': self.exp_obj.mapping_ctx.algo}

        self.tags: list = _get_tags_list(tags_tmpl=Conf.tags, tags_val=tags_val)

        self._src_exporter = SourceObjectExporter(env, self.exp_obj.src_ctx, self.exp_obj.uni_ctx)
        self._tgt_exporter = TargetObjectExporter(env=env, ctx=self.exp_obj.tgt_ctx, uni_ctx=self.exp_obj.uni_ctx)
        self._mapping_exporter = MappingObjectExporter(env=env, ctx=self.exp_obj.mapping_ctx, author=author,
                                                       uni_ctx=self.exp_obj.uni_ctx, tags=self.tags)

    def load(self):
        """
        Формирование целевых файлов
        Returns: None

        """
        # Описание таблицы - источника
        exp_path = os.path.join(self.path, r"etl-scale\general_ledger\src_rdv\schema\db_tables")
        self._src_exporter.export(exp_path)

        # Описание uni - ресурса таблицы источника
        exp_path = os.path.join(self.path, r"etl-scale\_resources\uni")
        self._src_exporter.export_uni_resource(exp_path)

        # Описание целевой таблицы (mart)
        exp_path = os.path.join(self.path, r"etl-scale\general_ledger\src_rdv\schema\ceh\rdv")
        self._tgt_exporter.export_yaml(exp_path)

        # Скрипт создания целевой таблицы (mart)
        exp_path = os.path.join(self.path, r"adgp\extensions\ripper\.data")
        self._tgt_exporter.export_sql(exp_path)

        # Необязательные скрипты создания/заполнения hub - таблиц
        exp_path = os.path.join(self.path, r"src\hub")
        self._tgt_exporter.export_hub_sql(exp_path)

        # Скрипт для формирования view на целевую таблицу
        exp_path = os.path.join(self.path, r"src")
        self._tgt_exporter.export_sql_view(exp_path)

        # Описание ресурса целевой таблицы (mart)
        exp_path = os.path.join(self.path, r"etl-scale\_resources\ceh\rdv")
        self._tgt_exporter.export_ceh_resource(exp_path)

        # Рабочий поток (wf_*.yaml)
        exp_path = os.path.join(self.path, r"etl-scale\general_ledger\src_rdv\schema\work_flows")
        self._mapping_exporter.export_wf(exp_path)

        # py - файл потока управления (cf_*.py)
        exp_path = os.path.join(self.path, r"etl-scale\general_ledger\src_rdv\flow_dumps")
        self._mapping_exporter.export_cf(exp_path)

        # py - файл рабочего потока (wf_*.py)
        exp_path = os.path.join(self.path, r"etl-scale\general_ledger\src_rdv\dags")
        self._mapping_exporter.export_py(exp_path)
