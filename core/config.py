import logging
import yaml
from jinja2 import Environment, FileSystemLoader
import os
from pathlib import Path
from core.exceptions import IncorrectConfigException


class Config:
    # Declare the static variables
    config: any
    tags: any
    resource_tags: any
    setting_up_field_lists: dict
    field_type_list: dict
    excel_data_definition: dict
    log_file: str
    env: any
    templates_path: str
    excel_file: str
    config_file: str
    is_warning: bool = False

    @staticmethod
    def load_config(config_name: str):

        if not os.path.exists(config_name):
            msg = f'Не найден файл конфигурации программы "{config_name}"'
            print(msg)
            raise FileExistsError(msg)

        Config.config_file = config_name
        with open(config_name, 'r', encoding='utf-8') as f:
            Config.config = yaml.safe_load(f)

        Config.tags = Config.config.get('tags', dict())
        Config.resource_tags = Config.config.get('resource_tags', dict())

        Config.setting_up_field_lists = Config.config.get('setting_up_field_lists', dict())
        Config.field_type_list = Config.config.get('field_type_list', dict())
        Config.excel_data_definition = Config.config.get('excel_data_definition', dict())

        Config.excel_file = Config.config.get('excel_file', '')

        # "Загрузка" файлов-шаблонов
        Config.templates_path = os.path.abspath(Config.config.get('templates', 'templates'))
        if not os.path.exists(Config.templates_path):
            msg = f'Не найден каталог с шаблонами "{Config.templates_path}"'
            print(msg)
            raise FileExistsError(msg)

        Config.env = Environment(loader=FileSystemLoader(Config.templates_path))

        # Файл журнала
        log_file: str = Config.config.get('log_file', 'generator.log')
        log_file = log_file.strip()
        log_file = 'generator.log' if not log_file else log_file
        Config.log_file = os.path.join(Path(__file__).parents[1], log_file)
        if os.path.exists(Config.log_file):
            if os.path.isfile(Config.log_file):
                os.remove(Config.log_file)
            else:
                raise FileExistsError(f'Объект "{Config.log_file}" не является файлом')

    @staticmethod
    def get_regexp(name: str) -> str:
        if name in Config.config:
            return Config.config.get(name)
        else:
            logging.error(f'В файле конфигурации не найден шаблон с именем "{name}"')
            raise IncorrectConfigException()


