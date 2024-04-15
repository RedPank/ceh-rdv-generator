import logging
from dataclasses import dataclass, field

import pandas

from core.config import Config as Conf, Config
from core.exceptions import IncorrectMappingException


@dataclass
class FieldContext:
    # Имя поля
    name: str
    # Тип поля
    datatype: str
    # Признак возможности присвоения значения NULL полю
    is_nullable: bool = True
    # Признак, что поле входит в состав "первичного" ключа таблицы
    pk: str = ''
    # Комментарий к полю
    comment: str = ''


@dataclass
class HubFieldContext:
    # [0] Имя_поля_в_источнике
    name: str
    # [1] Имя_BK_Schema
    bk_schema_name: str
    # [2] Имя_hub_таблицы со схемой
    hub_name: str
    # [3] Признак_null
    on_full_null: str
    # [4] Имя поля в таблице-источнике
    src_attr: str
    # [5] Вычисляемое выражение
    expression: str
    # [6] Имя схемы hub- таблицы
    hub_schema: str | None
    # [7] Имя_hub_таблицы без схемы
    hub_name_only: str | None
    # [8] Значение для поля short_name в шаблоне
    hub_short_name: str | None
    # [9] Имя поля в hub-таблице
    hub_field: str | None
    # [10] Если поле входит в PK mart-таблицы, то True
    is_bk: str | None
    # [11] Тип поля в целевой таблице
    tgt_type: str | None
    # [12] Тип поля в источнике
    src_type: str | None


@dataclass
class FieldMapContext:
    # Имя поля в целевой таблице
    tgt_field: str
    # Тип значения column/sql_expression
    type: str
    # Значение
    value: str
    # Тип поля в целевой таблице
    field_type: str


@dataclass
class TableContext:
    name: str
    src_cd: str
    field_context: list
    schema: str = '~'
    field_ctx_list: list[FieldContext] = field(default_factory=list)

    def __post_init__(self):
        # Проверка поля name на корректность, обработка в случае прихода "схема.таблица"
        match self.name.split('.'):
            case [schema_name, table_name]:
                self.schema = schema_name
                self.name = table_name
            case _:
                msg = f'Название таблицы {self.name} должно содержать название схемы'
                logging.error(msg)
                raise IncorrectMappingException(msg)

        # заполнение field_ctx_list с распаковкой данных из field_context
        for row in self.field_context:

            _pk: str = ''
            if len(row) >= 4 and row[5] == 'pk':
                _pk = 'pk'

            field_ctx = FieldContext(name=row[0],
                                     datatype=row[1],
                                     is_nullable=True if len(row) < 3 else row[2] != 'not null',
                                     pk=_pk,
                                     comment='' if len(row) < 5 else row[4])

            self.field_ctx_list.append(field_ctx)


@dataclass
class SourceContext(TableContext):
    data_capture_mode: str = None


@dataclass
class TargetContext(TableContext):
    hub_context: list = field(default_factory=list)

    # Список полей hub - таблиц
    hash_src_fields: list[str] = field(default_factory=list)
    # Список hub - таблиц
    hub_ctx_list: list[HubFieldContext] = field(default_factory=list)
    distributed_by: str = field(default_factory=str)
    multi_fields: set = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()

        # Список полей, которые не будут использоваться для формирования hash
        ignore_hash_set: list = Conf.setting_up_field_lists.get('ignore_hash_set', list())
        # Список полей, которые не включаются в опцию distributed_by / multi_fields
        ignore_distributed_src: list = Conf.setting_up_field_lists.get('ignore_distributed_src', list())

        # Цикл по списку hub_context
        for row in self.hub_context:
            self.hub_ctx_list.append(row)

        hub_fields: set = {hub_f.name for hub_f in self.hub_ctx_list}
        fields = {field_ctx.name for field_ctx in self.field_ctx_list}
        not_null_fields = {field_ctx.name for field_ctx in self.field_ctx_list if field_ctx.is_nullable is False}

        # Список первичных ключей для опции multi_fields. Поля, которые являются ссылками на hub - не включаются
        self.multi_fields = {field_ctx.name for field_ctx in self.field_ctx_list
                             if field_ctx.pk == "pk" and
                             field_ctx.name not in ignore_distributed_src and
                             field_ctx.name not in hub_fields}

        # Список первичных ключей для опции distributed by
        self.distributed_by = ','.join(sorted([field_ctx.name for field_ctx in self.field_ctx_list
                                               if field_ctx.pk == "pk"
                                               and field_ctx.name not in ignore_distributed_src]))

        # Поля, которые не входят в hash
        ignore_list: set = not_null_fields.union(ignore_hash_set)
        # Удаляем поля, которые являются ссылками на hub,  поля not null, поля из списка ignore_hash_set
        self.hash_src_fields = sorted(list(fields.difference(ignore_list).difference(hub_fields)))

        if len(self.hash_src_fields) > 100:
            Config.is_warning = True
            logging.warning(f"Количество полей для расчета hash: {len(self.hash_src_fields)}. "
                            f"Допустимое значение - 100")

        return


class UniContext:
    """
    Класс - контекст для заполнения шаблона uni_res.json
    Имена переменных должны совпадать с именами в шаблоне
    """
    # Имя источника (для uni-провайдера).
    source: str = None
    # Имя схемы таблицы-источника
    schema: str = None
    # Имя таблицы-источника
    table_name: str = None
    # Система источник (значение поля src_cd)
    src_cd: str = None
    # Имя "инстанса" - возможно описание и формирование неверно
    instance: str = None
    # Код ресурса
    resource_cd: str = None
    # Имя переменной
    actual_dttm_name: str = None
    # Имя поля даты источника, по которому ведется отбор новых записей
    hdp_processed: str = None
    # Округление даты
    hdp_processed_conversion: str | None
    # Имя поля историчности в целевой таблице
    tgt_history_field: str | None

    def __init__(self, source: str, schema: str, table_name: str, src_cd: str, hdp_processed: str,
                 hdp_processed_conversion: str, tgt_history_field: str):
        self.source = source.lower()
        self.schema = schema.lower()
        self.table_name = table_name.lower()
        self.src_cd = src_cd.lower()
        self.instance = self.source + '_' + self.schema.removeprefix("prod_")
        self.resource_cd = '.'.join([self.source, self.schema, self.table_name])
        self.actual_dttm_name = f"{self.src_cd}_actual_dttm"
        self.hdp_processed = hdp_processed
        self.hdp_processed_conversion = hdp_processed_conversion
        self.tgt_history_field = tgt_history_field


@dataclass
class MappingContext:
    field_map_context: list
    src_cd: str
    src_name: str
    src_schema: str
    tgt_name: str
    algo: str
    algo_sub: str
    data_capture_mode: str
    source_system: str
    # Список hub - таблиц
    # hub_pool: set[str]
    work_flow_name: str
    hub_ctx_list: list[HubFieldContext] = field(default_factory=list)
    field_map_ctx_list: list[FieldMapContext] = field(default_factory=list)
    delta_mode: str = 'new'

    def __post_init__(self):
        """
        Заполняет список полей/типов для секции field_map, используется в шаблоне wf.yaml
        Returns: None

        """
        # Список полей целевой таблицы, которые не будут добавлены в секцию field_map шаблона wf.yaml
        ignore_field_map_ctx_list: dict = Conf.setting_up_field_lists.get('ignore_field_map_ctx_list', dict())
        # Список полей с описанием, которые БУДУТ добавлены в секцию field_map шаблона wf.yaml
        add_field_map_ctx_lis: dict = Conf.setting_up_field_lists.get('add_field_map_ctx_list', dict())

        # 0-'src_attr', 1-'Tgt_attribute', 2-'Tgt_attr_datatype', 3-'src_attr_datatype', 4-'Expression'
        for row in self.field_map_context:
            # Поле - источник присутствует
            if not pandas.isnull(row[0]):
                # Необходимо явное преобразование типов
                if row[3] == 'string' and row[2] == 'timestamp':
                    field_map = FieldMapContext(tgt_field=row[1],
                                                type='sql_expression',
                                                value=row[0] + '::timestamp',
                                                field_type=row[2].upper()
                                                )
                elif row[3] == 'string' and row[2] == 'date':
                    field_map = FieldMapContext(tgt_field=row[1],
                                                type='sql_expression',
                                                value=row[0] + '::date',
                                                field_type=row[2].upper()
                                                )
                else:
                    field_map = FieldMapContext(tgt_field=row[1],
                                                type='column',
                                                value=row[0],
                                                field_type=row[2].upper()
                                                )
            # Выражение/Expression присутствует
            elif not pandas.isnull(row[4]):
                field_map = FieldMapContext(tgt_field=row[1],
                                            type='sql_expression',
                                            value=row[4].removeprefix('='),
                                            field_type=row[2].upper()
                                            )

            else:
                msg = f'Неверное описание поля: {row[0]}-{row[1]}-{row[2]}-{row[3]}-{row[4]}'
                raise IncorrectMappingException(msg)

            # Добавляем только те поля целевой таблицы, которые не входят в список исключений
            if row[1] not in ignore_field_map_ctx_list:
                self.field_map_ctx_list.append(field_map)

        # Добавляем поля
        if type(add_field_map_ctx_lis) is dict:
            for tgt_field in add_field_map_ctx_lis.keys():
                # Описание поля
                fld = add_field_map_ctx_lis[tgt_field]

                # Проверяем, если поле уже присутствует, то выдается ошибка
                if [ctx.tgt_field for ctx in self.field_map_ctx_list if ctx.tgt_field == tgt_field]:
                    msg = (f'Поле {tgt_field}, определенное в конфигурационном файле, '
                           f'нельзя добавить в список полей, т.к. оно уже присутствует в списке')
                    raise IncorrectMappingException(msg)

                deleted_flg_map = FieldMapContext(tgt_field=tgt_field,
                                                  type=fld['type'],
                                                  value=fld['value'],
                                                  field_type=fld['field_type']
                                                  )
                self.field_map_ctx_list.append(deleted_flg_map)
