import argparse
import ctypes
import logging
import os
import pathlib
import tkinter
import pandas as pd
from core.config import Config
from core.ui import MainWindow


def main() -> int:

    parser = argparse.ArgumentParser(prog="ceh-rdv-generator")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default='generator.yaml',
        help="Файл конфигурации"
    )
    args = parser.parse_args()

    # Файл настройки программы.
    config_name: str = os.path.abspath(args.config)
    Config.load_config(config_name=config_name)

    logging.basicConfig(level=logging.INFO, filename=Config.log_file, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s",
                        encoding='utf-8')

    print(f"config={config_name}")
    print(f"log_file={Config.log_file}")

    logging.info('START')
    logging.info(f"config={config_name}")
    logging.info(f"log_file={Config.log_file}")
    logging.info(f'templates_path="{Config.templates_path}"')

    # Настройка pandas
    # Включение режима "копирование при записи"
    pd.options.mode.copy_on_write = True
    # Печатаем все колонки
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 255)

    win = MainWindow()

    # Смена иконки программы
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VTB.ceh.ceh-rdv-generator.1_0")
    resource_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'res')
    win.iconbitmap(os.path.join(resource_path, "ceh-icon.ico"))
    image = tkinter.PhotoImage(file=os.path.join(resource_path, "ceh-icon.png"))
    win.iconphoto(True, image)

    win.mainloop()

    logging.info('STOP')
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
else:
    exit(100)
