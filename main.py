import argparse
import ctypes
import logging
import os
import tkinter

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

    win = MainWindow()

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VTB.ceh.ceh-rdv-generator.1_0")
    win.iconbitmap(r"res\ceh-icon.ico")
    image = tkinter.PhotoImage(file=r"res\ceh-icon.png")
    win.iconphoto(True, image)

    win.mainloop()

    logging.info('STOP')
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
else:
    exit(100)
