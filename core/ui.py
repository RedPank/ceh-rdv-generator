import subprocess
import tkinter as tk
from tkinter import ttk, SOLID
from tkinter.messagebox import showinfo, showerror, showwarning
from tkinter import filedialog
import logging

from jinja2 import TemplateNotFound

from core.map_gen import mapping_generator
import core.exceptions as exp
from core.config import Config as Conf


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        author: str = Conf.author
        self.file_path = tk.StringVar(value=Conf.excel_file)
        self.author = tk.StringVar(value=author)

        self.env = Conf.env

        self.wm_title("Генератор файлов описания src-rdv потока")
        # self.geometry("500x500")

        frame = tk.Frame(
            self,    # Обязательный параметр, который указывает окно для размещения Frame.
            padx=5,  # Задаём отступ по горизонтали.
            pady=5,  # Задаём отступ по вертикали.
            borderwidth=1,
            relief=SOLID
        )
        frame.pack(anchor='nw', fill='both', padx=5, pady=10)

        file_path_text = ttk.Entry(
            frame,
            textvariable=self.file_path,
            font=("Arial", 10),
            state='readonly')
        file_path_text.pack(fill=tk.X, padx=25, pady=10)

        open_file_dialog_button = ttk.Button(
            frame,
            text="Выбрать EXCEL-файл с маппингом",
            command=self._setup_file_path
        )
        open_file_dialog_button.pack(fill=tk.X, padx=25, pady=0)

        # Дополнительная информация
        text_info: str = (f'Конфиг:  {Conf.config_file}\n'
                          f'Шаблоны: {Conf.templates_path}\n'
                          '\n'
                          f'Каталог: {Conf.out_path}\n'
                          f'Журнал:  {Conf.log_file}\n'
                          f'Автор:   {author}\n'
                          )
        self.info_text = tk.StringVar(value=text_info)
        label_info = tk.Text(frame, font=("Courier New", 9), height=6)
        label_info.insert(index=tk.END, chars=text_info)
        label_info.configure(state=tk.DISABLED)
        label_info.pack(fill=tk.X, padx=25, pady=10)

        # Фрейм кнопок
        frame_key = tk.Frame(frame,  # Обязательный параметр, который указывает окно для размещения Frame.
                             padx=5,  # Задаём отступ по горизонтали.
                             pady=5,  # Задаём отступ по вертикали.
                             borderwidth=0,
                             relief=SOLID
                             )
        frame_key.pack(anchor='nw', fill='both', padx=5, pady=5)
        frame_key.columnconfigure(0, weight=1)
        frame_key.columnconfigure(1, weight=1)
        frame_key.columnconfigure(2, weight=1)

        start_export_button = tk.Button(
            frame_key,
            text="Формировать",
            command=self._export_mapping
        )
        start_export_button.grid(row=0, column=0, sticky=tk.E, padx=10)

        view_log_button = tk.Button(
            frame_key,
            text="Открыть журнал",
            command=self._view_log
        )
        view_log_button.grid(row=0, column=1, sticky=tk.E, padx=10)

        exit_button = tk.Button(
            frame_key,
            text="Завершить",
            command=self.destroy
        )
        exit_button.grid(row=0, column=2, sticky=tk.E, padx=10)

    def _setup_file_path(self):
        initial_dir: str = Conf.out_path
        filetypes = [('Excel files', '*.xls'), ('Excel files', '*.xlsx'), ('All files', '*.*')]
        title = "Выбор файла"

        file_path: str = filedialog.askopenfilename(filetypes=filetypes, initialdir=initial_dir, title=title)
        if file_path:
            self.file_path.set(file_path)

    @staticmethod
    def _view_log():
        subprocess.Popen(args=Conf.log_viewer)

    def _export_mapping(self):
        msg: str

        if not self.file_path.get():
            showerror("Ошибка", "EXCEL-файл с описанием данных не выбран")
            return

        if not all((
                self.file_path.get(),
                # self.out_path.get(),
                # self.author.get(),
        )):
            showerror("Ошибка", "Проверьте заполнение полей формы")
        else:
            try:

                logging.info('Формирование файлов описания потоков ...')

                mapping_generator(
                    file_path=self.file_path.get(),
                    out_path=Conf.out_path,
                    env=self.env,
                    author=self.author.get()
                )

                if Conf.is_error:
                    msg = ("Во время обработки были ошибки.\n"
                           "Прочитайте описание ошибок (error) "
                           "в журнале работы программы!")
                    showerror("Ошибка", msg)
                    logging.info("Обработка завершена с ошибками")

                elif Conf.is_warning:
                    msg = ("Обработка завершена c предупреждениями.\n"
                           "Прочитайте предупреждения (warning) "
                           "в журнале работы программы.")
                    showwarning("Предупреждение", msg)
                    logging.info("Обработка завершена с предупреждениями")

                else:
                    msg = ("Обработка завершена без ошибок.\n"
                           "Для получения подробной информации о сформированных потоках\n"
                           "прочитайте журнал работы программы.")
                    showinfo("Успешно", msg)
                    logging.info('Обработка завершена без ошибок')

            except (exp.IncorrectMappingException, ValueError) as err:
                logging.error(err)
                msg = f"Ошибка: {err}.\nПроверьте журнал работы программы."
                showerror(title="Ошибка", message=msg)

            except TemplateNotFound:
                msg = "Ошибка чтения шаблона.\nПроверьте журнал работы программы."
                logging.exception("Ошибка чтения шаблона")
                showerror(title="Ошибка", message=msg)
