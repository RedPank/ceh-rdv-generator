: Найдите файл venv\Scripts\activate.bat
: Найдите сктроку, где вызывается "питон" и добавьте(через пробел) обработку параметров:  %1 %2 %3 %4 %5 %6
: Должно получиться что-то похожее на: C:\ceh-rdv-generator\venv\Scripts\python.exe %1 %2 %3 %4 %5 %6
: В командной строке ниже:
: - Укажите полный путь до файла main.py. 
: - В параметре "-c" укажите путь до конфигурационного файла. Для формирования данных из разных источников могут 
:   потребоваться разные конфигурации

: E:\GitHub\ceh-rdv-generator\venv\Scripts\activate.bat  полный_путь_до_файла\main.py -c полный_путь_до_файла\generator_ods.yaml 
E:\GitHub\ceh-rdv-generator\venv\Scripts\activate.bat  main.py -c generator_DAPP.yaml 