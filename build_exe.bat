@echo off
REM Сборка Windows .exe при помощи PyInstaller.
REM Запускать на Windows с установленным Python 3.10+.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Модули scipy._external.array_api_compat.numpy.* (scipy >= 1.18) /
REM scipy._lib.array_api_compat.numpy.* (более старые scipy) подгружаются
REM внутри scipy динамически (через __import__ по строке), и старые версии
REM PyInstaller не видят такой импорт статическим анализом — из-за этого
REM сборка падает с ModuleNotFoundError при запуске exe. Актуальные версии
REM PyInstaller уже содержат исправление в своём хуке для scipy, но флаги
REM ниже не дают сборке сломаться даже на устаревшей версии PyInstaller
REM (несуществующий для установленной версии scipy путь просто даёт
REM безобидное предупреждение, а не ошибку сборки).
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name TensogramAnalyzer ^
    --hidden-import PIL._tkinter_finder ^
    --hidden-import scipy._external.array_api_compat.numpy.fft ^
    --hidden-import scipy._external.array_api_compat.numpy.linalg ^
    --hidden-import scipy._lib.array_api_compat.numpy.fft ^
    --hidden-import scipy._lib.array_api_compat.numpy.linalg ^
    main.py

echo.
echo Готово: dist\TensogramAnalyzer.exe
pause
