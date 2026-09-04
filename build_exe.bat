@echo off
REM Сборка Windows .exe при помощи PyInstaller.
REM Запускать на Windows с установленным Python 3.10+.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name TensogramAnalyzer ^
    --hidden-import PIL._tkinter_finder ^
    main.py

echo.
echo Готово: dist\TensogramAnalyzer.exe
pause
