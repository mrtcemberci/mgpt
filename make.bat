@echo off
if "%1"=="clean" (
    echo Cleaning Release build directory...
    if exist build_release rmdir /s /q build_release
    echo Clean complete!
    exit /b 0
)

if "%1"=="config" (
    echo Configuring CMake for Release...
    cmake -S . -B build_release
    exit /b 0
)

echo Building MGPT Release...
cmake --build build_release --config Release --target mgpt
