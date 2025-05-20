#!/bin/bash

# 가상환경 디렉토리 이름
VENV_DIR="venv"
PYTHON_PREFERRED_VERSION="3.11"

# Python 실행 명령어 결정
PYTHON_CMD=""

if command -v "python$PYTHON_PREFERRED_VERSION" &>/dev/null; then
    PYTHON_CMD="python$PYTHON_PREFERRED_VERSION"
    echo "Found preferred Python version: $PYTHON_CMD"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    echo "Preferred Python version $PYTHON_PREFERRED_VERSION not found. Using default python3: $PYTHON_CMD"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    echo "Preferred Python version $PYTHON_PREFERRED_VERSION and python3 not found. Using default python: $PYTHON_CMD"
else
    echo "Error: Python 3 (preferably version $PYTHON_PREFERRED_VERSION) is not installed. Please install Python 3."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# 가상환경 생성
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# requirements.txt 설치 (가상환경 내 pip 사용)
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt into '$VENV_DIR'..."
    # 운영체제에 따라 activate 스크립트 경로 및 pip 실행 파일 경로가 다를 수 있음
    # macOS/Linux:
    if [ -f "$VENV_DIR/bin/pip" ]; then
        "$VENV_DIR/bin/pip" install -r requirements.txt
    # Windows (Git Bash 등):
    elif [ -f "$VENV_DIR/Scripts/pip.exe" ]; then
        "$VENV_DIR/Scripts/pip.exe" install -r requirements.txt
    else
        echo "Error: pip executable not found in the virtual environment."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install packages from requirements.txt."
        exit 1
    fi
    echo "Packages installed successfully."
else
    echo "Warning: requirements.txt not found. No packages were installed."
fi

echo "
Setup complete.
To activate the virtual environment, run:
  source $VENV_DIR/bin/activate  (on macOS/Linux)
  source $VENV_DIR/Scripts/activate (on Windows with Git Bash or similar)

To deactivate, simply type: deactivate
" 