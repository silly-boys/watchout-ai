#!/bin/bash
set -e

echo "[1/3] 시스템 패키지 설치 중..."
sudo apt-get update -qq
sudo apt-get install -y \
  python3-venv python3-dev pkg-config \
  libopencv-dev

echo "[2/3] 가상환경 생성 중..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

echo "[3/3] Python 패키지 설치 중 (시간이 오래 걸릴 수 있습니다)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "완료! 실행: source venv/bin/activate && python main.py"
