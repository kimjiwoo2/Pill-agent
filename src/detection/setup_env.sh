#!/usr/bin/env bash
# 연구실 GPU 머신용 격리 venv 생성 (전역 파이썬 안 건드림, sudo 불필요).
# 사용:  bash setup_env.sh [ENV_DIR]
#   ENV_DIR 기본값 = $HOME/envs/pill-obb  (⚠️ Drive 동기화 폴더 안에 만들지 말 것)
set -e

ENV_DIR="${1:-$HOME/envs/pill-obb}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] Python: $(python3 --version 2>&1)"

echo "[2/4] venv 생성 → $ENV_DIR"
if ! python3 -m venv "$ENV_DIR" 2>/dev/null; then
  echo "  venv 실패(ensurepip 없음). micromamba 폴백 권장:"
  echo '    "${SHELL}" <(curl -L micro.mamba.pm/install.sh)'
  echo "    micromamba create -n pill-obb python=3.11 -y && micromamba activate pill-obb"
  echo "    pip install -r $HERE/requirements.txt"
  exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "[3/4] 패키지 설치"
python -m pip install -U pip wheel
pip install -r "$HERE/requirements.txt"

echo "[4/4] GPU 인식 확인"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda:", ok,
      ("| " + torch.cuda.get_device_name(0)) if ok else "(GPU 미인식 — CUDA 휠 확인)")
PY

echo
echo "완료. 매번 이걸로 활성화:  source $ENV_DIR/bin/activate"
echo "이후 실행 예:  python $HERE/obb_rotation_train.py --mode verify --data-root /data/pill_obb"
