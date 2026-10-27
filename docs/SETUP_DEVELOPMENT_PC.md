# 開発PC setup

## ROS workspace

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/nrc-log build \
  --base-paths ar_ws/src robot_ws/src \
  --build-base /tmp/nrc-build \
  --install-base /tmp/nrc-install
```

Robot driver等のsystem依存がないPCでは、packageごとにbuild/testし、利用できない
実機依存をPASSと扱いません。AR viewerはread-only設計ですが、C920入力を選ぶと
Camera V7と競合するため同時起動しません。

## Camera V7 Python

```bash
cd camera_v7/development_20261019
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r ../requirements.txt
python -B v7_startup_preflight.py --dependencies-only
```

YOLO modelは`models/README.md`のhashを照合して別管理先から配置します。Gemmaは
LM Studioのlocal API/modelとして管理します。Git repositoryへmodelを追加しません。

## Test

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/nrc-log test \
  --base-paths ar_ws/src robot_ws/src \
  --build-base /tmp/nrc-build \
  --install-base /tmp/nrc-install

cd camera_v7/development_20261019
venv/bin/python -m unittest discover -s . -p 'test_v7_*.py'
```

Camera、ROS graph、走行を伴わないtestから開始します。camera open、Gemma API起動、
Robot接続は別段階の検証です。
