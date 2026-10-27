# 実機NUC setupと安全運用

## 配置原則

1. 開発PCでtest済みcommitまたはtagを指定する。
2. NUCの既存Robot/Camera workspaceとは別directoryへcloneする。
3. `git status`、commit SHA、`SOURCE_MANIFEST.sha256`を照合する。
4. 既存本番workspaceを`git pull`、copy上書き、package installで変更しない。
5. buildも独立したbuild/install/logを使用する。

例（接続先と配置先はREAD-ONLY調査後に確定）:

```bash
git clone --branch <tested-tag> --depth 1 \
  git@github.com:yuki-inoue12345/nakanoshima-robot-challenge.git \
  ~/nakanoshima-robot-challenge
cd ~/nakanoshima-robot-challenge
git rev-parse HEAD
sha256sum -c docs/SOURCE_MANIFEST.sha256
```

Camera V7を独立配置から起動する場合は、実機で確認したdomain IDをtemplateへ設定します。

```bash
cp camera_v7/robot.env.example camera_v7/robot.env
# robot.envのplaceholderをREAD-ONLY確認済みの数値へ置換する
export ROBOT_ENV="$PWD/camera_v7/robot.env"
```

`camera_v7/robot.env`はGit対象外です。既存本番側の`robot.env`を編集しません。

## READ-ONLY事前確認

```bash
printenv ROS_DOMAIN_ID
ps -ef | grep -E 'daikoudai_v7|waypoint_follow|nav2' | grep -v grep
ros2 node list
ros2 service list | grep koko_waypoint_follower
ros2 param get /koko_waypoint_follower navigation_mode
readlink -f /dev/v4l/by-id/*C920* 2>/dev/null
fuser /dev/video* 2>/dev/null
```

上記ROS CLIもDDS discovery通信を行うため、現場のREAD-ONLY確認として許可された
network/domainだけで実施します。service/action call、topic publishは行いません。

## 起動前ゲート

- operator、非常停止担当、物理非常停止、走行区域を確保する
- active route/map、`navigation_mode`、ROS distro/domainを記録する
- Camera V7の実行scriptと対象PythonのSHA256を照合する
- C920のby-idとvideo index、他processがopenしていないことを確認する
- model hash、Python/Ultralytics/Torch/OpenCV versionを確認する
- まず保存入力、次にcamera単体、最後にRobot連携の順で試験する
- 実機走行は別途許可された試験計画と低速設定でのみ行う

ARとCamera V7の双方がC920を直接所有する設定では同時起動しません。Waypoint、
`person_area`、Gemma応答を含む統合は、log時刻と実行commitを必ず記録します。
