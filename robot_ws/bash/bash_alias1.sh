# ココちゃん（robotbase_ws）専用の基本ショートカット。
# Sirius側と共存できるよう、公開する名前はすべて koko_* に限定する。

_KOKO_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${_KOKO_WS}/robot.env"

# Siriusをsource済みの端末でも検索パスを混在させず、ココちゃん環境へ切り替える。
alias koko_env='source "${_KOKO_WS}/bash/activate_koko_env.sh"'
alias koko_sim_env='source "${_KOKO_WS}/bash/activate_koko_sim_env.sh"'

# .bashrc編集
alias koko_ebash='code ~/.bashrc && source ~/.bashrc'

# ROSのインストールディレクトリ
alias koko_rosapt='code /opt/ros/jazzy/share/'

# robotbase_wsへ移動
alias koko_ws='cd "${_KOKO_WS}"'

# ココちゃん環境を有効化
alias koko_src='koko_env && koko_ws && source install/setup.bash'
alias koko_sim_src='koko_sim_env && koko_ws && source install/setup.bash'

# rosdep / build
alias koko_rdep='koko_ws && rosdep install --from-paths src --ignore-src -riy'
alias koko_build='koko_env && koko_ws && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --executor sequential --allow-overriding nav2_costmap_2d'

# ROS診断
alias koko_rqt_console='koko_src && ros2 run rqt_console rqt_console'
alias koko_tftree='koko_src && ros2 run rqt_tf_tree rqt_tf_tree'

# 注意: 同じDomain 57に属するROSプロセス全体へ影響する。
alias koko_ros_daemon_restart='koko_env && ros2 daemon stop && ros2 daemon start'

# メモリキャッシュ解放（システム全体へ影響するため明示名）
alias koko_system_freemem='sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" && echo "Memory cache cleared!"'
