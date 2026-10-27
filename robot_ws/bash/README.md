# ココちゃん用Bashショートカット

シリウス用aliasとの競合を避けるため、このWSが公開するコマンドはすべて `koko_*` です。

既存の `robotbase_ws` と併用する場合は、[ランチャー2の導入手順](../DOCS/LAUNCHER2.md) を使ってください。元の `koko_*` を維持するため、以下を `.bashrc` に追加しません。

以下は、このWSだけを端末の既定環境として使う場合の設定です。

```bash
source ~/robotbase_ws_renya/bash/bash_alias1.sh
source ~/robotbase_ws_renya/bash/bash_alias2.sh
```

読み込んだだけでは現在の `ROS_DOMAIN_ID` を変更しません。`koko_*` の実行時だけ `robot.env` に従ってココちゃん用DomainとGazebo partitionへ切り替わります。

`koko_env` はSirius由来の検索パスも除去します。`koko_build` もこのクリーン環境でビルドするため、`install/setup.bash` がSiriusを親overlayとして再登録することを防ぎます。

主なコマンド:

```bash
koko_src       # 実機: robotbase_ws_renya + ROS_DOMAIN_ID=54
koko_sim_src   # シミュレーション: robotbase_ws_renya + ROS_DOMAIN_ID=67
koko_build
koko_sim
koko_rviz_sim
koko_slamtoolbox_sim
koko_nav2_sim_map
koko_nav2_sim_slam
koko_keyop2_sim
koko_map_save_sim
koko_hokuyo
koko_slamtoolbox_real
koko_nav2_real
koko_nav2_real_slam
koko_rviz_real
koko_launcher2
```

表示名・Domain IDの変更とデスクトップUIについては `DOCS/LAUNCHER.md` を参照してください。
