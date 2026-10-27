# RVizユーザー設定

`koko_rviz_sim` と `koko_rviz_real` は、TF接頭辞ごとの `robotbase_<prefix>.rviz` をこのフォルダから読み込む。

- 現在の既定ファイル: `robotbase_robot.rviz`
- RVizの通常の `File -> Save Config` で同じファイルへ保存され、次回起動にも反映される。
- `robot.env` でTF接頭辞を変更すると、その接頭辞用ファイルがテンプレートから自動生成される。
- テンプレートの状態へ戻す場合はlaunchへ `reset_config:=true` を一度だけ渡す。現在のユーザー設定は上書きされる。

配布用テンプレートは `src/robotbase_bringup/rviz/robotbase.rviz` にあり、通常のSave Configでは変更されない。

