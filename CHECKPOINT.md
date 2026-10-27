# CHECKPOINT

- Current phase: source separation、provenance、local build/test、Private Git管理準備完了
- Private remote: `https://github.com/yuki-inoue12345/nakanoshima-robot-challenge`
- Initial source snapshot: `cb6ca41d64f441953407e40d7e44b9be5fe91a8a`
- Source snapshot date: 2026-10-27
- Completed:
  - ARの実体を`ar_navigation_project/ar_path_viewer`と特定
  - Camera V7のruntime、production、developmentを区別して収録
  - Robot 2026-10-25 archiveからWaypoint/person_area連携と関連資産を収録
  - origin copy 189 filesのSHA256一致を確認
  - ROS 2 4 packages build、132 tests PASS
  - Camera V7 py_compile、213 tests PASS
  - model、録画、bag、venv、build、cache、credentialをGit対象外に設定
  - model/license/privacyの未確認事項を文書化
- Not completed:
  - 現在のNUC実行版とのREAD-ONLY hash照合
  - C920、Gemma model、実ROS、実機走行での統合検証
  - Camera独自sourceと音声assetの外部再配布条件確認
- Safety:
  - 実機ROSへのpublish/service/action callはこの準備作業では未実施
  - Camera/Robot/ARの元directoryは変更していない
  - 実機本番workspaceをGit pullで直接更新しない
- Next exact action:
  - test済みcommit/tagを独立したNUC directoryへ取得し、`docs/SETUP_NUC.md`の
    READ-ONLY照合と保存入力testから開始する
