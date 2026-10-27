# Security、privacy、license確認

## Secret・個人情報

選定fileに一般的なcredential file名、private key、token代入がないことをcommit前に
検索します。preflight内の`password`文字列はURL userinfo拒否のvalidationであり、
credentialではありません。録画・実験画像・ROS bagは施設情報や人物を含み得るため
Git対象外です。

Waypoint/mapは施設経路情報を含む可能性があります。このrepositoryはPrivateを維持し、
collaboratorも必要最小限にします。Public化は行いません。

## License状況

| 対象 | metadataで確認できたlicense | 注意 |
|---|---|---|
| `ar_path_viewer` | Apache-2.0 | package metadataによる |
| Robot 3 packages | MIT | package metadataによる |
| Camera V7独自source | 明示license未確認 | Private共同開発に限定し、外部配布前に権利者確認 |
| Ultralytics | AGPL-3.0（参照venv metadata） | 利用・配布形態を公開前に確認 |
| OpenCV Python | Apache-2.0（参照venv metadata） | dependencyとして再構築 |
| PyTorch | BSD-3-Clause（参照venv metadata） | dependencyとして再構築 |
| YOLO weight | 未確定 | repositoryに含めず配布条件を確認 |
| Gemma model | model固有条件 | LM Studio側で管理しGitに含めない |
| `sounds/*.wav` | 出典・再配布条件未確認 | Private限定。外部配布前に差替えまたは権利確認 |

この一覧は法的判断ではなく、repository収録可否の保守的な管理記録です。

## Commit前audit

```bash
git ls-files -z | xargs -0 grep -IlE \
  '(api[_-]?key|secret|token|password|BEGIN .*PRIVATE KEY)' || true
find . -type f -size +10M -print
git status --short
git diff --cached --check
```

候補を検出した場合は文脈を確認し、実credentialをcommitしません。
