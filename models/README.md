# Models（Git管理外）

Model binaryはこのrepositoryへ含めません。

| Model | 識別情報 | 管理 |
|---|---|---|
| YOLO11n | `yolo11n.pt`, SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`, 5,613,764 bytes | 配布条件を確認したartifact store |
| Gemma | `gemma-4-e2b-it-qat` | LM Studio側で取得・管理 |

Git LFSは準備PCで未導入です。YOLO weightは小容量でも、Ultralytics/weightの
配布条件確認が完了するまで別管理とします。NUCでは承認された入手元から取得し、
起動前にSHA256を照合してください。Gemma modelをGitへコピーしないでください。
