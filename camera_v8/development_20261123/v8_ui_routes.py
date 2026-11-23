"""Pure routing policy for the V7-layout V8 UI; no ROS or hardware imports."""

import json


class V8RoutingError(ValueError):
    pass


def build_live_search_command(python, engine, config, common_args, tuning_args):
    """Run the original V8 ROS-only engine, never V7 live-camera recognition.

    The input is the existing V7 UI's target profile dictionary; photo target
    extraction and appearance matching stay in their original unmodified engines.
    """
    if config.get('input_mode') != 'camera':
        raise V8RoutingError('V8 ROS人物探索には「実機カメラ」を選択してください。動画実験は直接実行を使用します。')
    targets = config.get('targets')
    if not isinstance(targets, list) or not targets or not all(isinstance(t, dict) for t in targets):
        raise V8RoutingError('探索対象設定が不正です')
    try:
        encoded = json.dumps(targets, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise V8RoutingError('探索対象JSONのエンコードに失敗しました') from exc
    topics = [config.get(k) for k in ('person_arrival_topic', 'person_done_topic', 'person_detail_topic')]
    if not all(isinstance(t, str) and t.startswith('/') and ' ' not in t for t in topics):
        raise V8RoutingError('ROSトピック名が不正です')
    cmd = [str(python), str(engine), '--person-search-server',
           '--arrival_topic', topics[0], '--done_topic', topics[1],
           '--detail_topic', topics[2], '--targets_json', encoded,
           '--raw_image_topic', '/koko/person_search/raw_image']
    cmd.extend(common_args)
    cmd.extend(tuning_args)
    bad_flags = {'--record_run_video', '--record_search_video', '--camera_device', '--camera_index'}
    if any(arg in bad_flags for arg in cmd):
        raise V8RoutingError('V8実機人物探索で直カメラ・録画指定を検出したため拒否しました')
    return cmd


def choose_photo_preview_mode(shared_owner_exists, camera_busy):
    """Never open C920 a second time, including when regular AR already owns it."""
    if shared_owner_exists:
        return 'ros_shared'
    if camera_busy:
        raise V8RoutingError('C920は他のアプリが使用中です。通常ARを終了するか共有ARを使ってください。')
    return 'standalone'
