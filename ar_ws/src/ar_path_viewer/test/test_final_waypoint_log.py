"""Follower's own threshold notice is not a Nav2 SUCCEEDED state."""
from ar_path_viewer.final_waypoint_log import verified_final_arrival


def test_follower_final_arrival_match():
    assert verified_final_arrival('koko_waypoint_follower',
           '[WP:30] threshold reached at 0.34 m', 0.1, 30, 30) == 0.34


def test_other_waypoint_or_other_rosout_node_rejected():
    assert verified_final_arrival('koko_waypoint_follower',
           '[WP:29] threshold reached at 0.99 m', 0.1, 30, 30) is None
    assert verified_final_arrival('fake_follower',
           '[WP:30] threshold reached at 0.34 m', 0.1, 30, 30) is None


def test_stale_messages_and_wrong_number_rejected():
    assert verified_final_arrival('koko_waypoint_follower',
           '[WP:30] threshold reached at 0.34 m', 2.0, 30, 30) is None
    assert verified_final_arrival('koko_waypoint_follower',
           '[WP:30] threshold reached at 0.34 m', 0.1, 31, 30) is None


def test_cancel_or_map_failure_is_not_arrival():
    assert verified_final_arrival('koko_waypoint_follower',
           'Map switch failed: Keepout is active', 0.1, 30, 30) is None
    assert verified_final_arrival('bt_navigator',
           'Goal canceled', 0.1, 30, 30) is None


def test_different_follower_and_nonsequential_final_number():
    assert verified_final_arrival(
        'nakanoshima_follower', '[WP:87] threshold reached at 0.32 m',
        0.1, 87, 16, expected_node_name='nakanoshima_follower',
    ) == 0.32
    assert verified_final_arrival(
        'koko_waypoint_follower', '[WP:87] threshold reached at 0.32 m',
        0.1, 87, 16, expected_node_name='nakanoshima_follower',
    ) is None
    assert verified_final_arrival(
        'nakanoshima_follower', '[WP:0] threshold reached at 0.20 m',
        0.1, 0, 1, expected_node_name='nakanoshima_follower',
    ) == 0.20
