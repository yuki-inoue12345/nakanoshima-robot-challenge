from types import SimpleNamespace

import pytest

from ar_path_viewer.route_destination import (
    destination_from_reference_state,
    direct_distance_m,
)


def _state(*, valid=True, route='a' * 64, count=2, x=4.0, y=6.0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='map'),
        final_waypoint_valid=valid,
        final_waypoint_number=30,
        final_waypoint_position=SimpleNamespace(x=x, y=y),
        route_sha256=route,
        waypoint_count=count,
    )


def test_typed_reference_metadata_exposes_only_direct_final_distance():
    destination = destination_from_reference_state(_state())

    assert destination.waypoint_number == 30
    assert direct_distance_m(destination, 'a' * 64, 2, 1.0, 2.0) == 5.0
    assert destination_from_reference_state(_state(valid=False)) is None


def test_route_hash_count_and_current_identity_must_match():
    with pytest.raises(ValueError, match='hash_invalid'):
        destination_from_reference_state(_state(route='not-a-hash'))
    with pytest.raises(ValueError, match='count_invalid'):
        destination_from_reference_state(_state(count=0))
    with pytest.raises(ValueError, match='position_invalid'):
        destination_from_reference_state(_state(x=float('nan')))

    destination = destination_from_reference_state(_state())
    assert direct_distance_m(destination, '1' * 64, 2, 1.0, 2.0) is None
    assert direct_distance_m(destination, 'a' * 64, 3, 1.0, 2.0) is None
    assert direct_distance_m(
        destination, 'a' * 64, 2, float('nan'), 2.0
    ) is None


def test_route_destination_requires_map_frame():
    message = _state()
    message.header.frame_id = 'odom'
    with pytest.raises(ValueError, match='frame_mismatch'):
        destination_from_reference_state(message)
