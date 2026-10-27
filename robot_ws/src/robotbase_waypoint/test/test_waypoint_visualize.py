"""Verify route ordering, orientation and replacement in RViz."""

import math

import pytest
from robotbase_waypoint.waypoint_io import Waypoint
from robotbase_waypoint.waypoint_visualize import route_markers
from visualization_msgs.msg import Marker


def test_markers_preserve_yaml_order_numbers_and_heading():
    """Nonsequential numbers do not change route order or marker identity."""
    message = route_markers([
        Waypoint(number=70, x=2.0, y=3.0, angle_radians=math.pi),
        Waypoint(number=5, x=-1.0, y=4.0, angle_radians=0.0, stop=True),
    ], 'map')
    assert message.markers[0].action == Marker.DELETEALL
    line, first_arrow, first_label, second_arrow, second_label = message.markers[1:]
    assert [(p.x, p.y) for p in line.points] == [(2.0, 3.0), (-1.0, 4.0)]
    assert (first_label.text, second_label.text) == ('70', '5')
    assert (first_arrow.id, second_arrow.id) == (0, 1)
    assert first_arrow.pose.orientation.z == pytest.approx(1.0)
    assert first_arrow.pose.orientation.w == pytest.approx(0.0, abs=1e-12)
    assert second_arrow.pose.orientation.w == 1.0
    assert first_arrow.pose.position.x == first_label.pose.position.x == 2.0
    assert first_arrow.color.r != second_arrow.color.r
    for marker in message.markers[1:]:
        assert marker.header.frame_id == 'map'
        assert marker.lifetime.sec == 0
        assert marker.pose.orientation.w ** 2 + marker.pose.orientation.z ** 2 == (
            pytest.approx(1.0))
