"""Fail-closed selected-map and Keepout tests."""

from pathlib import Path
from types import SimpleNamespace

from ar_reference_planner.keepout_config import (
    KeepoutConfigurationError,
    MapSignature,
    params_with_keepout,
    resolve_keepout_selection,
)
from ar_reference_planner.keepout_gate import (
    KeepoutReadinessGate,
    occupancy_data_sha256,
)
from ar_reference_planner.test_fixture import create_test_map_bundle
import pytest


def _signature(data=(0, 100, -1, 0)):
    return MapSignature(
        yaml_path=Path('/test/map.yaml'),
        resolution=0.05,
        width=2,
        height=2,
        origin_x=-1.0,
        origin_y=-2.0,
        origin_yaw=0.0,
        data_sha256=occupancy_data_sha256(data),
    )


def _grid(data=(0, 100, -1, 0), *, origin_x=-1.0):
    orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
    origin = SimpleNamespace(
        position=SimpleNamespace(x=origin_x, y=-2.0),
        orientation=orientation,
    )
    info = SimpleNamespace(
        width=2, height=2, resolution=0.05, origin=origin)
    return SimpleNamespace(info=info, data=list(data))


def _filter_info(mask_topic='/keepout_filter_mask'):
    return SimpleNamespace(
        type=0,
        filter_mask_topic=mask_topic,
        base=0.0,
        multiplier=1.0,
    )


def test_disabled_map_is_verified_without_a_mask():
    """A deliberately disabled map retains ordinary reference planning."""
    selected, _ = create_test_map_bundle(False)

    result = resolve_keepout_selection(selected)

    assert not result.enabled
    assert result.reason == 'settings_disabled'
    assert result.mask_signature is None


def test_enabled_map_and_mask_are_tied_to_exact_content():
    """An enabled fixture resolves one exact selected map and mask."""
    selected, mask = create_test_map_bundle(True)

    result = resolve_keepout_selection(selected)

    assert result.enabled
    assert result.mask_signature.yaml_path == mask.resolve()
    assert result.map_signature.data_sha256 != result.mask_signature.data_sha256


def test_missing_enabled_mask_fails_closed():
    """Never silently plan without a mask declared as enabled."""
    selected, mask = create_test_map_bundle(True)
    mask.unlink()

    with pytest.raises(KeepoutConfigurationError, match='map YAML is missing'):
        resolve_keepout_selection(selected)


def test_source_map_mismatch_fails_closed():
    """Never use Keepout state belonging to a different selected map."""
    selected, _ = create_test_map_bundle(True)
    settings = selected.parent / '.keepout' / selected.stem / 'settings.json'
    contents = settings.read_text(encoding='utf-8').replace(
        str(selected), '/other/maps_waypoints/maps/other.yaml')
    settings.write_text(contents, encoding='utf-8')

    with pytest.raises(KeepoutConfigurationError, match='source_map'):
        resolve_keepout_selection(selected)


def test_map_metadata_mismatch_fails_closed():
    """Reject saved masks made for another map geometry."""
    selected, _ = create_test_map_bundle(True)
    settings = selected.parent / '.keepout' / selected.stem / 'settings.json'
    contents = settings.read_text(encoding='utf-8').replace(
        '"resolution": 0.05', '"resolution": 0.10')
    settings.write_text(contents, encoding='utf-8')

    with pytest.raises(KeepoutConfigurationError, match='map_metadata'):
        resolve_keepout_selection(selected)


def test_mask_geometry_mismatch_fails_closed():
    """Reject a mask YAML whose origin differs from the selected map."""
    selected, mask = create_test_map_bundle(True)
    contents = mask.read_text(encoding='utf-8').replace(
        'origin:\n- -5.0', 'origin:\n- -4.5')
    mask.write_text(contents, encoding='utf-8')

    with pytest.raises(KeepoutConfigurationError, match='mask geometry'):
        resolve_keepout_selection(selected)


def test_filter_is_injected_only_into_derived_parameters():
    """Keep the checked-in baseline free of unconditional filters."""
    source = {
        'global_costmap': {
            'global_costmap': {'ros__parameters': {'plugins': ['static']}}
        }
    }

    result = params_with_keepout(source, '/costmap_filter_info')

    assert 'filters' not in source['global_costmap']['global_costmap'][
        'ros__parameters']
    parameters = result['global_costmap']['global_costmap']['ros__parameters']
    assert parameters['filters'] == ['keepout_filter']
    assert parameters['keepout_filter']['filter_info_topic'] == (
        '/costmap_filter_info')


def test_disabled_runtime_gate_requires_exact_selected_map():
    """Map switching invalidates even a Keepout-disabled plan."""
    signature = _signature()
    gate = KeepoutReadinessGate(
        signature,
        keepout_required=False,
        mask_signature=None,
        filter_info_topic='/costmap_filter_info',
        mask_topic='/keepout_filter_mask',
    )

    gate.observe_map(_grid())
    assert gate.ready
    assert gate.reason == 'keepout_disabled_map_verified'

    gate.observe_map(_grid(origin_x=-1.1))
    assert not gate.ready
    assert 'geometry_mismatch' in gate.reason


def test_enabled_runtime_gate_requires_all_shared_inputs_active():
    """Exact map, mask, metadata and both lifecycle states are mandatory."""
    signature = _signature()
    gate = KeepoutReadinessGate(
        signature,
        keepout_required=True,
        mask_signature=signature,
        filter_info_topic='/costmap_filter_info',
        mask_topic='/keepout_filter_mask',
    )
    gate.observe_map(_grid())
    gate.observe_mask(_grid())
    gate.observe_filter_info(_filter_info())
    assert not gate.ready
    assert gate.reason == 'filter_mask_server_not_active'

    gate.set_lifecycle('mask', True)
    gate.set_lifecycle('info', True)
    assert gate.ready


@pytest.mark.parametrize(
    ('change', 'expected'),
    [
        ('mask_content', 'mask_content_mismatch'),
        ('mask_topic', 'mask_topic='),
        ('filter_type', 'filter_type='),
    ],
)
def test_runtime_gate_rejects_stale_or_wrong_filter_data(change, expected):
    """A stale mask/info stream must not authorize a new reference path."""
    signature = _signature()
    gate = KeepoutReadinessGate(
        signature,
        keepout_required=True,
        mask_signature=signature,
        filter_info_topic='/costmap_filter_info',
        mask_topic='/keepout_filter_mask',
    )
    gate.observe_map(_grid())
    if change == 'mask_content':
        gate.observe_mask(_grid(data=(0, 0, 0, 0)))
    else:
        gate.observe_mask(_grid())
    info = _filter_info(
        '/wrong_mask' if change == 'mask_topic' else '/keepout_filter_mask')
    if change == 'filter_type':
        info.type = 1
    gate.observe_filter_info(info)
    gate.set_lifecycle('mask', True)
    gate.set_lifecycle('info', True)

    assert not gate.ready
    assert expected in gate.reason
