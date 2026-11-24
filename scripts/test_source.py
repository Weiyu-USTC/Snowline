#!/usr/bin/env python3
"""Run source-only forecast contracts; outbound network connections are forbidden.

This intentionally excludes research replay tests requiring downloaded archives.
The optional MOS replay and real DPR decoder tests skip if their fixtures are absent.
Only local HTTP test-server connections are allowed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ipaddress
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    'test_runtime_install',
    'test_dashboard_forecasts',
    'test_dashboard_alta',
    'test_dashboard_neural48',
    'test_dashboard_portfolio',
    'test_dashboard_server',
    'test_dashboard_season',
    'test_season_observations',
    'test_mammoth_observations',
    'test_mammoth_radar',
    'test_mammoth_weather',
    'test_train_mammoth_hourly',
    'test_mammoth_forecast',
    'test_mammoth_depth',
    'test_refresh_deployed_forecasts',
    'test_alta_collins_fallback',
    'test_forecast_portfolio',
    'test_forecast_neural48',
    'test_hrrr_portfolio',
    'test_nexrad_acquire_cache',
    'test_nexrad_current_validation',
    'test_nexrad_short',
    'test_neural48_dataset',
    'test_neural48_nwp',
    'test_neural48_radar',
    'test_neural48_targets',
    'test_neural48_models',
    'test_train_neural48',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'src')]
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def check_address(address):
        if not isinstance(address, tuple) or len(address) < 2:
            raise AssertionError('Source tests must not open non-IP sockets')
        host = address[0]
        if host == 'localhost':
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError(f'Outbound network access is disabled in source tests: {host}')

    def local_connect(sock, address):
        check_address(address)
        return connect(sock, address)

    def local_connect_ex(sock, address):
        check_address(address)
        return connect_ex(sock, address)

    with tempfile.TemporaryDirectory(prefix='snowline-tests-') as temp, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {'MPLCONFIGDIR': temp}))
        stack.enter_context(patch.object(socket.socket, 'connect', local_connect))
        stack.enter_context(patch.object(socket.socket, 'connect_ex', local_connect_ex))
        suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
