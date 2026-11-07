"""Shared messages are authenticated data, never submitted paths or code."""
import json
import os
from pathlib import Path
import time

import pytest


def test_descriptor_is_automatic_private_and_path_free(tmp_path):
    from submission.shared_channel import make_descriptor, read_descriptor, write_descriptor
    first, second = make_descriptor(), make_descriptor()
    assert first['agent_id'] != second['agent_id']
    assert set(first) == {'version', 'protocol', 'transport', 'agent_id', 'access_token'}
    path = tmp_path / 'endpoint.json'
    write_descriptor(path, first)
    assert read_descriptor(path) == first
    if os.name == 'posix': assert path.stat().st_mode & 0o077 == 0
    for updates in ({'agent_id': '../outside'}, {'shared_root': '/etc'}, {'service_url': 'http://example.com'},
                    {'access_token': ''}, {'version': True}):
        path.write_text(json.dumps({**first, **updates}))
        with pytest.raises(ValueError): read_descriptor(path)


def test_signature_binds_direction_payload_expiry_and_request(tmp_path):
    from submission.shared_channel import make_descriptor, encode_packet, decode_packet, digest
    endpoint = make_descriptor()
    raw = encode_packet(endpoint, 'a' * 32, 'request', {'method': 'act'}, expires_at=time.time()+30)
    assert endpoint['access_token'].encode() not in raw
    request = decode_packet(raw, endpoint, 'request', request_id='a'*32)
    assert request['payload']['method'] == 'act'
    with pytest.raises(ValueError): decode_packet(raw, endpoint, 'response')
    value = json.loads(raw); value['payload']['method'] = 'reset'
    with pytest.raises(ValueError): decode_packet(json.dumps(value).encode(), endpoint, 'request')
    response = encode_packet(endpoint, 'a'*32, 'response', {'ok': True}, expires_at=time.time()+30,
                             request_digest=digest(raw))
    with pytest.raises(ValueError): decode_packet(response, endpoint, 'response', request_digest='wrong')
    with pytest.raises(ValueError): decode_packet(raw, endpoint, 'request', now=time.time()+60)


def test_atomic_bounded_files_and_exclusive_lock(tmp_path):
    from submission.shared_files import Mailbox
    with Mailbox(tmp_path, 'a'*32, create=True) as mailbox:
        mailbox.requests.write('b'*32+'.json', b'{"ok":true}')
        assert mailbox.requests.read('b'*32+'.json', 100) == b'{"ok":true}'
        with pytest.raises(ValueError): mailbox.requests.read('b'*32+'.json', 2)
        assert mailbox.requests.names() == ['b'*32+'.json']
        with mailbox.lock('client.lock'):
            with Mailbox(tmp_path, 'a'*32) as other:
                with pytest.raises(RuntimeError):
                    with other.lock('client.lock'): pass
        with mailbox.lock('client.lock'): pass
        with pytest.raises(ValueError): mailbox.requests.write('../outside', b'x')


def test_symlinked_mailbox_is_rejected(tmp_path):
    from submission.shared_files import Mailbox
    outside = tmp_path/'outside'; outside.mkdir()
    try: (tmp_path/('a'*32)).symlink_to(outside, target_is_directory=True)
    except OSError: pytest.skip('Symlink creation unavailable on this Windows account')
    with pytest.raises(ValueError): Mailbox(tmp_path, 'a'*32)


@pytest.mark.skipif(os.name != 'posix', reason='Linux directory-fd confinement')
def test_pinned_directory_survives_symlink_swap_without_writing_outside(tmp_path):
    from submission.shared_files import Mailbox
    root = tmp_path/'shared'; root.mkdir()
    outside = tmp_path/'outside'; outside.mkdir()
    with Mailbox(root, 'a'*32, create=True) as mailbox:
        requests = root/('a'*32)/'requests'
        requests.rename(requests.with_name('old-requests'))
        requests.symlink_to(outside, target_is_directory=True)
        mailbox.requests.write('b'*32+'.json', b'{}')
        assert not list(outside.iterdir())
        assert (requests.with_name('old-requests')/('b'*32+'.json')).exists()


@pytest.mark.skipif(os.name != 'posix', reason='Project-group mode on Linux')
def test_new_mailboxes_grant_project_group_write_despite_umask(tmp_path):
    from submission.shared_files import Mailbox
    previous = os.umask(0o077)
    try:
        with Mailbox(tmp_path/'new-root', 'a'*32, create=True):
            for path in [tmp_path/'new-root', tmp_path/'new-root'/('a'*32), tmp_path/'new-root'/('a'*32)/'requests']:
                assert path.stat().st_mode & 0o770 == 0o770
    finally: os.umask(previous)
