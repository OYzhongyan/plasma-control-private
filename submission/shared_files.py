"""Atomic bounded mailbox I/O. Linux opens stay pinned beneath trusted roots."""
from contextlib import ExitStack, contextmanager
import errno
import os
from pathlib import Path
import re
import secrets
import stat


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,128}', value) or value in {'.', '..'}:
        raise ValueError('Invalid mailbox filename')
    return value


class Directory:
    def __init__(self, path, *, parent=None, name=None, create=False):
        self.path, self.fd = Path(path), None
        created = False
        if parent is not None:
            _name(name)
            if create:
                try:
                    if parent.fd is not None: os.mkdir(name, 0o2770, dir_fd=parent.fd)
                    else: self.path.mkdir(mode=0o2770)
                    created = True
                except FileExistsError: pass
            if parent.fd is not None:
                try:
                    self.fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent.fd)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}: raise ValueError('Mailbox directories must not be symlinks') from exc
                    raise
            elif self.path.is_symlink():
                raise ValueError('Mailbox directories must not be symlinks')
        elif os.name == 'posix':
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        if self.fd is None and not self.path.is_dir():
            raise FileNotFoundError('Mailbox directory does not exist')
        if created and self.fd is not None: os.fchmod(self.fd, 0o2770)

    def open(self, name, flags, mode=0o660):
        _name(name)
        if self.fd is not None:
            try:
                fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=self.fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP: raise ValueError('Mailbox files must not be symlinks') from exc
                raise
        else:
            path = self.path / name
            if path.is_symlink(): raise ValueError('Mailbox files must not be symlinks')
            fd = os.open(path, flags | getattr(os, 'O_BINARY', 0), mode)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError('Mailbox entry must be a regular file')
        return fd

    def read(self, name, maximum):
        with os.fdopen(self.open(name, os.O_RDONLY), 'rb') as stream:
            if os.fstat(stream.fileno()).st_size > maximum: raise ValueError('Mailbox file is too large')
            raw = stream.read(maximum + 1)
        if len(raw) > maximum: raise ValueError('Mailbox file is too large')
        return raw

    def write(self, name, raw):
        _name(name)
        temporary = '.tmp-' + secrets.token_hex(16)
        try:
            with os.fdopen(self.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY), 'wb') as stream:
                if os.name == 'posix': os.fchmod(stream.fileno(), 0o660)
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if self.fd is not None:
                os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            else: os.replace(self.path / temporary, self.path / name)
        finally:
            self.unlink(temporary)

    def unlink(self, name):
        _name(name)
        try:
            if self.fd is not None: os.unlink(name, dir_fd=self.fd)
            else: (self.path / name).unlink()
        except FileNotFoundError: pass

    def names(self, maximum=256):
        result = []
        with os.scandir(self.fd if self.fd is not None else self.path) as entries:
            for entry in entries:
                result.append(entry.name)
                if len(result) >= maximum: break
        return sorted(result)

    @contextmanager
    def lock(self, name):
        # Never unlink a lock file: replacing its inode would split the lock.
        try: fd = self.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        except FileExistsError: fd = self.open(name, os.O_RDWR)
        else:
            if os.name == 'posix': os.fchmod(fd, 0o660)
        acquired = False
        try:
            try:
                if os.name == 'posix':
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt
                    if os.fstat(fd).st_size == 0: os.write(fd, b'0')
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as exc: raise RuntimeError('This mailbox is already in use') from exc
            yield
        finally:
            if acquired:
                if os.name == 'posix':
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET); msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None


class Mailbox:
    def __init__(self, root, agent_id, *, create=False):
        if not isinstance(agent_id, str) or not re.fullmatch('[0-9a-f]{32}', agent_id):
            raise ValueError('Invalid agent ID')
        self.stack = ExitStack()
        try:
            root = Path(root).resolve()
            if create:
                try:
                    root.mkdir(parents=True, mode=0o2770)
                    if os.name == 'posix': root.chmod(0o2770)
                except FileExistsError: pass
            self.root = self._own(Directory(root))
            self.directory = self._own(Directory(root / agent_id, parent=self.root, name=agent_id, create=create))
            self.requests = self._own(Directory(root/agent_id/'requests', parent=self.directory, name='requests', create=create))
            self.responses = self._own(Directory(root/agent_id/'responses', parent=self.directory, name='responses', create=create))
        except BaseException:
            self.close(); raise

    def _own(self, directory):
        self.stack.callback(directory.close)
        return directory

    def lock(self, name): return self.directory.lock(name)
    def close(self): self.stack.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
