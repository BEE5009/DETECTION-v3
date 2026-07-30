import sys


def configure_console_encoding(encoding='utf-8'):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding=encoding, errors='replace')
        except Exception:
            pass


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for arg in args:
            if isinstance(arg, Exception):
                arg = str(arg)
            if isinstance(arg, str):
                safe_args.append(arg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)
    except Exception:
        try:
            sys.__stdout__.write(' '.join(str(arg) for arg in args) + '\n')
        except Exception:
            pass
