import sys

def get_arg_by_name_simple(arg_name, default=None):
    """Simple extraction from sys.argv (format must be --name=value or --name value)"""
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == f'--{arg_name}':
            # Handle --name value format
            if i+1 < len(args) and not args[i+1].startswith('--'):
                return args[i+1]
        elif arg.startswith(f'--{arg_name}='):
            # Handle --name=value format
            return arg.split('=')[1]
    return default

