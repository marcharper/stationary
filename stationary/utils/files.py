import os

def ensure_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def ensure_digits(num, s):
    """Prepends a string s with zeros to enforce a set num of digits."""
    if len(s) < num:
        return "0"*(num - len(s)) + s
    return s