# Temporary fix for removed 'cgi' module in Python 3.13
def parse_header(line):
    return line, {}

def parse_multipart(fp, pdict):
    return {}, []
