
def filter(s):
    s.replace('*', '')
    s.replace('-', ' ')
    s.replace('\n', ' ')
    s.replace('\t', ' ')
    return s
