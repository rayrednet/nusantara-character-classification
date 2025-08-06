# utils/alias_normalizer.py
EXCEPTIONS = {'allah', 'nabilah', 'kahlil', 'permaisuri', 'istri', 'puteri', 'karyawan'}

def normalize_alias_custom(name):
    name = name.lower().strip()
    if name in EXCEPTIONS:
        return name

    tokens = name.split()

    if len(tokens) == 1:
        for suffix in ['nya', 'ku', 'mu']:
            if name.endswith(suffix) and len(name) > len(suffix) + 2:
                name = name[:-len(suffix)]

    for suffix in ['lah', 'kah', 'tah', 'nda']:
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[:-len(suffix)]

    return name.strip()
