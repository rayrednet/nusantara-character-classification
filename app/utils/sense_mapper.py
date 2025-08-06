import pandas as pd
import re
from collections import defaultdict

# === ROLE KEYWORDS ===
ROLE_KEYWORDS = [
    ('ayah', 'ayah'),
    ('bapak', 'ayah'),
    ('pak', 'ayah'),
    ('suami', 'ayah'),
    ('yah', 'ayah'),
    ('papa', 'ayah'),
    ('pa', 'ayah'),

    ('ibu', 'ibu'),
    ('bunda', 'ibu'),
    ('bu', 'ibu'),
    ('istri', 'ibu'),
    ('isteri', 'ibu'),
    ('istrinya', 'ibu'),
    ('istri petani', 'ibu'),
    ('mama', 'ibu'),
    ('ma', 'ibu'),

    ('kakek', 'kakek'),
    ('kek', 'kakek'),

    ('nenek', 'nenek'),
    ('nek', 'nenek'),

    ('anak', 'anak'),
    ('nak', 'anak'),
    ('puteri sulung', 'anak perempuan sulung'),
    ('puteri bungsu', 'anak perempuan bungsu'),

    ('puteri', 'anak perempuan'),
    ('putri', 'anak perempuan'),
    ('gadis', 'anak perempuan'),

    ('putera', 'anak laki-laki'),
    ('kanda', 'anak laki-laki'),

    ('kakak', 'kakak'),
    ('kak', 'kakak'),
    ('kaka', 'kakak'),
    ('ka', 'kakak'),

    ('adik', 'adik'), 
    ('adek', 'adik'), 
    ('dik', 'adik'),

    ('abang', 'abang'), 
    ('bang', 'abang'),

    ('wak', 'penolong'),
    ('pawang', 'penolong'),

    ('penduduk', 'warga'),
    ('warga', 'warga'),
    ('rakyat', 'warga'),
    ('masyarakat', 'warga'),

    ('tuhan', 'tuhan'),
    ('tuhan yang maha esa', 'tuhan'),
    ('yang maha esa', 'tuhan'),
    ('yang maha kuasa', 'tuhan'),
    ('maha kuasa', 'tuhan'),
    ('yang maha agung', 'tuhan'),
    ('maha agung', 'tuhan'),
    ('yang kuasa', 'tuhan'),
    ('yang agung', 'tuhan'),
    ('yang maha bijaksana', 'tuhan'),
    ('maha bijaksana', 'tuhan'),
    ('penguasa alam', 'tuhan'),
    ('pencipta', 'tuhan'),
    ('sang pencipta', 'tuhan'),
    ('sang penguasa', 'tuhan'),
    ('sang maha kuasa', 'tuhan'),
]

# === EXCLUSION RULES ===
EXCLUDE_KEYWORD_PAIRS = [('sulung', 'bungsu'), ('muda', 'tua'), ('mahkota', 'biasa')]
ORDINAL_PATTERN = re.compile(r'\bke(?:-?\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|belas|puluh|ratus|ribu)\b')

def normalize(text):
    return text.lower().strip()

def get_role(alias):
    alias = normalize(alias)
    for keyword, role in ROLE_KEYWORDS:
        if alias == keyword:
            return role
    words = alias.split()
    for keyword, role in ROLE_KEYWORDS:
        if keyword in words:
            return role
    return None

def has_exclusion_conflict(aliases_i, aliases_j):
    for a in aliases_i:
        for b in aliases_j:
            for x, y in EXCLUDE_KEYWORD_PAIRS:
                if x in a and y in b or x in b and y in a:
                    return True
    return False

def contains_ordinal(alias):
    return bool(ORDINAL_PATTERN.search(normalize(alias)))

# === MAIN FUNCTION ===
def apply_role_based_merging(df_cluster_input: pd.DataFrame) -> pd.DataFrame:
    df = df_cluster_input.copy()
    df["aliases"] = df["aliases"].apply(lambda x: [normalize(a) for a in x])

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row['story_id']].append({
            'person': row['person'],
            'aliases': row['aliases']
        })

    final_results = []
    for story_id, clusters in grouped.items():
        merged = []
        visited = [False] * len(clusters)

        for i in range(len(clusters)):
            if visited[i]:
                continue

            current_aliases = set(clusters[i]['aliases'])
            merged_indices = [i]
            roles_i = {get_role(a) for a in current_aliases if get_role(a)}

            for j in range(i + 1, len(clusters)):
                if visited[j]:
                    continue

                other_aliases = set(clusters[j]['aliases'])
                roles_j = {get_role(a) for a in other_aliases if get_role(a)}

                if any(contains_ordinal(a) for a in current_aliases | other_aliases):
                    continue

                if roles_i and roles_i == roles_j and not has_exclusion_conflict(current_aliases, other_aliases):
                    current_aliases |= other_aliases
                    visited[j] = True
                    merged_indices.append(j)

            merged.append({
                "aliases": list(current_aliases),
                "role": next(iter(roles_i)) if roles_i else None
            })
            for idx in merged_indices:
                visited[idx] = True

        # Handle 'orang tua' alias
        orang_tua_alias = 'orang tua'
        orang_tua_cluster_idx = None
        for idx, group in enumerate(merged):
            if any('orang tua' in alias for alias in group['aliases']):
                orang_tua_cluster_idx = idx
                break

        if orang_tua_cluster_idx is not None:
            orang_tua_aliases = [alias for alias in merged[orang_tua_cluster_idx]['aliases'] if 'orang tua' in alias]

            for idx, group in enumerate(merged):
                if idx != orang_tua_cluster_idx and group['role'] in ['ayah', 'ibu']:
                    group['aliases'].extend(orang_tua_aliases)
                    group['aliases'] = list(set(group['aliases']))

            if all('orang tua' in alias for alias in merged[orang_tua_cluster_idx]['aliases']):
                merged.pop(orang_tua_cluster_idx)

        for idx, group in enumerate(merged, 1):
            final_results.append({
                "story_id": story_id,
                "person": f"Tokoh-{idx}",
                "aliases": sorted(set(group['aliases'])),
                "role": group['role']
            })

    return pd.DataFrame(final_results)
