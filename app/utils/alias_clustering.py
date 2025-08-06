# utils/alias_clustering.py
import textdistance

POINTERS = [
    'ibu', 'pak', 'puteri', 'permaisuri', 'raja', 'putera', 'ayah', 'istri', 'suami', 'uwak', 'menteri',
    'bunda', 'anak', 'kakak', 'adik', 'kakek', 'orang tua', 'tetangga', 'putri', 'beru tandang',
    'putroe', 'telangkai', 'tuhan', 'abang'
]

EXCLUDE_PAIRS = [
    ('bungsu', 'sulung'),
    ('muda', 'tua'),
    ('mahkota', 'biasa')
]

FALSE_MERGE = set([
    ('raja', 'rajawali'),
    ('putri', 'putri malu'),
    ('raja', 'rajagaluh')
])

def normalize(text):
    return text.lower().strip()

def compute_similarity(s1, s2):
    return {
        'jaccard': textdistance.jaccard(s1, s2),
        'jaro': textdistance.jaro(s1, s2),
    }

def is_excluded(name1, name2):
    name1 = normalize(name1)
    name2 = normalize(name2)
    for a, b in EXCLUDE_PAIRS:
        if a in name1 and b in name2 or b in name1 and a in name2:
            return True
    return False

def cluster_without_pointers(characters_list, aliases_clusters, threshold):
    cluster_id = len(aliases_clusters) + 1
    for character in characters_list:
        character = normalize(character)
        found = False
        for key, cluster in aliases_clusters.items():
            for name in cluster:
                name = normalize(name)
                if len(character) < 4 or len(name) < 4:
                    if character != name:
                        if f" {character} " in f" {name} " or f" {name} " in f" {character} ":
                            cluster.add(character)
                            found = True
                            break
                        continue
                if character.endswith("nya") and name.endswith("nya"):
                    if compute_similarity(character[:-3], name[:-3])['jaro'] >= threshold:
                        cluster.add(character)
                        found = True
                        break
                    else:
                        break
                if f" {character} " in f" {name} " or f" {name} " in f" {character} ":
                    cluster.add(character)
                    found = True
                    break
                if (character, name) in FALSE_MERGE or (name, character) in FALSE_MERGE:
                    continue
                if compute_similarity(character, name)['jaro'] >= threshold:
                    if character in name or name in character:
                        continue
                    cluster.add(character)
                    found = True
                    break
            if found:
                break
        if not found:
            aliases_clusters[f"person-{cluster_id}"] = {character}
            cluster_id += 1
    return aliases_clusters

def cluster_with_pointers(characters_with_pointer, pointers, threshold):
    all_pointer_item_clusters = []
    for p in pointers:
        pointer_cluster = {}
        pointer_cluster_id = 1
        one_token_list = []
        character_per_pointer = [char for char in characters_with_pointer if normalize(char).startswith(p)]
        for character in character_per_pointer:
            character = normalize(character)
            tokens = character.split()
            if len(tokens) == 1:
                one_token_list.append(character)
            else:
                found = False
                for cluster in pointer_cluster.values():
                    for name in cluster:
                        if is_excluded(character, name):
                            continue
                        suffix_char = character[len(p):].strip()
                        suffix_name = name[len(p):].strip()
                        if len(suffix_char) < 4 or len(suffix_name) < 4:
                            if suffix_char != suffix_name:
                                continue
                        if suffix_char in suffix_name or compute_similarity(suffix_char, suffix_name)['jaccard'] >= threshold:
                            cluster.append(character)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    pointer_cluster[pointer_cluster_id] = [character]
                    pointer_cluster_id += 1
        if one_token_list:
            if len(pointer_cluster) == 1:
                for item in one_token_list:
                    pointer_cluster[1].append(item)
            elif not pointer_cluster:
                pointer_cluster[pointer_cluster_id] = one_token_list
        all_pointer_item_clusters.extend(pointer_cluster.values())
    return all_pointer_item_clusters

def merge_clusters(aliases_clusters, pointer_clusters):
    final_clusters = {}
    counter = 1
    for cluster in pointer_clusters:
        final_clusters[f"Tokoh-{counter}"] = list(set(cluster))
        counter += 1
    for key, value in aliases_clusters.items():
        final_clusters[f"Tokoh-{counter}"] = list(set(value))
        counter += 1
    return final_clusters

def cluster_character_aliases(characters_list):
    aliases_clusters = {}
    characters_with_pointer = [c for c in characters_list if any(normalize(c).startswith(p) for p in POINTERS)]
    characters_without_pointer = [c for c in characters_list if c not in characters_with_pointer]
    aliases_clusters = cluster_without_pointers(characters_without_pointer, aliases_clusters, 0.82)
    pointer_clusters = cluster_with_pointers(characters_with_pointer, POINTERS, 0.75)
    return merge_clusters(aliases_clusters, pointer_clusters)
