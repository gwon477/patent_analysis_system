import re

def split_claims_to_chunks(claims, max_chunk_length=300):
    sentences = re.split(r'(?<=[.!?]) +', claims.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chunk_length:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks 