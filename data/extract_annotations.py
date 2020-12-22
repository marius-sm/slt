import pickle
import gzip

# This script allows to generate new lighter files without the embeddings
# The original file with embeddings contains
# a list [{'name': ..., 'text': ..., 'gloss': ..., 'signer': ..., 'sign': ...}]
# The new file contains
# a list [{'name': ..., 'text': ..., 'gloss': ..., 'signer': ...}]
# where the 'sign' attribute (the embedding) has been removed.

file = 'data/phoenix14t.pami0.train'

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

samples = []

tmp = load_dataset_file(file)

print('Successfully loaded file')
print('Reading file...')

for s in tmp:
    samples.append({k: v for k, v in s.items() if k != 'sign'})

file = gzip.GzipFile(f'{file}.annotations_only', 'wb')
file.write(pickle.dumps(samples, protocol=3))
file.close()

print('Successfully wrote new file')