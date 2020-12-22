import pickle
import gzip
import random

file = 'data/phoenix14t.pami0.train.annotations_only'

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

samples = []

tmp = load_dataset_file(file)

print('Successfully loaded file')
print('Reading file...')

for s in tmp:
    glosses = s['gloss'].split(' ')
    random.shuffle(glosses)
    glosses = ' '.join(glosses)
    print(glosses)
    samples.append({k: v for k, v in s.items() if k != 'sign'})

file = gzip.GzipFile(f'{file}.shuffled_glosses', 'wb')
file.write(pickle.dumps(samples, protocol=3))
file.close()

print('Successfully wrote new file')