from torchtext import data


field = data.Field(tokenize='spacy', lower=True, batch_first=True)

fields = [
    ('query', field),
    ('response', field),
]

dataset = data.TabularDataset(
    path='data/processed/data.csv',
    format='csv',
    fields=fields
)

field.build_vocab(dataset)

train_iter = data.BucketIterator(
    dataset,
    batch_size=32,
    repeat=False,
    device=-1
)

batch = next(iter(train_iter))
x, y = batch.query, batch.response

query = []
for i in x[0, :].numpy():
    word = field.vocab.itos[i]
    if word != '<pad>':
        query.append(word)

response = []
for i in y[0, :].numpy():
    word = field.vocab.itos[i]
    if word != '<pad>':
        response.append(word)

print('\nQ: {}'.format(' '.join(query)))
print('R: {}'.format(' '.join(response)))
