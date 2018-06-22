import torch
mw = torch.load('data/models/mnemonic.mdl', map_location=lambda storage, loc: storage)
dw = torch.load('data/reader/multitask.mdl', map_location=lambda storage, loc: storage)
mw['word_dict'] = dw['word_dict']
mw['state_dict']['embedding.weight'] = dw['state_dict']['embedding.weight']
mw['args'].vocab_size = dw['args'].vocab_size
torch.save(mw, 'data/models/mnemonic-emb.mdl')
rw = torch.load('data/models/rnet.mdl', map_location=lambda storage, loc: storage)
rw['word_dict'] = dw['word_dict']
rw['state_dict']['embedding.weight'] = dw['state_dict']['embedding.weight']
rw['args'].vocab_size = dw['args'].vocab_size
torch.save(rw, 'data/models/rnet-emb.mdl')
