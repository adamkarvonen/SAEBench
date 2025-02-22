from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cuda', cache_dir='/share/u/can/models')

with model.session() as session:
    with session.iter([0, 1, 2, 3]) as (batch, iterator):
        pass