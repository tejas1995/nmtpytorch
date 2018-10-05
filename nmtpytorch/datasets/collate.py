# -*- coding: utf-8 -*-

# This will eventually disappear as this only provides .size
# which can be inferred if we guarantee that batch_dim is always at
# a given position regardless of input/output feature/tensor types.


class Batch(dict):
    """A custom dictionary representing a batch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = next(iter(self.values())).size(1)

    def device(self, device):
        self.update({k: v.to(device) for k, v in self.items()})

    def __repr__(self):
        s = "Batch(size={})\n".format(self.size)
        for data_source, tensor in self.items():
            s += "  {:10s} -> {} - {}\n".format(
                str(data_source), tensor.shape, tensor.device)
        return s


def get_collate(data_sources):
    """Returns a special collate_fn which will view the underlying data
    in terms of the given DataSource keys."""

    def collate_fn(batch):
        return Batch(
            {ds: ds.to_torch([elem[ds] for elem in batch]) for ds in data_sources},
        )

    return collate_fn
