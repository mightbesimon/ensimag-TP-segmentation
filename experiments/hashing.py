# identifies a model by what it actually is, not by the config numbers used
# to build it: a checkpoint is only safe to resume from if the current code
# would produce a state_dict with the exact same parameter names and shapes.
# this catches cases like adding a final head conv to HourGlass, where the
# enc/dec channel tuples stay the same but the architecture doesn't.

import hashlib


def hash_model(model):
	signature = sorted(
		(name, tuple(param.shape))
		for name, param in model.state_dict().items()
	)
	digest = hashlib.sha256(repr(signature).encode('utf-8')).hexdigest()
	return digest[:8]
