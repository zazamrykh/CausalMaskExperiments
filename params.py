import enum


class PosEncodeTypes(enum.Enum):  # Should be removed
    learnable_zeros = 0
    sincos = 1
    sincos_learnable = 2
    rot_pos = 3


class GPTConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    pos_enc_dropout = 0.1
    num_heads = 12
    num_blocks = 12
    embed_dim = 768
    pos_encode_type = PosEncodeTypes.learnable_zeros
    max_len = 256

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"attn_dropout={self.attn_dropout}, embed_dropout={self.embed_dropout}, ff_dropout={self.ff_dropout}, " \
               f"pos_enc_dropout={self.pos_enc_dropout}, num_heads={self.num_heads}, " \
               f"num_blocks={self.num_blocks}, embed_dim={self.embed_dim}, " \
               f"pos_encode_type={self.pos_encode_type}, max_len={self.max_len}"


class ModelTypes(enum.Enum):
    RNN = 0
    Transformer = 1
    LLaMA_masked = 2
    LLaMA_maskless = 3

    @staticmethod
    def get_LLaMA_types():
        return [ModelTypes.LLaMA_masked, ModelTypes.LLaMA_maskless]

    def __str__(self):
        return self.name


class DatasetTypes(enum.Enum):
    whole = 0
    small = 1
    tiny = 2
    half_seq_len = 3  # 128 sequence_len
    half_tiny = 4  # 128 seq len and tiny


class Params:  # Should be changed
    @staticmethod
    def get(print_params=True):
        params = Params()
        if print_params:
            print(params)
        return params

    def __init__(self, model_type=ModelTypes.Transformer, batch_size=64, learning_rate=0.001, n_epoch=25,
                 weight_decay=1e-5,
                 scheduler=True, scheduler_step=7, scheduler_gamma=0.25, warmup_steps=100, offset=64,
                 dataset_type=DatasetTypes.whole, gpt_config=None):
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.warmup_steps = warmup_steps
        self.offset = offset
        self.dataset_type = dataset_type
        self.GPTConfig = gpt_config

    def __str__(self):
        output = f"model_type={self.model_type}, batch_size={self.batch_size}, lr={self.learning_rate}, " \
                 f"n_epoch={self.n_epoch}, weight_decay={self.weight_decay}, " \
                 f"scheduler={self.scheduler}, scheduler_step={self.scheduler_step}, " \
                 f"scheduler_gamma={self.scheduler_gamma}, offset={self.offset}, dataset_type={self.dataset_type}"
        if self.GPTConfig is None:
            return output
        else:
            return output + '\nGPT config:' + str(self.GPTConfig)
