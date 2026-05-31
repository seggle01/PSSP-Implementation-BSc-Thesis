"""
Encoder Transformer for Protein Secondary Structure Prediction

This module implements an Encoder-Only Transformer architecture for predicting
protein secondary structures from Protein Language Model (PLM) embeddings.
Furthermore, an Ensemble class is implemented that combines multiple Encoder-Only
models and averages their probabilities for the final output.

Author: Stefanos Englezou
Last Modified: 28/05/2026
"""
import torch
import torch.nn as nn
from math import sqrt, log
from config import Config

#########  Module-level configuration #########
cfg       = Config()
device    = cfg["DEVICE"]
embed_dim = cfg['ESM2']['embed_dim']

######### Base-model factory #########
def build_base_model(params):
    """
    Instantiate a PSSPEncoder on the configured device from a hyperparameter
    dictionary.

    Parameters
    ----------
    params : dict
        Hyperparameter dictionary that MUST contain the following keys:
        'embed_dim', 'tokens', 'd_ff_multiplier', 'num_heads', 'layers',
        'mlp_multiplier', 'dropout', and 'num_classes'.

    Returns
    -------
    PSSPEncoder
        A freshly initialised PSSPEncoder model moved to the global `device`.
    """
    return PSSPEncoder(
        embed_dim       = params['embed_dim'],
        tokens          = params['tokens'],
        d_ff_multiplier = params['d_ff_multiplier'],
        num_heads       = params['num_heads'],
        layers          = params['layers'],
        mlp_multiplier  = params['mlp_multiplier'],
        dropout         = params['dropout'],
        num_classes     = params['num_classes']
    ).to(device)

######### PSSPEncoder — Encoder-only Transformer for per-residue PSSP #########
class PSSPEncoder(nn.Module):
    """
    The class of the encoder-transformer for protein secondary structure prediction.
    The model takes the embedding from the PLM of a position, slices it into a number
    of tokens, adds positional information to each token and through a number of layers
    it performs self-attention. After capturing intra-relationships between the slices,
    mean pooling is applied and forwarded to the final MLP head.


    Attributes
    ----------
    tokens : int
        The number of tokens the PLM embedding is sliced into.
    token_dim : int
        The dimension of each token (embed_dim // tokens).
    pos_e : PositionalEncoding
        Positional Embeddings Module.
    layers : nn.ModuleList
        List of Encoder Layer modules defined by a given number of blocks.
    mlp_head : nn.Sequential
        Feed-forward network with two linear layers and GELU activation function.
    """
    def __init__(self,
                 embed_dim: int       = 1280,
                 tokens: int          = 5,
                 d_ff_multiplier: int = 2,
                 num_heads: int       = 4,
                 layers: int          = 1,
                 mlp_multiplier: int  = 2,
                 num_classes: int     = 3,
                 dropout: float       = 0.1):
        """
        Constructor of PSSPEncoder.

        Parameters
        ----------
        embed_dim : int, default=1280
            The embedding dimension given by the Protein Language Model (PLM).
        tokens : int, default=5
            The number of equally sized tokens to slice the embedding vector.
        d_ff_multiplier : int, default=2
            The multipler to be applied to the token dimension for the MLP inside a Layer.
        num_heads : int, default=4
            The number of heads for the self-attention module.
        layers : int, default=1
            The number of layers in the Transformer.
        mlp_multiplier : int, default=2
            The multipler to be applied to the token dimension for the final MLP head.
        num_classes : int, default=3
            The number of classes to predict.
        dropout : float, default=0.1
            Value of dropout to be applied.
        """
        super().__init__()
        # Argument validation
        assert isinstance(embed_dim, int) and embed_dim > 0, "embed_dim must be a positive integer."
        assert isinstance(tokens, int) and tokens > 0, "tokens must be a positive integer."
        assert embed_dim % tokens == 0, "embed_dim must be divisible by tokens."
        assert isinstance(d_ff_multiplier, int) and d_ff_multiplier >= 1, "d_ff_multiplier must be an integer starting from 1."
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer."
        assert (embed_dim // tokens) % num_heads == 0, "token_dim (embed_dim // tokens) must be divisible by num_heads."
        assert isinstance(layers, int) and layers >= 1, "layers must be an integer starting from 1."
        assert isinstance(mlp_multiplier, int) and mlp_multiplier >= 1, "mlp_multiplier must be an integer starting from 1."
        assert isinstance(num_classes, int) and num_classes in [3, 8], "num_classes must be either 3 or 8"
        assert isinstance(dropout, float) and dropout >= 0.0 and dropout < 1.0, "dropout must be a float in the range [0.0, 1.0)."

        # Derived dimensions 
        self.tokens    = tokens
        self.token_dim = embed_dim // tokens
        d_ff           = d_ff_multiplier * self.token_dim
        mlp_dim        = mlp_multiplier  * self.token_dim

        # Components: positional encoding, stack, classification head 
        self.pos_e  = PositionalEncoding(self.token_dim, tokens, dropout)
        self.layers = nn.ModuleList([
            Layer(self.token_dim, d_ff, num_heads, dropout) for _ in range(layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.Linear(self.token_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through PSSPEncoder.

        This method takes the given PLM embedding of a certain residue,
        reshapes it into token slices, adds absolute positional embeddings,
        then forwards through each layer to self-attend between the
        embedding slices. After that the mean is taken across the token
        dimension and passed into the final MLP head for classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, embed_dim)
            with dtype torch.float32.

        Returns
        -------
        tuple of (torch.Tensor, list)
            logits : torch.Tensor
                Output tensor of shape (batch_size, num_classes)
                containing the classification logits.
            encoder_attention_weights : list of torch.Tensor
                List of attention weight tensors, one per layer, each of
                shape (batch_size, num_heads, tokens, tokens).
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.dtype == torch.float32, "x must have dtype torch.float32."
        assert x.dim() == 2, "x must be a 2D tensor of shape (batch_size, embed_dim)."

        # Slice PLM embedding into [batch_size, tokens, token_dim] 
        x = x.reshape(x.shape[0], self.tokens, self.token_dim)

        # Add positional information 
        x = self.pos_e(x)

        # Encoder stack 
        encoder_attention_weights = []
        for layer in self.layers:
            x, attn_w = layer(x)
            encoder_attention_weights.append(attn_w)

        # Mean-pool over tokens, classify 
        logits = self.mlp_head(torch.mean(x, dim=1))

        return logits, encoder_attention_weights
    
######### Encoder block (multi-head self-attention + FFN, post-norm) #########
class Layer(nn.Module):
    """
    Transformer encoder layer.

    This class implements a single transformer encoder layer consisting
    of multi-head self-attention followed by a Feed Forward Network (FFN).

    Attributes
    ----------
    attention : MultiHeadAttention
        Multi-head self-attention mechanism.
    mlp : nn.Sequential
        Feed-forward network with two linear layers and ReLU activation function.
    layer_norm_1 : nn.LayerNorm
        Layer normalization applied after attention.
    layer_norm_2 : nn.LayerNorm
        Layer normalization applied after MLP.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        """
        Constructor of encoder layer.

        Parameters
        ----------
        d_model : int
            Dimension of the model embeddings.
        d_ff : int
            Dimension of the feed-forward network hidden layer.
        num_heads : int
            Number of attention heads in multi-head attention.
        dropout : float, default=0.1
            Dropout probability for regularization. Must be between 0 and 1.
        """
        super().__init__()
        # Argument validation 
        assert isinstance(d_model, int) and d_model > 0, "d_model must be a positive integer."
        assert isinstance(d_ff, int) and d_ff > 0, "d_ff must be a positive integer."
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer."
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        assert isinstance(dropout, float) and dropout >= 0.0 and dropout < 1.0, "dropout must be a float in the range [0.0, 1.0)."

        # Sub-modules 
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through encoder layer.

        This method applies the full encoder layer operations. It is
        important to note that Layer-Normalization is applied after each
        sub-layer (following post-norm approach).

        The flow follows:
        x -> MultiHeadAttention -> Dropout -> Add(x) -> LayerNorm -> MLP -> Dropout -> Add -> LayerNorm -> output

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
            with dtype torch.float32.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            x : torch.Tensor
                Output tensor of shape (batch_size, seq_len, d_model)
                after applying attention and MLP with residual connections.
            attn_weights : torch.Tensor
                Attention weight tensor of shape
                (batch_size, num_heads, seq_len, seq_len).
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.dtype == torch.float32, "x must have dtype torch.float32."
        assert x.dim() == 3, "x must be a 3D tensor of shape (batch_size, seq_len, d_model)."

        # Self-attention sub-layer
        attn_out, attn_weights = self.attention((x, x, x))
        x = self.layer_norm_1(x + self.dropout(attn_out))

        # Feed-forward sub-layer
        mlp_out = self.mlp(x)
        x = self.layer_norm_2(x + self.dropout(mlp_out))

        return x, attn_weights

######### Multi-Head Self-Attention #########
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This class implements the multi-head attention mechanism from
    "Attention is All You Need" (Vaswani et al., 2017). The input
    is projected into query, key, and value representations, which
    are then split into multiple heads. Scaled dot-product attention
    is computed in parallel for each head, and the results are
    concatenated and projected to produce the final output.

    The attention mechanism allows the model to focus on different
    positions and representation subspaces simultaneously, capturing
    complex relationships in the input data.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    d_k : int
        Dimension of each attention head (d_model / num_heads).
    sqrt_d_k : float
        Square root of d_k used for scaling attention scores.
    w_q : nn.Linear
        Linear projection for query.
    w_k : nn.Linear
        Linear projection for key.
    w_v : nn.Linear
        Linear projection for value.
    w_o : nn.Linear
        Output projection after concatenating heads.
    """
    def __init__(self, d_model: int, num_heads: int):
        """
        Constructor of multi-head attention module.

        This constructor creates the linear projection layers for
        query, key, value, and output. The model dimension is split
        across multiple heads, with each head having dimension d_k.
        All projections are initialized without bias terms.

        Parameters
        ----------
        d_model : int
            Dimension of the model embeddings. Must be a positive
            integer divisible by num_heads.
        num_heads : int
            Number of attention heads. Must be a positive integer and
            d_model must be divisible by num_heads to ensure each head
            has integer dimension.
        """
        super().__init__()
        # Argument validation 
        assert isinstance(d_model, int) and d_model > 0, "d_model must be a positive integer."
        assert isinstance(num_heads, int) and num_heads > 0, "num_heads must be a positive integer."
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.sqrt_d_k  = sqrt(self.d_k)

        # Query / Key / Value / Output projections (bias-free)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Perform scaled dot-product attention.

        This method computes attention weights by taking the dot product
        of queries and keys, scaling by the square root of the key dimension,
        computing softmax, and finally multiplying by values.

        The attention formula is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Parameters
        ----------
        Q : torch.Tensor
            Query tensor of shape (batch_size, num_heads, tgt_len, d_k)
            with dtype torch.float32.
        K : torch.Tensor
            Key tensor of shape (batch_size, num_heads, src_len, d_k)
            with dtype torch.float32.
        V : torch.Tensor
            Value tensor of shape (batch_size, num_heads, src_len, d_k)
            with dtype torch.float32.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            attention_out : torch.Tensor
                Attention output of shape (batch_size, num_heads, tgt_len, d_k)
                representing the weighted combination of values based on
                attention weights.
            attention_weights : torch.Tensor
                Attention weight matrix of shape
                (batch_size, num_heads, tgt_len, src_len).
        """
        assert isinstance(Q, torch.Tensor), "Q must be a torch.Tensor."
        assert isinstance(K, torch.Tensor), "K must be a torch.Tensor."
        assert isinstance(V, torch.Tensor), "V must be a torch.Tensor."
        assert Q.dtype == torch.float32, "Q must have dtype torch.float32."
        assert K.dtype == torch.float32, "K must have dtype torch.float32."
        assert V.dtype == torch.float32, "V must have dtype torch.float32."
        assert Q.dim() == 4, "Q must be a 4D tensor of shape (batch_size, num_heads, tgt_len, d_k)."
        assert K.dim() == 4, "K must be a 4D tensor of shape (batch_size, num_heads, src_len, d_k)."
        assert V.dim() == 4, "V must be a 4D tensor of shape (batch_size, num_heads, src_len, d_k)."

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d_k  # [batch_size, num_heads, tgt_len, src_len]

        # Row-wise softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum over values
        attention_out = torch.matmul(attention_weights, V)

        return attention_out, attention_weights

    def forward(self, QKV: tuple):
        """
        Forward pass through multi-head attention.

        This method applies multi-head attention by:
        1. Projecting inputs to query, key, value representations
        2. Splitting projections into multiple heads
        3. Computing scaled dot-product attention for each head
        4. Concatenating head outputs
        5. Applying final output projection

        For self-attention, all three inputs (query, key, value) are
        typically the same tensor.

        Parameters
        ----------
        QKV : tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            Tuple of three tensors (query, key, value), each with
            dtype torch.float32. Query has shape
            (batch_size, tgt_len, d_model), key and value have shape
            (batch_size, src_len, d_model). For self-attention, all
            three are identical.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            output : torch.Tensor
                Output tensor of shape (batch_size, tgt_len, d_model)
                after applying multi-head attention and output projection.
            attention_weights : torch.Tensor
                Attention weight matrix of shape
                (batch_size, num_heads, tgt_len, src_len).
        """
        # Argument validation
        assert isinstance(QKV, tuple) and len(QKV) == 3, "QKV must be a tuple of three tensors (query, key, value)."
        query, key, value = QKV
        assert isinstance(query, torch.Tensor), "query must be a torch.Tensor."
        assert isinstance(key, torch.Tensor), "key must be a torch.Tensor."
        assert isinstance(value, torch.Tensor), "value must be a torch.Tensor."
        assert query.dtype == torch.float32, "query must have dtype torch.float32."
        assert key.dtype == torch.float32, "key must have dtype torch.float32."
        assert value.dtype == torch.float32, "value must have dtype torch.float32."
        assert query.dim() == 3, "query must be a 3D tensor of shape (batch_size, tgt_len, d_model)."
        assert key.dim() == 3, "key must be a 3D tensor of shape (batch_size, src_len, d_model)."
        assert value.dim() == 3, "value must be a 3D tensor of shape (batch_size, src_len, d_model)."

        batch_size, tgt_len, d_model = query.shape
        src_len = key.shape[1]

        # Project + split heads → [batch_size, num_heads, seq_len, d_k] 
        Q = self.w_q(query).view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key  ).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention 
        attention_out, attention_weights = self.scaled_dot_product_attention(Q, K, V)  # [batch_size, num_heads, tgt_len, d_k]

        # Concatenate heads → [batch_size, tgt_len, d_model]
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, d_model)

        # Output projection
        return self.w_o(attention_out), attention_weights

######### Sinusoidal Positional Encoding (Vaswani et al., 2017) #########
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    This class implements fixed sinusoidal positional encodings as
    described in "Attention is All You Need" (Vaswani et al., 2017).
    The encodings are added to input embeddings to provide absolute
    position information to the transformer.

    Attributes
    ----------
    pe : torch.Tensor
        Sinusoidal positional encoding buffer of shape
        (1, max_len, d_model). Registered as a non-trainable buffer.
    dropout : nn.Dropout
        Dropout layer applied after adding positional encodings.
    """
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        """
        Constructor of sinusoidal positional encoding.

        This constructor precomputes the sinusoidal positional encoding
        matrix and registers it as a buffer. The encoding uses sine
        for even dimensions and cosine for odd dimensions, with
        frequencies decreasing geometrically across dimensions.

        Parameters
        ----------
        d_model : int
            Dimension of the model embeddings. Positional encodings
            have the same dimension and are added element-wise.
        max_len : int
            Maximum sequence length supported. Determines the number
            of precomputed positional encoding vectors.
        dropout : float, default=0.1
            Dropout probability applied after adding positional
            encodings. Must be between 0 and 1.
        """
        super().__init__()
        # Argument validation
        assert isinstance(d_model, int) and d_model > 0, "d_model must be a positive integer."
        assert isinstance(max_len, int) and max_len > 0, "max_len must be a positive integer."
        assert isinstance(dropout, float) and dropout >= 0.0 and dropout < 1.0, "dropout must be a float in the range [0.0, 1.0)."

        self.dropout = nn.Dropout(p=dropout)

        # Precompute the sinusoidal table and register as buffer
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor):
        """
        Add positional encodings to input embeddings.

        This method adds the precomputed sinusoidal positional
        encodings to the input embeddings and applies dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch_size, seq_len, d_model)
            with dtype torch.float32.

        Returns
        -------
        torch.Tensor
            Embeddings with positional information added, of shape
            (batch_size, seq_len, d_model), after dropout.
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.dtype == torch.float32, "x must have dtype torch.float32."
        assert x.dim() == 3, "x must be a 3D tensor of shape (batch_size, seq_len, d_model)."

        # Add positional information and apply dropout
        return self.dropout(x + self.pe[:, :x.size(1), :])

######### Seed Ensemble (averages logits across independently trained models) #########
class EnsemblePSSP(nn.Module):
    """
    Ensemble of PSSPEncoder models for protein secondary structure prediction.

    This class combines multiple PSSPEncoder models and averages their output
    probabilities to produce a final prediction. Each model is independently
    loaded from a saved state dictionary and set to evaluation mode with frozen
    parameters. The ensemble approach reduces variance and improves robustness
    compared to a single model.

    Attributes
    ----------
    ensemble : nn.ModuleList
        List of PSSPEncoder models loaded from the provided state dictionaries.
    """
    def __init__(self, state: str, state_dict_paths: list[str]):
        """
        Constructor of EnsemblePSSP.

        This constructor instantiates a set of PSSPEncoder models using the
        hyperparameters defined in the configuration for the given Q value.
        Each model is then loaded with its corresponding state dictionary,
        set to evaluation mode, and its parameters are frozen to prevent
        any gradient updates during inference.

        Parameters
        ----------
        state : str
            Selector that picks which hyperparameter block to load from the
            configuration file. Must be one of ``"3class"`` or ``"8class"``;
            the constructor looks up ``MODEL_CONFIG[f"{state}_params"]`` to
            obtain ``tokens``, ``d_ff_multiplier``, ``num_heads``, ``layers``,
            ``mlp_multiplier``, ``dropout`` and ``num_classes`` for every
            member of the ensemble.
        state_dict_paths : list of str
            List of file paths to the saved state dictionaries, one per
            model in the ensemble. The number of models in the ensemble
            equals the length of this list.
        """
        super().__init__()
        # Argument validation
        assert isinstance(state, str), "state must be str"
        assert isinstance(state_dict_paths, list) and len(state_dict_paths) > 0, "state_dict_paths must be a non-empty list."
        assert all(isinstance(p, str) for p in state_dict_paths), "All entries in state_dict_paths must be strings."

        # Retrieve hyperparameters from configuration
        params = dict(cfg["MODEL_CONFIG"][f"{state}_params"])
        params.setdefault('embed_dim', embed_dim)

        # Instantiate one PSSPEncoder per checkpoint
        self.ensemble = nn.ModuleList(
            PSSPEncoder(
                embed_dim       = params['embed_dim'],
                tokens          = params['tokens'],
                d_ff_multiplier = params['d_ff_multiplier'],
                num_heads       = params['num_heads'],
                layers          = params['layers'],
                mlp_multiplier  = params['mlp_multiplier'],
                dropout         = params['dropout'],
                num_classes     = params['num_classes']
            )
            for _ in range(len(state_dict_paths))
        )

        # Load each checkpoint, set eval mode, freeze parameters
        for model, path in zip(self.ensemble, state_dict_paths):
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the ensemble.

        This method runs inference through each PSSPEncoder model in the
        ensemble independently and averages their output logits to produce
        a single prediction. All models are evaluated under torch.no_grad()
        since their parameters are frozen.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, embed_dim)
            with dtype torch.float32.

        Returns
        -------
        torch.Tensor
            Averaged logits tensor of shape (batch_size, num_classes)
            representing the mean prediction across all ensemble models.
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor."
        assert x.dtype == torch.float32, "x must have dtype torch.float32."
        assert x.dim() == 2, "x must be a 2D tensor of shape (batch_size, embed_dim)."

        # Collect per-model logits
        outputs = []
        with torch.no_grad():
            for model in self.ensemble:
                outputs.append(model(x)[0])  # [B, num_classes]

        # Average across the ensemble dimension
        return torch.stack(outputs, dim=0).mean(dim=0)  # [num_models, B, C] -> [B, C]
