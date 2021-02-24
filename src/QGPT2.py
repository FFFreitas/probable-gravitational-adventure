import pytorch_lightning as pl
import torch
from torch import nn as nn
import pennylane as qml

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, num_qubits, num_qc_layers):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        dev = qml.device("default.qubit", wires=num_qubits, shots=1, analytic=True)
        @qml.qnode(dev, interface='torch')
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        weight_shapes = {"weights": (num_qc_layers, num_qubits)}
        q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qmlp = nn.Sequential(
            nn.Linear(512,num_qubits),
            q_layer,
            nn.Linear(num_qubits,512)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a 
        #m = self.mlp(self.ln_2(x))
        a2 = self.qmlp(self.ln_2(x).reshape(x.shape[0],-1))
        a2 = a2.reshape(x.shape)
        x = x + a2
        return x



class GPT2(pl.LightningModule):
    
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        layers: int,
        num_positions: int,
        vocab_size: int,
        num_classes: int,
        num_qubits: int, 
        num_qc_layers: int
    ):
        super(GPT2, self).__init__()
        self.save_hyperparameters()

        self._init_sos_token()
        self._init_embeddings()
        self._init_layers()

    def _init_sos_token(self):
        self.sos = torch.nn.Parameter(torch.zeros(self.hparams.embed_dim))
        nn.init.normal_(self.sos)

    def _init_embeddings(self):
        self.token_embeddings = nn.Embedding(self.hparams.vocab_size, self.hparams.embed_dim)
        self.position_embeddings = nn.Embedding(self.hparams.num_positions, self.hparams.embed_dim)

    def _init_layers(self):
        self.layers = nn.ModuleList()
        for _ in range(self.hparams.layers):
            self.layers.append(Block(self.hparams.embed_dim, self.hparams.heads, self.hparams.num_qubits, self.hparams.num_qc_layers))

        self.ln_f = nn.LayerNorm(self.hparams.embed_dim)
        self.head = nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size, bias=False)
        self.clf_head = nn.Linear(self.hparams.embed_dim, self.hparams.num_classes)

    def forward(self, x, classify=False):
        
        length, batch = x.shape

        h = self.token_embeddings(x.long())

        # prepend sos token
        sos = torch.ones(1, batch, self.hparams.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        
        if not classify:
            # return logits
            return self.head(h)
            
        h = self.ln_f(h)
        logits = self.head(h)

        h = torch.mean(h, dim=0)  # average pool over sequence
        return self.clf_head(h), logits  # return classification logits, modeifiquei aqui