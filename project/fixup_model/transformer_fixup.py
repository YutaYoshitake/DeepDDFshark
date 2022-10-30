import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import CustomLayerNorm
from fixup_model.multi_head_atten import *





class TransformerEncoder_FixUp(nn.Module):

    def __init__(
        self, 
        d_model=256, 
        nhead=8, 
        dim_feedforward=1024, 
        dropout=0.1, 
        activation="relu", 
        encoder_layers = 5, 
        two_mha = False,  
        T_Fixup = False, 
        ):
        super().__init__()

        self.dropout = dropout
        self.layers = nn.ModuleList([])
        if two_mha:
            self.layers.extend([
                TransformerEncoderLayer_FixUp02(
                                encoder_embed_dim=d_model, 
                                encoder_attention_heads=nhead, 
                                attention_dropout=0.0, 
                                dropout=dropout, 
                                activation_dropout=0.1,  # 0.0, 
                                encoder_ffn_embed_dim=dim_feedforward, 
                                activation_fn=activation, 
                                use_norm=False, 
                                )
                for i in range(encoder_layers)]
            )
        else:
            self.layers.extend([
                TransformerEncoderLayer_FixUp(
                                encoder_embed_dim=d_model, 
                                encoder_attention_heads=nhead, 
                                attention_dropout=0.0, 
                                dropout=dropout, 
                                activation_dropout=0.1,  # 0.0, 
                                encoder_ffn_embed_dim=dim_feedforward, 
                                activation_fn=activation, 
                                use_norm=False, 
                                )
                for i in range(encoder_layers)]
            )

        print('##########   ENCODER   ##########')
        print(self.layers[0].fc1.weight[0, :5])
        # ======================== fixup calling initialization in layers ==================================
        if T_Fixup:
            for encoder_layer in self.layers:
                encoder_layer.fixup_initialization(en_layers=encoder_layers)
        # =================================================================
        print(self.layers[0].fc1.weight[0, :5])
        print('##########     END     ##########')

    def forward(self, src, atten_mask=None, seq_padding_mask=None, itr_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, atten_mask, seq_padding_mask, itr_padding_mask)
        return output





class TransformerDecoder_FixUp(nn.Module):

    def __init__(
        self, 
        d_model=256, 
        nhead=8, 
        dim_feedforward=1024, 
        dropout=0.1, 
        activation="relu", 
        decoder_layers = 5, 
        T_Fixup = False,  
        ):
        super().__init__()

        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer_FixUp(
                                add_bias_kv = False, 
                                add_zero_attn = False, 
                                decoder_embed_dim = d_model, 
                                cross_self_attention = False, 
                                decoder_attention_heads = nhead, 
                                attention_dropout = 0.0, 
                                dropout = dropout, 
                                activation_dropout = 0.1, 
                                activation_fn = activation, 
                                decoder_ffn_embed_dim = dim_feedforward, 
                                use_norm=False, 
                                )
            for _ in range(decoder_layers)]
        )

        print('##########   DECODER   ##########')
        print(self.layers[0].fc1.weight[0, :5])
        # ======================== fixup calling initialization in layers ==================================
        if T_Fixup:
            for decoder_layer in self.layers:
                decoder_layer.fixup_initialization(de_layers=decoder_layers)
        # =================================================================
        print(self.layers[0].fc1.weight[0, :5])
        print('##########     END     ##########')

    def forward(self, tgt, memory, atten_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, atten_mask)
        return output



def _get_activation_fn(activation):
    if activation == "relu": return F.relu
    elif activation == "gelu": return F.gelu



class TransformerEncoderLayer_FixUp(nn.Module):

    def fixup_initialization(self, en_layers):
        temp_state_dic = {}

        for name, param in self.named_parameters():
            if name in ["fc1.weight",
                        "fc2.weight",
                        "self_attn.out_proj.weight",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
                print(f'T-Fixup {name}')
            elif name in ["self_attn.v_proj.weight",]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))
                print(f'T-Fixup {name}')

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __init__(
        self, 
        encoder_embed_dim = 256, 
        encoder_attention_heads = 8, 
        attention_dropout = 0.0, 
        dropout = 0.5, # 0.1, 
        activation_dropout = 0.1, # 0.0, 
        encoder_ffn_embed_dim = 1024, 
        activation_fn = "relu", 
        use_norm=False, 
        ):
        super().__init__()
        attention_dropout = 0.0
        activation_dropout = 0.1

        self.embed_dim = encoder_embed_dim
        self.self_attn = MultiheadAttention__(
            self.embed_dim,
            encoder_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout
        self.fc1 = Linear__(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear__(encoder_ffn_embed_dim, self.embed_dim)
        # self.use_norm = use_norm
        # if self.use_norm:
        #     self.norm1 = CustomLayerNorm(self.embed_dim, variable_length='false')
        #     self.norm2 = CustomLayerNorm(self.embed_dim, variable_length='false')

    def forward(self, x, atten_mask=None, seq_padding_mask=None, itr_padding_mask=None):
        # x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        # output of shape `(seq_len, batch, embed_dim)`
        residual = x
        x, self.attn_output_weights = self.self_attn(query=x, key=x, value=x, key_padding_mask=None, attn_mask=atten_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # if self.use_norm:
        #     x = self.norm1(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # if self.use_norm:
        #     x = self.norm2(x)
        return x



class TransformerDecoderLayer_FixUp(nn.Module):

    def fixup_initialization(self, de_layers):
        temp_state_dic = {}

        # if args.Tfixup:
        for name, param in self.named_parameters():
            if name in ["fc1.weight",
                        "fc2.weight",
                        "self_attn.out_proj.weight",
                        "encoder_attn.out_proj.weight",
                        ]:
                temp_state_dic[name] = (9 * de_layers) ** (- 1. / 4.) * param
                print(f'T-Fixup {name}')
            elif name in ["self_attn.v_proj.weight","encoder_attn.v_proj.weight",]:
                temp_state_dic[name] = (9 * de_layers) ** (- 1. / 4.) * (param * (2**0.5))
                print(f'T-Fixup {name}')

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __init__(
        self, 
        add_bias_kv = False, 
        add_zero_attn = False, 
        decoder_embed_dim = 256, 
        cross_self_attention = False, 
        decoder_attention_heads = 8, 
        attention_dropout = 0.0, 
        dropout = 0.5, # 0.1, 
        activation_fn = "relu", 
        activation_dropout = 0.1, # 0.0, 
        decoder_ffn_embed_dim = 1024, 
        use_norm=False, 
        ):
        super().__init__()
        attention_dropout = 0.0
        activation_dropout = 0.1

        self.embed_dim = decoder_embed_dim
        self.cross_self_attention = cross_self_attention
        self.self_attn = MultiheadAttention__(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout

        self.encoder_attn = MultiheadAttention__(
            self.embed_dim, 
            decoder_attention_heads,
            kdim=None,
            vdim=None,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )

        self.fc1 = Linear__(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = Linear__(decoder_ffn_embed_dim, self.embed_dim)
        self.use_norm = use_norm
        # if self.use_norm:
        #     self.norm1 = CustomLayerNorm(self.embed_dim, variable_length='false_dec')
        #     self.norm2 = CustomLayerNorm(self.embed_dim, variable_length='false_dec')
        #     self.norm3 = CustomLayerNorm(self.embed_dim, variable_length='false_dec')

    def forward(self, x, encoder_out, atten_mask=None):
        residual = x
        y = x
        x, attn = self.self_attn(query=x, key=y, value=y, key_padding_mask=None, attn_mask=atten_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # if self.use_norm:
        #     x = self.norm1(x, optim_idx)

        residual = x
        x, attn = self.encoder_attn(query=x, key=encoder_out, value=encoder_out, key_padding_mask=None, attn_mask=None,) # 異なるビューでCrossAtten取る？
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # if self.use_norm:
        #     x = self.norm2(x, optim_idx)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # if self.use_norm:
        #     x = self.norm3(x, optim_idx)
        return x



class TransformerEncoderLayer_FixUp02(nn.Module):

    def fixup_initialization(self, en_layers):
        temp_state_dic = {}

        for name, param in self.named_parameters():
            if name in ["fc1.weight",
                        "fc2.weight",
                        "multi_view_attn.out_proj.weight", 
                        "self_attn.out_proj.weight", 
                        ]:
                temp_state_dic[name] = (0.57 * (en_layers) ** (- 1. / 4.)) * param
                print(f'T-Fixup {name}')
            elif name in ["self_attn.v_proj.weight", 
                        "multi_view_attn.v_proj.weight", 
                        ]:
                temp_state_dic[name] = (0.57 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))
                print(f'T-Fixup {name}')

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def __init__(
        self, 
        encoder_embed_dim = 256, 
        encoder_attention_heads = 8, 
        attention_dropout = 0.0, 
        dropout = 0.5, # 0.1, 
        activation_dropout = 0.1, # 0.0, 
        encoder_ffn_embed_dim = 1024, 
        activation_fn = "relu", 
        use_norm=False, 
        ):
        super().__init__()
        attention_dropout = 0.0
        activation_dropout = 0.1

        self.embed_dim = encoder_embed_dim
        self.multi_view_attn = MultiheadAttention__(
            self.embed_dim,
            encoder_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )
        self.self_attn = MultiheadAttention__(
            self.embed_dim,
            encoder_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout
        self.fc1 = Linear__(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear__(encoder_ffn_embed_dim, self.embed_dim)
        print(f'encoder_dropout : {self.dropout}')
        # self.use_norm = use_norm
        # if self.use_norm:
        #     self.norm1 = CustomLayerNorm(self.embed_dim, variable_length='false')
        #     self.norm2 = CustomLayerNorm(self.embed_dim, variable_length='false')

    def forward(self, x, atten_mask=None, seq_padding_mask=None, itr_padding_mask=None):
        # x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        # output of shape `(seq_len, batch, embed_dim)`

        itr, seq, batch, dim = x.shape
        ##############################
        x = x.permute(1, 0, 2, 3) # [seq, itr, batch, dim]
        x = x.reshape(seq, itr*batch, -1) # [seq, (itr_0(b_0, b_1, ...), itr_1(b_0, b_1, ...), ...), dim]
        residual = x
        x, self.multi_view_attn_output_weights = self.multi_view_attn(query=x, key=x, value=x, attn_mask=atten_mask, key_padding_mask=seq_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = (residual + x).reshape(seq, itr, batch, -1) # [seq, itr, batch, dim]
        ##############################
        x = x.permute(1, 0, 2, 3) # [itr, seq, batch, dim]
        x = x.reshape(itr, seq*batch, -1) # [itr, (seq_0(b_0, b_1, ...), seq_1(b_0, b_1, ...), ...), dim]
        residual = x
        x, self.attn_output_weights = self.self_attn(query=x, key=x, value=x, attn_mask=atten_mask, key_padding_mask=itr_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = (residual + x).reshape(itr, seq, batch, -1)
        ##############################
        # ((self.multi_view_attn(query=x_tes[:2, 0],  key=x_tes[:2, 0], value=x_tes[:2, 0])[0]-x[:2, :3])<1e-5).all()
        # (self.self_attn(query=x_tes[2:, 3],  key=x_tes[2:, 3], value=x_tes[2:, 3])[0] - x_[2:, 9:12, :] < 1e-5).all()
        ##############################
        # x_ = x.permute(1, 0, 2, 3).reshape(seq, itr*batch, -1)
        # residual = x_
        # x_, self.multi_view_attn_output_weights = self.multi_view_attn(query=x_, key=x_, value=x_, key_padding_mask=None, attn_mask=atten_mask,)
        # x_ = F.dropout(x_, p=self.dropout, training=self.training)
        # x_ = (residual + x_).reshape(seq, itr, batch, -1)
        ##############################
        # x_ = x_.permute(1, 0, 2, 3).reshape(itr, seq*batch, -1)
        # residual = x_
        # x_, self.attn_output_weights = self.self_attn(query=x_, key=x_, value=x_, key_padding_mask=None, attn_mask=atten_mask,)
        # x_ = F.dropout(x_, p=self.dropout, training=self.training)
        # x_ = (residual + x_).reshape(itr, seq, batch, -1)
        ##############################

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x



if __name__=='__main__':
    e, d = TransformerEncoder_FixUp(), TransformerDecoder_FixUp()
    inp = torch.randn(20, 32, 256)
    e(inp, True)