import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, config):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.config = config

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.randn(*shape, device = self.config.device), requires_grad = True)
            init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.randn(length, device = self.config.device), requires_grad = True)
            init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class GCRUCell(nn.Module):
    def __init__(self, config, rnn_num_units, input_dim):
        super().__init__()
        self.config = config

        self.max_diffusion_step = self.config.graph_diffusion_step
        self._num_units = rnn_num_units
        self.input_dim = input_dim

        self.weight = nn.Parameter(torch.randn(2*self.config.cheb_k * (self._num_units + self.input_dim),
                                                      self._num_units + self.input_dim, device = self.config.device), requires_grad = True) 
        init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.randn(self._num_units + self.input_dim, device = self.config.device), requires_grad = True)
        init.constant_(self.bias, val = 0)
        self._gconv_params = LayerParams(rnn_network = self, layer_type= 'GCRUCell', config=self.config) # 인식이 안됨


        #self.gconv_weight = nn.Parameter(torch.randn((self.max_diffusion_step +1) * (self.config.V + self._num_units), self._num_units * 2,
        #                                             device = config.device), requires_grad = True)
        #self.gconv_bias = nn.Parameter(torch.randn(self._num_units * 2))
        self.to(config.device)
    
    @staticmethod
    def _concat(x, x_):
        # x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=2)
    
    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.config.V, self._num_units), device = self.config.device)

    def supports_cal(self, supports, input, weights, bias):

        support_set = []
        x_g = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(self.config.device), support]

            for k in range(2, self.config.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)  # (V,V)

        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, input))  # (B, V, hidden + input_dim = 32 + 1)
            
        x_g = torch.cat(x_g, dim=-1) # (B, V, 2 * cheb_k * dim_in)
        x_gconv = torch.einsum('bni,io->bno', x_g, weights)  + bias  # (B, V, hidden_dim)

        return x_gconv

    def gconv(self, inputs, supports, state, output_size, bias_start=0.0): # input 이랑 adjmx 랑 합치는과정임
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        
        x0 = x
        
        x1 = self.supports_cal(supports, inputs_and_state, self.weight, self.bias)
        x = self._concat(inputs_and_state, x1)

        for k in range(2, self.max_diffusion_step + 1):

            x2 = 2 * self.supports_cal(supports, x1.reshape(batch_size, self.config.V, self._num_units + self.input_dim), self.weight, self.bias) - x0
            x = self._concat(x, x2)
            x1, x0 = x2, x1
        
        num_matrices = self.max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self.config.V, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self.config.V, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self.config.V, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        return torch.reshape(x, [batch_size, self.config.V, output_size])
    
    def forward(self, inputs, hx, adj):
        output_size = 2 * self._num_units

        value = torch.sigmoid(self.gconv(inputs, adj, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.config.V, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        c = self.gconv(inputs, adj, r * hx, self._num_units)

        new_state = u * hx + (1.0 - u) * c

        return new_state

class Encoder(nn.Module):
    def __init__(self, config, rnn_num_units, input_dim):
        super().__init__()
        
        self.config = config
        self.rnn_num_units = rnn_num_units
        self.input_dim = input_dim

        self.cells = nn.ModuleList()
        self.cells.append(GCRUCell(self.config, self.rnn_num_units, self.input_dim))
        self.norm = nn.LayerNorm([self.config.V, self.rnn_num_units])

        for _ in range(self.config.num_layers):
            self.cells.append(GCRUCell(self.config, self.rnn_num_units, self.rnn_num_units))
        
        self.to(config.device)

    def init_hiddens(self, batch_size):
        init_states = []
        for i in range(self.config.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size))
        return init_states

    def forward(self, input_, init_hidden, supports):
        seq_length = input_.size(1)
        current_inputs = input_
        output_hidden = []

        for i in range(self.config.num_layers):
            state = init_hidden[i]
            inner_states = []
            # print('encoder start {} GRUCell layers'.format(i))
            for t in range(seq_length):
                state = self.cells[i](current_inputs[:,t,:,:], state, supports)
                state = self.norm(state.reshape(input_.size(0), self.config.V, self.rnn_num_units))
                inner_states.append(state)

            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim = 1)

        return current_inputs, output_hidden

class Decoder(nn.Module):
    def __init__(self, config, rnn_num_units, input_dim):
        super().__init__()

        self.config = config
        self.rnn_num_units = rnn_num_units
        self.input_dim = input_dim
        self.cells = nn.ModuleList()
        self.cells.append(GCRUCell(self.config, rnn_num_units, input_dim))
        self.norm = nn.LayerNorm([config.V, rnn_num_units])
        for _ in range(self.config.num_layers):
            self.cells.append(GCRUCell(self.config, rnn_num_units, rnn_num_units))
        self.to(config.device)
    
    def forward(self, input_, encoder_hidden, supports):
        current_inputs = input_
        output_hidden = []

        for i in range(self.config.num_layers):
            state = self.cells[i](current_inputs, encoder_hidden[i], supports)
            state = self.norm(state)
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, output_hidden

class GraphNet(nn.Module):
    def __init__(self, config, rnn_num_units, input_dim, y_cov_dim, out_dim,
                  cl_decay_steps = 2000, use_curriculum_learning=True):
        super().__init__()
        self.config = config
        self.rnn_num_units = rnn_num_units
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.y_cov_dim = y_cov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        self.encoder = Encoder(self.config, self.rnn_num_units, self.input_dim)
        self.decoder = Decoder(self.config, self.rnn_num_units * 2, self.input_dim + self.y_cov_dim)

        self.proj = nn.Sequential(nn.Linear(self.rnn_num_units* 2, self.out_dim, bias=True))

        self.We1 = nn.Parameter(torch.randn(self.config.V, self.rnn_num_units, device = self.config.device), requires_grad = True) 
        self.We2 = nn.Parameter(torch.randn(self.config.V, self.rnn_num_units, device = self.config.device), requires_grad = True) 
        self.Memory = nn.Parameter(torch.randn(self.rnn_num_units, self.rnn_num_units, device = self.config.device), requires_grad = True)
        self.Wq = nn.Parameter(torch.randn(self.rnn_num_units, self.rnn_num_units, device = self.config.device), requires_grad=True)
        # self.x_proj = nn.Conv2d(input_dim + y_cov_dim, input_dim, (input_dim, input_dim))

        init.xavier_normal_(self.We1)
        init.xavier_normal_(self.We2)
        init.xavier_normal_(self.Memory)
        init.xavier_normal_(self.Wq)
        self.to(config.device)
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.Wq)     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.Memory.t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.Memory)     # (B, N, d)
        # _, ind = torch.topk(att_score, k=2, dim=-1)

        return value, query
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + torch.exp(torch.tensor(batches_seen / self.cl_decay_steps)))

    def forward(self, history, xt, labels = None, training= True, batches_seen=None):
        # adj
        node_embeddings1 = torch.matmul(self.We1, self.Memory)
        node_embeddings2 = torch.matmul(self.We2, self.Memory)
        a1 = torch.softmax(torch.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        a2 = torch.softmax(torch.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [a1, a2]

        # encoder
        # x = torch.cat((history, xt), dim=3)
        # x = self.x_proj(x.transpose(1,3)).transpose(1,3).to(history.device)
        
        init_state = self.encoder.init_hiddens(history.size(0))
        h_en, state_en = self.encoder(history, init_state, supports)

        h_t = h_en[:, -1, :, :]
        
        # attention
        h_att, query = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
        
        # decoder
        ht_list = [h_t] * self.config.num_layers
        go = torch.zeros((history.shape[0], self.config.V, 1), device=history.device)
        out = []
        for t in range(self.config.T_h):
            
            h_de, ht_list = self.decoder(torch.cat([go, xt[:, t,:,:]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
            if training and self.use_curriculum_learning:
                c = torch.rand(1)
                if c < self.compute_sampling_threshold(batches_seen):
                   go = labels[:, t, :,:]
        output = torch.stack(out, dim=1)
        
        return output

