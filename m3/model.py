import sys
sys.path.append('.')
import torch
from torch import nn
from torch.nn import functional as F

class ConfirmShape(nn.Module):
    def __init__(self,
                 dl,
                 train_window,
                 predict_window,
                 batch_size=32,
                 backoffset=0):
        super().__init__()
        for batch in dl:
            break
        self.enc_shape = batch['encode_features'].shape
        self.dec_shape = batch['decode_features'].shape
        self.batch_size = batch_size
        self.predict_window = predict_window

class TemporalEncoder(ConfirmShape):
    def __init__(self,
                 dl,
                 train_window=200,
                 predict_window=16,
                 backoffset=0,
                 hidden_size=128,
                 batch_size=32,
                 encoder_layers=1,
                 bidirectional=False,
                 encoder_name='lstm'):
        super().__init__(dl=dl,
                         train_window=train_window,
                         predict_window=predict_window,
                         backoffset=backoffset,
                         batch_size=batch_size)
        encode_params = {'input_size':self.enc_shape[-1],
                         'hidden_size':hidden_size,
                         'bias':True}
        encoder_dict = {'lstm':nn.LSTMCell(**encode_params),
                        'rnn':nn.RNNCell(**encode_params),
                        'gru':nn.GRUCell(**encode_params)}
        
        cell = encoder_dict[encoder_name]
        self.encoders = {}
        for i in range(encoder_layers):
            layer_name = f'encoder_layer_{i}'
            self.add_module(layer_name,cell)
            self.encoders[layer_name] = cell

        self.bidirectional = bidirectional
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.predict_window = predict_window
    
    def forward(self,batch):
        x_enc = batch['encode_features']
        x_len = batch['x_len']
        self.x_len = x_len
        self.x_shape = x_enc.shape
        
        if self.bidirectional:
            h = []
            x_forward = x_enc
            x_backward = torch.flip(x_enc,dims=(1,))
            index_forward = torch.where(x_len==0,torch.zeros_like(x_len),x_len-1)
            index_backward = torch.where(x_len==self.x_shape[1],torch.zeros_like(index_forward),index_forward+1)
            h_forward = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
            h_backward = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
            if self.encoder_name == 'lstm':
                c = []
                c_forward = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
                c_backward = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
                for layer in range(self.encoder_layers):
                    hf,hb = [],[]
                    cf,cb = [],[]
                    h_forwards,h_backwards = [],[]
                    c_forwards,c_backwards = [],[]
                    cell = self.encoders[f'encoder_layer_{layer}']
                    for i,(x_f,x_b,h_f,c_f,h_b,c_b) in enumerate(zip(x_forward,x_backward,h_forward,c_forward,h_backward,c_backward)):
                        h_for,c_for = cell(x_f,(h_f,c_f))
                        h_back,c_back = cell(x_b,(h_b,c_b))
                        h_forwards.append(h_for),c_forwards.append(c_for)
                        h_backwards.append(h_back),c_backwards.append(c_back)
                        h_for,h_back = h_for[index_forward[i]],h_back[index_backward[i]]
                        c_for,c_back = c_for[index_forward[i]],c_back[index_backward[i]]
                        hf.append(h_for),hb.append(h_back)
                        cf.append(c_for),cb.append(c_back)
                    
                    h_forward,c_forward = torch.stack(h_forwards),torch.stack(c_forwards)
                    h_backward,c_backward = torch.stack(h_backwards),torch.stack(c_backwards)
                        
                    hf,hb = torch.stack(hf),torch.stack(hb)
                    cf,cb = torch.stack(cf),torch.stack(cb)
                    h.append(torch.stack([hb,hf]))
                    c.append(torch.stack([cb,cf]))
                h,c = torch.cat(h),torch.cat(c)
                return h,c
            else:
                for layer in range(self.encoder_layers):
                    hf,hb = [],[]
                    h_forwards,h_backwards = [],[]
                    cell = self.encoders[f'encoder_layer_{layer}']
                    for i,(x_f,x_b,h_f,h_b) in enumerate(zip(x_forward,x_backward,h_forward,h_backward)):
                        h_for = cell(x_f,h_f)
                        h_back = cell(x_b,h_b)
                        h_forwards.append(h_for),h_backwards.append(h_back)
                        h_for = h_for[index_forward[i]]
                        h_back = h_back[index_backward[i]]
                        hf.append(h_for),hb.append(h_back)
                    h_forward,h_backward = torch.stack(h_forwards),torch.stack(h_backwards)
                    hf,hb = torch.stack(hf),torch.stack(hb)
                    h.append(torch.stack([hb,hf]))
                h = torch.cat(h)
                return h
        else:
            hs = []
            index = torch.where(x_len==0,torch.zeros_like(x_len),x_len-1)  
            h = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
            if self.encoder_name == 'lstm':
                cs = []
                c = torch.zeros([self.x_shape[0],self.x_shape[1],self.hidden_size]).cuda()
                for layer in range(self.encoder_layers):
                    hh,cc,h_temporal,c_temporal = [],[],[],[]
                    cell = self.encoders[f'encoder_layer_{layer}']
                    for i,(x_f,h_,c_) in enumerate(zip(x_enc,h,c)):
                        h_tmp,c_tmp = cell(x_f,(h_,c_))
                        hh.append(h_tmp),cc.append(c_tmp)
                        h_,c_ = h_tmp[index[i]],c_tmp[index[i]]
                        h_temporal.append(h_),c_temporal.append(c_)
                    h,c = torch.stack(hh),torch.stack(cc)
                    hs.append(torch.stack(h_temporal))
                    cs.append(torch.stack(c_temporal))
                h,c = torch.stack(hs),torch.stack(cs)
                return h,c
            else:
                for layer in range(self.encoder_layers):
                    hh,h_temporal = [],[]
                    cell = self.encoders[f'encoder_layer_{layer}']
                    for x,h_,ind in zip(x_enc,h,index):
                        h_tmp = cell(x,h_)
                        hh.append(h_tmp)
                        h_ = h_tmp[ind]
                        h_temporal.append(h_)
                    h = torch.stack(hh)
                    hs.append(torch.stack(h_temporal))
                h = torch.stack(hs)
                return h

class TemporalDecoder(TemporalEncoder):
    def __init__(self,
                 dl,
                 train_window=200,
                 predict_window=16,
                 backoffset=0,
                 hidden_size=128,
                 batch_size=32,
                 encoder_layers=1,
                 decoder_layers=1,
                 dropout_rate=0.5,
                 bidirectional=False,
                 encoder_name='lstm',
                 decoder_name='lstm',
                 preprocess_encoder_states=False):
        super().__init__(dl=dl,
                         train_window=train_window,
                         predict_window=predict_window,
                         backoffset=backoffset,
                         hidden_size=hidden_size,
                         batch_size=batch_size,
                         encoder_layers=encoder_layers,
                         bidirectional=bidirectional,
                         encoder_name=encoder_name)
        decoder_params = {
                          'hidden_size':hidden_size,
                          'bias':True}
        decoder_dict = {'lstm':lambda layer:nn.LSTMCell(input_size=self.dec_shape[-1]+1 if layer==0 else hidden_size,
                                                         **decoder_params),
                        'rnn':lambda layer:nn.RNNCell(input_size=self.dec_shape[-1]+1 if layer==0 else hidden_size,
                                                      **decoder_params),
                        'gru':lambda layer:nn.GRUCell(input_size=self.dec_shape[-1]+1 if layer==0 else hidden_size,
                                                      **decoder_params)}
        
        self.decoders = {}
        decoder = decoder_dict[decoder_name]
        for i in range(decoder_layers):
            layer_name = f'decoder_layer_{i}'
            cell = decoder(i)
            self.add_module(layer_name,cell)
            self.decoders[layer_name] = cell

        self.decoder_layers = decoder_layers
        self.dropout_rate = dropout_rate
        self.decoder_name = decoder_name
        self.preprocess_encoder_states = preprocess_encoder_states
        
    def _fill_layers(self,h_states,c_states,diff_layers):
        if not isinstance(h_states,list):
            h_states = [h_state for h_state in h_states]
            c_states = [c_state for c_state in c_states]
            
        fill_layers = [torch.zeros_like(h_states) for _ in range(diff_layers)]
        h_states += fill_layers
        c_states += fill_layers
        return h_states,c_states
            
    def _clean_states(self,h_states,c_states,diff_layeres,mode='v1'):
        assert mode in ['v1','v2']
        h_states = [F.dropout(state,self.dropout_rate) for state in h_states]
        if c_states.sum() > 0:
            c_states = [F.dropout(state,self.dropout_rate) for state in c_states]
        c_states = [c_state for c_state in c_states]
        if mode == 'v2':
            h_states,c_states = self._fill_layers(h_states,c_states,diff_layeres)
        return h_states,c_states
        
    def forward(self,batch):
        if self.encoder_name == 'lstm':
            h_encoder,c_encoder = super().forward(batch)
        else:
            h_encoder = super().forward(batch)
            c_encoder = torch.zeros_like(h_encoder)
        
        encoder_layers = self.encoder_layers * 2 if self.bidirectional else self.encoder_layers
        decoder_layers = self.decoder_layers * 2 if self.bidirectional else self.decoder_layers
        diff_layers = decoder_layers - encoder_layers
        if encoder_layers >= decoder_layers:
            start_layer = encoder_layers - decoder_layers
            h_encoder = h_encoder[start_layer:]
            if c_encoder is not None:
                c_encoder = c_encoder[start_layer:]
            if self.preprocess_encoder_states:
                h_states,c_states = self._clean_states(h_encoder,c_encoder,diff_layers,mode='v1')
            else:
                h_states,c_states = self._fill_layers(h_encoder,c_encoder,diff_layers)
        else:
            if self.preprocess_encoder_states:
                h_states,c_states = self._clean_states(h_encoder,c_encoder,diff_layers,mode='v2')
            else:
                h_states,c_states = self._fill_layers(h_encoder,c_encoder,diff_layers)
        return h_states,c_states

class TemporalSeqToSeq(TemporalDecoder):
    def __init__(self,
                 dl,
                 train_window=200,
                 predict_window=16,
                 backoffset=0,
                 hidden_size=128,
                 batch_size=32,
                 encoder_layers=1,
                 decoder_layers=1,
                 dropout_rate=0.5,
                 bidirectional=False,
                 encoder_name='lstm',
                 decoder_name='lstm',
                 preprocess_encoder_states=False):
        super().__init__(dl=dl,
                         train_window=train_window,
                         predict_window=predict_window,
                         backoffset=backoffset,
                         hidden_size=hidden_size,
                         batch_size=batch_size,
                         encoder_layers=encoder_layers,
                         decoder_layers=decoder_layers,
                         bidirectional=bidirectional,
                         dropout_rate=dropout_rate,
                         encoder_name=encoder_name,
                         decoder_name=decoder_name,
                         preprocess_encoder_states=preprocess_encoder_states)
        self.fc1 = nn.Linear(hidden_size,128)
        self.fc2 = nn.Linear(128,1)
    
    def forward(self,batch):
        h_states,c_states = super().forward(batch)
        preds = []
        x_dec = batch['decode_features']
        x = batch['x']
        row_index = torch.arange(x_dec.shape[0]).cuda()
        index = row_index * self.x_shape[1] + self.x_len - 1
        index = torch.where(index<0,torch.zeros_like(index),index)
        prev_output = torch.index_select(x.flatten(),index=index,dim=0)[:,None]
        x_dec = torch.transpose(x_dec,1,0)
        for time in range(self.predict_window):
            h,c = [],[]
            inp = torch.cat([x_dec[time].float(),prev_output.float()],axis=1)
            for i,decoder in enumerate(self.decoders.values()):
                if self.decoder_name == 'lstm':
                    inp,c_state = decoder(inp.float(),(h_states[i],c_states[i]))
                    c.append(c_state)
                else:
                    inp = decoder(inp.float(),h_states[i])
                h.append(inp)
            h_states,c_states = h,c
            prev_output = self.fc2(self.fc1(inp))
            preds.append(prev_output)
        preds = torch.cat(preds,axis=1)
        preds = (preds * batch['std'][:,None]) + batch['mean'][:,None] 
        
        return preds
                
                
            

#%%
