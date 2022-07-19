import torch
import numpy as np
from torch import nn
from collections import OrderedDict

class ConfirmShape(nn.Module):
    def __init__(self,
                 reader,
                 batch_size,
                 build_inp,
                 use_bilstm=False,
                 num_decode_steps=16):
        super().__init__()
        batch_gen = reader.train_batch_generator(batch_size=batch_size)
        for batch in batch_gen:
            break
        build_inp(batch,use_bilstm)
        self.encode_shape = build_inp.encode_features.shape
        self.decode_shape = build_inp.decode_features.shape
        self.num_decode_steps = num_decode_steps
    
    def init_encoder_weights(self,x):
        x.weight.data.normal_(mean=0.0,std=1 / self.encode_shape[-1])
    
    def init_decoder_weights(self,x):
        x.weight.data.normal_(mean=0.0,std=1 / self.decode_shape[-1])
        
class TCNEncoder(ConfirmShape):
    def __init__(self,
                 reader,
                 batch_size,
                 build_inp,
                 use_bilstm=False,
                 receiptive_field=9,
                 residual_channel=32,
                 num_decode_steps=16,
                 filter_widths=[2 for i in range(8)],
                 dilations=[2**i for i in range(8)]):
        super().__init__(reader=reader,
                         batch_size=batch_size,
                         build_inp=build_inp,
                         use_bilstm=use_bilstm,
                         num_decode_steps=num_decode_steps)
        self.h_encode_ln = nn.Sequential(OrderedDict([('l1',nn.Linear(in_features=self.encode_shape[-1],
                                                                      out_features=residual_channel)),
                                                      ('t1',nn.Tanh())]))
        self.c_encode_ln = nn.Sequential(OrderedDict([('l1',nn.Linear(in_features=self.encode_shape[-1],
                                                                      out_features=residual_channel)),
                                                      ('t1',nn.Tanh())]))
        self.init_encoder_weights(self.h_encode_ln.l1)
        self.init_encoder_weights(self.c_encode_ln.l1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.residual_channel = residual_channel
        self.receiptive_field = receiptive_field
        
        self.dilated_encode_names = {}
        for i,(dilation,filter_width) in enumerate(zip(dilations,filter_widths)):
            layer_name = f'dilated_encoder_layer_{i}'
            conv = nn.Conv1d(in_channels=residual_channel,
                             out_channels=4*residual_channel,
                             kernel_size=filter_width,
                             dilation=dilation,
                             padding='same')
            conv.weight.data.normal_(mean=0,std=1 / (filter_width*self.encode_shape[-1])**0.5)
            self.dilated_encode_names[layer_name] = conv
            self.add_module(layer_name,conv)
    
    def forward(self,x):
        h = self.h_encode_ln(x)
        c = self.c_encode_ln(x)

        self.conv_inputs = [h]
        for i,name in enumerate(self.dilated_encode_names):
            dilated_conv = self.dilated_encode_names[name]
            dilation = self.dilations[i]
            filter_width = self.filter_widths[i]
            
            causal_shift = (filter_width - 1) + int((dilation - 1) / 2)
            padding = torch.zeros([h.shape[0],causal_shift,h.shape[2]]).cuda()
            h = torch.cat([h,padding],dim=1)

            h = h.transpose(1,2)
            h = dilated_conv(h)
            h = h.transpose(1,2)
            h = h[:,:-causal_shift,:]
            input_gate,conv_filter,conv_gate,emit_gate = torch.split(h,self.residual_channel,dim=2)
            c = self.sigmoid(input_gate) * c + self.sigmoid(conv_gate) * self.tanh(conv_filter)
            h = self.sigmoid(emit_gate) * self.tanh(c)
            
            self.conv_inputs.append(h)
        self.conv_inputs = self.conv_inputs[:-1]

class InitilizeDecodeShapes(TCNEncoder):
    def __init__(self,
             reader,
             batch_size,
             build_inp,
             receiptive_field,
             use_bilstm=False,
             residual_channel=32,
             num_decode_steps=16,
             filter_widths=[2 for i in range(8)],
             dilations=[2**i for i in range(8)]):
        super().__init__(reader=reader,
                         receiptive_field=receiptive_field,
                         batch_size=batch_size,
                         build_inp=build_inp,
                         use_bilstm=use_bilstm,
                         filter_widths=filter_widths,
                         dilations=dilations,
                         residual_channel=residual_channel,
                         num_decode_steps=num_decode_steps)
        self.h_decode_ln = nn.Sequential(OrderedDict([('l1',nn.Linear(in_features=self.decode_shape[-1]+1,
                                                                      out_features=residual_channel)),
                                                      ('t1',nn.Tanh())]))
        self.c_decode_ln = nn.Sequential(OrderedDict([('l1',nn.Linear(in_features=self.decode_shape[-1]+1,
                                                                      out_features=residual_channel)),
                                                      ('t1',nn.Tanh())]))
        self.init_decoder_weights(self.h_decode_ln.l1)
        self.init_decoder_weights(self.c_decode_ln.l1)
        self.relu = nn.ReLU()
        final_input_size = len(filter_widths) * residual_channel
        self.final_decode_input = nn.Sequential(OrderedDict(fc1=nn.Linear(in_features=final_input_size,
                                                                          out_features=128),
                                                            act1=self.relu))
        self.final_decode_output = nn.Linear(in_features=128,
                                             out_features=2)
        self.init_decoder_weights(self.final_decode_input.fc1)
        self.init_decoder_weights(self.final_decode_output)
        
        self.decode_convs = {}
        self.dilate_decode_names = []
        for i,(dilation,filter_width) in enumerate(zip(dilations,filter_widths)):
            layer_name = f'dilated_decoder_layer_{i}'
            self.dilate_decode_names.append(layer_name)
            conv = nn.Conv1d(in_channels=residual_channel,
                             out_channels=4*residual_channel,
                             kernel_size=filter_width,
                             dilation=dilation,
                             padding='same'
                             )
            conv.weight.data.normal_(mean=0,std=1 / (filter_width*self.decode_shape[-1])**0.5)
            self.decode_convs[layer_name] = conv
            self.add_module(layer_name,conv)
    
    def forward(self,x_enc,x_dec,x_centered,batch):
        super().forward(x=x_enc)
        x_len = torch.from_numpy(batch['x_len']).cuda() - 1
        col_idx=  torch.where(x_len<0,torch.ones_like(x_len),x_len)[:,None].long()
        x = torch.gather(x_centered,index=col_idx,dim=1)
        x_init = torch.stack([x for _ in range(self.decode_shape[1])],dim=1)
        x_dec = torch.cat([x_init,x_dec],dim=2)
        
        self.conv_weights,self.conv_bias = {},{}
        h = self.h_decode_ln(x_dec)
        c = self.c_decode_ln(x_dec)
        self.conv_weights['decode_fc_h'] = self.h_decode_ln.l1.weight
        self.conv_bias['decode_fc_h'] = self.h_decode_ln.l1.bias
        self.conv_weights['decode_fc_c'] = self.c_decode_ln.l1.weight
        self.conv_bias['decode_fc_c'] = self.c_decode_ln.l1.bias
        
        decode_outputs = []
        for i,dilate_name in enumerate(self.dilate_decode_names):
            dilation = self.dilations[i]
            filter_width = self.filter_widths[i]
            causal_shift = (filter_width - 1) + int((dilation - 1) / 2)
            padding = torch.zeros([h.shape[0],causal_shift,h.shape[2]]).cuda()
            h = torch.cat([padding,h],axis=1)
            
            dilated_conv = self.decode_convs[dilate_name]
            h = h.transpose(1,2)
            h = dilated_conv(h)
            h = h.transpose(1,2)

            h = h[:,:-causal_shift,:]
            
            input_gate,conv_filter,conv_gate,emit_gate = torch.split(h,self.residual_channel,dim=2)
            c = self.sigmoid(input_gate) * c + self.sigmoid(conv_gate) * self.tanh(conv_filter)
            h = self.sigmoid(emit_gate) * self.tanh(c)
            
            self.conv_weights[dilate_name] = dilated_conv.weight
            self.conv_bias[dilate_name] = dilated_conv.bias
            decode_outputs.append(h)
        
        decode_outputs = torch.cat(decode_outputs,dim=2)
        h = self.final_decode_input(decode_outputs)
        y = self.final_decode_output(h)
        self.conv_weights['decode_input'] = self.final_decode_input.fc1.weight
        self.conv_bias['decode_input'] = self.final_decode_input.fc1.bias
        self.conv_weights['decode_output'] = self.final_decode_output.weight
        self.conv_bias['decode_output'] = self.final_decode_output.bias
        
        return y

class TCNDecoder(InitilizeDecodeShapes):
    def __init__(self,
                 reader,
                 batch_size,
                 receiptive_field,
                 build_inp,
                 use_bilstm=False,
                 residual_channel=32,
                 num_decode_steps=16,
                 filter_widths=[2 for i in range(8)],
                 dilations=[2**i for i in range(8)]): 
        super().__init__(reader=reader,
                         receiptive_field=receiptive_field,
                         batch_size=batch_size,
                         build_inp=build_inp,
                         use_bilstm=use_bilstm,
                         residual_channel=residual_channel,
                         num_decode_steps=num_decode_steps,
                         filter_widths=filter_widths,
                         dilations=dilations)
    
    def forward(self,x_enc,x_dec,x_centered,batch):
        super().forward(x_enc=x_enc,
                        x_dec=x_dec,
                        x_centered=x_centered,
                        batch=batch)
        
        state_queues = []
        y_len = batch['y_len']
        x_len = torch.from_numpy(batch['x_len']).cuda()
        for conv_input,dilation in zip(self.conv_inputs,self.dilations):
            conv_shape = conv_input.shape
            batch_idx = torch.arange(self.decode_shape[0])[None]
            flatten_row_idx = torch.repeat_interleave(batch_idx[None],dilation,dim=0).flatten().cuda()[:,None]
            col_idx = (x_len[:,None] + torch.arange(dilation).cuda()).long().flatten()[:,None]
            index = torch.cat([flatten_row_idx,col_idx],axis=1)
            
            conv_input = conv_input.reshape(conv_shape[0]*conv_shape[1],-1)
            padding = torch.zeros([dilation,conv_shape[-1]]).cuda()
            conv_input = torch.cat([padding,conv_input],dim=0)
            index = index[:,0] * conv_shape[1] + index[:,1]
            slices = torch.index_select(conv_input,index=index,dim=0)
            slices = slices.view(conv_shape[0],dilation,conv_shape[2])
            slices = slices.permute(1,0,2)
            state_queues.append(slices)
        
        col_idx = x_len - 1
        col_idx = torch.where(col_idx<0,torch.zeros_like(x_len),x_len).cuda().long()[:,None]
        x_init = torch.gather(x_centered,dim=1,index=col_idx)
        
        x_dec = x_dec.permute(1,0,2)
        emit_ta = []
        elements_finished = 0 >= y_len
        time = 0

        while not np.all(elements_finished):
            feat = x_dec[time]
            current_input = torch.cat([feat,x_init],axis=1)
            
            w_fc_h,w_fc_c = self.conv_weights['decode_fc_h'],self.conv_weights['decode_fc_c']
            b_fc_h,b_fc_c = self.conv_bias['decode_fc_h'],self.conv_bias['decode_fc_c']
            
            h = self.tanh(torch.matmul(current_input,torch.transpose(w_fc_h,1,0)) + b_fc_h)
            c = self.tanh(torch.matmul(current_input,torch.transpose(w_fc_c,1,0)) + b_fc_c)
            
            skip_outputs = []
            for i,(state_queue,dilation,dilated_name) in enumerate(zip(state_queues,self.dilations,self.dilate_decode_names)):
                state = state_queue[time]
                
                w_conv = self.conv_weights[dilated_name]
                bias = self.conv_bias[dilated_name]
                state_conv = torch.transpose(w_conv[:,:,0],1,0)
                h_conv = torch.transpose(w_conv[:,:,1],1,0)
                dilated_conv = torch.matmul(state,state_conv) + torch.matmul(h,h_conv) + bias
                
                input_gate,conv_filter,forget_gate,emit_gate = torch.split(dilated_conv,self.residual_channel,dim=1)
                c = self.sigmoid(input_gate) * c + self.sigmoid(forget_gate) + self.tanh(conv_filter)
                h = self.sigmoid(emit_gate) * self.tanh(c)

                new_queue = torch.cat([state_queue,h[None]])
                state_queues[i] = new_queue
                skip_outputs.append(h)
            
            skip_outputs = torch.cat(skip_outputs,axis=1)
            w_h,w_y = self.conv_weights['decode_input'],self.conv_weights['decode_output']
            b_h,b_y = self.conv_bias['decode_input'],self.conv_bias['decode_output']
            h = self.relu(torch.matmul(skip_outputs,torch.transpose(w_h,1,0)) + b_h)
            y = torch.matmul(h,torch.transpose(w_y,1,0)) + b_y
            
            elements_finished = time >= y_len
            finished = np.all(elements_finished)
            next_element_finished = time >= self.num_decode_steps - 1
            next_input = y if not finished else torch.zeros_like(y)
            
            emit = torch.where(torch.from_numpy(elements_finished[:,None]).cuda(),torch.zeros_like(next_input),next_input)
            emit_ta.append(emit)
            elements_finished = elements_finished | next_element_finished
            time += 1
        
        return emit_ta

class TCN(TCNDecoder):
    def __init__(self,
                 reader,
                 batch_size,
                 receiptive_field,
                 build_inp,
                 use_bilstm=False,
                 residual_channel=32,
                 num_decode_steps=16,
                 filter_widths=[2 for i in range(8)],
                 dilations=[2**i for i in range(8)]): 
        super().__init__(reader=reader,
                         receiptive_field=receiptive_field,
                         batch_size=batch_size,
                         build_inp=build_inp,
                         use_bilstm=use_bilstm,
                         residual_channel=residual_channel,
                         num_decode_steps=num_decode_steps,
                         filter_widths=filter_widths,
                         dilations=dilations)
    
    def forward(self,x_enc,x_dec,x_centered,batch,x_mean):
        preds = []
        y_hats = super().forward(x_enc=x_enc,
                                 x_dec=x_dec,
                                 x_centered=x_centered,
                                 batch=batch)
        for y in y_hats:
            p,y_hat = torch.split(y,1,dim=-1)
            y = self.sigmoid(p) * (y_hat + x_mean)
            preds.append(y)
        preds = torch.cat(preds,axis=1)
        return preds
        

#%%
