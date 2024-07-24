import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sympy as sp

class MM_DIT(nn.Module):                                #  X,Y,C=[batch,token,channel]
    def __init__(self,channel,mlp_ratio,num_head=8):
        super(MM_DIT,self).__init__()

        self.channel = channel

        self.mlp_ratio =mlp_ratio
        self.num_head = num_head


        self.layer_norm = torch.nn.LayerNorm(normalized_shape=channel,eps=1e-6,elementwise_affine=False)    #大小不会变化,经layernorm以后,对channel进行norm

    def adaLN_modulation(self,Y):    # from Y[B,T,C]-->silu[B,T,C]-->flatten[B,T*C]-->linear[B,6*T*C]-->6 parameter(αβγ)[B,T,C]


        batchsize_Y, token_Y, channel_Y = Y.shape


        Y_reshape = Y.reshape(batchsize_Y*token_Y,channel_Y)    #⭐output.reshape(B,C*T)
        out = F.silu(Y_reshape)
        linear_layer = nn.Linear(channel_Y,6*channel_Y).to(device)
        out = linear_layer(out)
        out = out.reshape(batchsize_Y, token_Y,6*channel_Y)

        alpha, beta, gama, alpha_mlp, beta_mlp, gama_mlp = torch.chunk(out,6,2)
        return alpha, beta, gama, alpha_mlp, beta_mlp, gama_mlp

    def MLP(self,X):    # from[B,T,C]-->linear[B,T*C*ratio]-->SILU-->linear[B,T,C]
        B,T,C = X.shape
        X = X.reshape(B*T,C)                       #⭐output.reshape(B,C*T)
        linear1 = nn.Linear(in_features=C,out_features=C*self.mlp_ratio).to(device)
        linear2 = nn.Linear(in_features=C*self.mlp_ratio, out_features=C).to(device)

        output = linear1(X)
        output = F.silu(output)
        output = linear2(output)
        output = output.reshape(B,T,C)
        return output

    def modulate(self,x, shift, scale):       #  MOD=αX+β
        return x * (1 + scale) + shift

    def RMS_norm(self,x,eps=1e-6):
        # 计算通道（channels）维度上的均值和均方根
        mean = x.mean(dim=(1, 2), keepdim=True)
        rms = (x ** 2).mean(dim=(1, 2), keepdim=True).sqrt() + eps
        # 归一化
        normalized_x = (x - mean) / rms
        return normalized_x

    def forward(self,Y,C,X):
        batchsize_Y, token_Y, channel_Y = Y.shape
        batchsize_C, token_C, channel_C = C.shape
        batchsize_X, token_X, channel_X = X.shape

        padding_needed = (3 - token_C % 3) % 3         #154-->156,need to divide 3
        padding = torch.zeros(batchsize_C, padding_needed, channel_C).to(device)
        C = torch.cat((C, padding), dim=1)
        X = torch.cat((X, padding), dim=1)
        padding_Y = torch.zeros(batchsize_Y,token_C-token_Y,channel_Y).to(device)
        Y = torch.cat((Y, padding_Y), dim=1)
        Y = torch.cat((Y, padding), dim=1)


        alpha_L, beta_L, gama_L, alpha_mlp_L, beta_mlp_L, gama_mlp_L = self.adaLN_modulation(Y)  #parameter fromY,left and right for different condition(text,photo)
        alpha_R, beta_R, gama_R, alpha_mlp_R, beta_mlp_R, gama_mlp_R = self.adaLN_modulation(Y)


        C_MOD = self.modulate(self.layer_norm(C), alpha_L, beta_L)  # C-->layernorm-->MOD
        X_MOD = self.modulate(self.layer_norm(X), alpha_R, beta_R)  # X-->layernorm-->MOD

        batch_cmod, token_cmod, channel_cmod = C_MOD.shape                #[B,T,C]-->[B,T*C]-->linear-->OUT[B,T,C]
        batch_xmod, token_xmod, channel_xmod = X_MOD.shape
        C_MOD = C_MOD.reshape(batch_cmod*token_cmod,-1)
        X_MOD = X_MOD.reshape(batch_xmod*token_xmod,-1)

        C_linear = nn.Linear(in_features=channel_cmod,out_features=channel_cmod).to(device)   #⭐output.reshape(B,C*T)
        X_linear = nn.Linear(in_features=channel_xmod, out_features=channel_xmod).to(device)

        multi_C = C_linear(C_MOD)
        multi_X = X_linear(X_MOD)

        multi_C = multi_C.reshape(batch_cmod,token_cmod,channel_cmod)
        multi_X = multi_X.reshape(batch_xmod,token_xmod,channel_xmod)


        Q_C,K_C,V_C = torch.chunk(multi_C,3,dim=1)         #[B,T,C]-->divide3-->3*[B,T/3,C]
        Q_X,K_X,V_X = torch.chunk(multi_X,3,dim=1)



        Q = torch.cat([self.RMS_norm(Q_C),self.RMS_norm(Q_X)],dim=1)
        K = torch.cat([self.RMS_norm(K_C),self.RMS_norm(K_X)],dim=1)
        V = torch.cat([V_C,V_X],dim=1)


        Q = Q.transpose(1, 0).to(device)
        K = K.transpose(1, 0).to(device)
        V = V.transpose(1, 0).to(device)


        Attentionlayer = nn.MultiheadAttention(embed_dim=Q.size(2), num_heads=self.num_head).to(device)
        output, attn_weight = Attentionlayer(Q, K, V)     # output from attention = [B,T/3,C]


        T,B,C = output.shape              # output[B,T/3,C]-->linear[B,T,C]-->out*gama+c[B,T,C]
        output = output.reshape(B*C,T)                     #⭐output.reshape(B,C*T)
        linear_layer = nn.Linear(in_features=T,out_features=token_cmod).to(device)
        output = linear_layer(output)
        output = output.reshape(B,token_cmod,C)
        output_L = C + gama_L * output
        output_R = C + gama_R * output



        C_MOD_2 = self.modulate(self.layer_norm(output_L), alpha_mlp_L, beta_mlp_L)  #-->layernorm-->mod-->mlp-->out=c+gama*out
        X_MOD_2 = self.modulate(self.layer_norm(output_R), alpha_mlp_R, beta_mlp_R)


        output_L = C + gama_mlp_L * self.MLP(C_MOD_2)
        output_R = C + gama_mlp_R * self.MLP(X_MOD_2)


        return output_L,output_R      #final input=output=[B,T,C]



class C(nn.Module):   #  [b,154,4096]
    def __init__(self,input_size):
        super(C, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=input_size)    #input_size=4096

    def forward(self,x):
        x = self.linear(x)
        return x

class X(nn.Module):
    def __init__(self,latent_size,latent_channel,patch_size,embed_dim):                  # after latent photo[B,latent_channel,latent_size,latent_size],patch_size are small photo size
        super(X,self).__init__()
        self.latent_size = latent_size      #1024*1024
        self.latent_channel = latent_channel  #1
        self.patch_size = patch_size      #16,
        self.embed_dim = embed_dim     #linear out_channel

        self.num_patch = (self.latent_size//self.patch_size)**2  #(1024/16)^2=64^2=4096,token =4096

        self.linear = nn.Linear(patch_size*patch_size*latent_channel,embed_dim)

        self.positional_embedding = self.generate_positional_embedding()

    def generate_positional_embedding(self):
        position_embedding = torch.zeros(1, self.num_patch, self.embed_dim)   #[B=1,H=4096,W=embed_dim]
        scale = 2 * math.pi / math.sqrt(self.embed_dim)
        for pos in range(self.num_patch):
            for i in range(self.embed_dim):

                position_embedding[0, pos, i] = math.sin(scale * (pos + 1) ** (2 * i / self.embed_dim))
        return position_embedding.to(device)

    def forward(self,latent_img):
        batch_size,latent_channel,img_H,img_W = latent_img.shape   #[B,C,H,W]=[B,1,1024,1024]

        latent_img = latent_img.view(batch_size, latent_channel, img_H // self.patch_size, self.patch_size,img_W // self.patch_size, self.patch_size)  #[B,1,64,16,64,16]

        latent_img = latent_img.permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, -1, latent_channel * self.patch_size * self.patch_size)   #[B,4096,1*16*16]

        x = self.linear(latent_img)
        x = x+self.positional_embedding
        x = x.view(batch_size,self.embed_dim,self.num_patch)

        return x    # [B,1,1024,1024]-->[B,4096(1024/16^2),256(C*16*16)]-->[B,4096,embed]=[B,token,channel]





class Y(nn.Module):  #timestep+sinusoidal+MLP   [B,T,channel]-->[B,T,channel]   77*2048
    def __init__(self,token,channel):
        super(Y,self).__init__()
        self.token = token
        self.time_step = self.token

        self.channel = channel


        self.mlp = nn.Sequential(nn.Linear(self.channel,self.channel//2),
                                 nn.ReLU(),
                                 nn.Linear(self.channel//2,self.channel)
                                 )                                                     #out_channel = 64*D

        self.time_encoding = self.generate_positional_encoding()


    def generate_positional_encoding(self):

        position_encoding = torch.zeros(self.time_step, self.channel)
        for pos in range(self.time_step):
            for i in range(0, self.channel,2):

                position_encoding[pos, i] = math.sin(pos / (10000 ** ((i // 2) / self.channel)))

                position_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i // 2) / self.channel)))
        return position_encoding.to(device).unsqueeze(0)

    def forward(self,x):    #x=[B,t,C]
        sinusoidal_encoding = self.time_encoding[:, :x.size(1), :].detach()
        y = self.mlp(sinusoidal_encoding)
        x = self.mlp(x)
        x = x + y
        x = self.mlp(x.view(x.size(0) * x.size(1), -1)).view(x.size(0), x.size(1), self.channel)
        return x


class ModulationBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModulationBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x1,x2):
        x = torch.cat((x1, x2), dim=1)
        return self.activation(self.linear(x))

class UpPatchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpPatchBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.conv_transpose(x)

class CompositeBlock(nn.Module):   #modulation+linear+unpatching[1,156,4096]-->[1,233,4096]-->[1,233,4096]-->[1,3,1024,1024]
    def __init__(self, modulation_dim, up_patch_dim):
        super(CompositeBlock, self).__init__()

        self.up_patch_dim = up_patch_dim


        self.modulation = ModulationBlock(4096, modulation_dim)  #[B,T,C]=[1,233,4096]
        self.up_patching = UpPatchBlock(up_patch_dim, 3, scale_factor=16)  # 根据需要调整上采样的尺寸,64*16=1024

    def forward(self, input1, input2):
        # 调制和线性变换
        modulated_input = self.modulation(input1,input2)  #[1,233,4096]
        batch_size, channel, features = modulated_input.shape

        modulated_input = modulated_input.reshape(batch_size*channel,features) #[1,233*4096]

        linear_layer = nn.Linear(in_features= features,out_features=features).to(device)
        x = linear_layer(modulated_input)       #[1,233,4096]

        features = math.sqrt(features)
        x = x.reshape(batch_size,channel,int(features),int(features))

        output = self.up_patching(x)
        return output


#  ---------------------------------------------数据选择-------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channel = 4096
token = 77
batch_size = 4
img_size = 1024  # input image size
patch_size = 16   # （1024/16=64，64^2=4096）
in_channels = 3   # [batch_size,in_channel,img_size,img_size]
D = 5   #d,MM-DIT块的个数,D>1
condition = "photo"   #MMDIT,input are photo or text(left and right parameter)
epoch = 3
t_step = 4

#输入的张量
y_input = torch.randn(batch_size, token, channel).to(device)    #[1,77,4096]
x_input = torch.randn(batch_size, in_channels, img_size, img_size).to(device)  #[batchsize:1,in_channel:3,1024,1024]from latent photo
c_input = torch.randn(batch_size,token*2,channel).to(device)  #[1,154,4096] from CLIP

#  ------------------------------------------------------------------------------------------------------------------



modelX = X(img_size, in_channels, patch_size, token*2).cuda()
modelY = Y(token, channel).cuda()
modelC = C(channel).cuda()
composite_block = CompositeBlock(modulation_dim=4096, up_patch_dim=233).cuda()
model_DIT = MM_DIT(channel,mlp_ratio=4).cuda()            #mlp_ration=64*D

class AllModel(nn.Module):
    def __init__(self,modelX,modelY,modelC,composite_block,model_DIT):
        super(AllModel, self).__init__()
        # 初始化你的模型组件
        self.modelY = modelY  # 替换为实际的类名
        self.modelX = modelX
        self.modelC = modelC
        self.model_DIT = model_DIT
        self.composite_block = composite_block

    def forward(self, input_x, input_y, input_c, D, condition="photo"):
        Y = self.modelY(input_y)
        X = self.modelX(input_x)
        C = self.modelC(input_c)

        for d in range(1, D):
            output_L, output_R = self.model_DIT(Y, C, X)
            Y = Y
            C = output_L
            X = output_R

        if condition == "photo":
            out = output_L
        else:      #text
            out = output_R

        output = self.composite_block(Y, out)
        return output

all_model = AllModel(modelX,modelY,modelC,composite_block,model_DIT)



def pi_t(t,m,s,mode="logit"):
    if mode == "logit":
        x = (t/(1-t)-m)**2/2*s*s
        x = -1*x
        y = s*math.sqrt(2*math.pi)*t*(1-t)
        out = 1/y*math.exp(x)

    elif mode == "heavy tails":
        f = 1-t-s*(math.cos(math.pi*t/2)**2-1+t)
        f = 1/f
        out = sp.diff(f,t)

    elif mode == "cosmap":
        out = math.pi-2*math.pi*t+2*math.pi*t*t
        out = 2/out
    return out


def loss_function(model,x_0,D,t,forward="RF"):

    e = torch.randn_like(x_0)

    if forward == "RF":
        z_t = (1-t)*x_0+t*e
        w_t = t/(1-t)                                #⭐w_t = t/(1-t)*pi_t(t,0,1)
        lambda_t = -2/(t-t**2)
    elif forward == "EDM":
        z_t = x_0+1*e
    elif forward =="cosine":
        z_t = math.cos(math.pi/2*t)*x_0+math.sin(math.pi/2*t)*e

    output = model(z_t,y_input,c_input,D)

    loss = -1*0.5* w_t * lambda_t * (output-e).square().mean()
    return loss



t = [random.random() for _ in range(t_step)]
t = sorted(t)


optimizer = torch.optim.Adam(all_model.parameters(),lr=1e-3)

for i in range(epoch):
    for index, value in enumerate(t):
        loss = loss_function(all_model, x_input, D, value)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_model.parameters(), 1.)
        optimizer.step()
        print("epoch:", i,"t_step:",index, "loss:", loss)




