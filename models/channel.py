
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math 
import random
import sys
sys.path.append('./')

from models.utils import clipping, add_cp_s,add_cp_m, rm_cp_m, batch_conv1d, PAPR, normalize, rm_cp_s



# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, opt, device):
        super(Channel, self).__init__()
        self.opt = opt

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).view(1,1,opt.L)     # 1x1xL
        self.power = power/torch.sum(power)   # Normalize the path power to sum to 1
        self.device = device
        eye = torch.flip(torch.eye(self.opt.L),[1]).to(self.device)
        eye = eye.repeat(opt.batch_size*opt.P,1,1).view(-1,opt.L)
        eye = eye/opt.L
        self.eye = torch.complex(eye,torch.zeros_like(eye)).to(torch.complex128)
        # self.eye = torch.complex(eye,torch.zeros_like(eye))
        


    def sample(self, N, P, M, L):
        
        # Sample the channel coefficients
        # cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        
        # 瑞利信道
        cof = torch.sqrt(self.power/2) * torch.tensor((np.random.rayleigh(size=(N,P,L)))).float()
        # print("shape",cof.shape)
        
        # H_t = None
        if self.opt.feedforward ==  'EXPLICIT-RES':
            cof_zp = torch.cat((cof, torch.zeros((N,P,self.opt.M-L))), -1)
            H_t = torch.fft.fft(cof_zp, dim=-1).to(self.device)
        else :

            H_t = None
        # print("ht_shape",H_t.shape)
        return cof, H_t


    def forward(self, input, cof=None,Ns=0,velocity=None):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK = input.shape
        if(self.opt.pkt=='OFDM'):
            if self.opt.is_pb:
                S = Ns+ self.opt.N_pilot
                M = self.opt.M+self.opt.K+ self.opt.N_pilot
            elif self.opt.is_pm:
                S = Ns+ self.opt.N_pilot
                M = self.opt.M+self.opt.K 
            else:
                S = Ns
                M = self.opt.M+self.opt.K + self.opt.N_pilot
        elif(self.opt.pkt=='OTFS'):
            S = Ns+ self.opt.K
            M = self.opt.M+self.opt.N_pilot 
        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, self.opt.M, self.opt.L)
            # cof, H_t = self.sampleJakes2(N, P, self.opt.M, self.opt.L,v)
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,M-self.opt.L,2))), 2)  
            cof_zp = torch.view_as_complex(cof_zp) 
            H_t = torch.fft.fft(cof_zp, dim=-1)
        

        # cof = cof.numpy()
        # ene = np.abs(cof)**2
        # sort_ind = np.argsort(-ene,axis=-1)
        # cof = np.take_along_axis(cof,sort_ind,axis=-1)
        # cof = torch.tensor(cof).to(self.device)
        # print(cof.shape)
        cof,_= torch.sort(cof,-1,descending=True)

        carrier_freq = 3.6e9
        subcarrierSpacing = 15e3
        samplingRate = M * subcarrierSpacing
        t = np.arange(S) / subcarrierSpacing
        t = torch.tensor(t).view(1,1,1,-1).to(self.device)
        # modulated_input = torch.tensor(np.exp(2j*np.pi*carrier_freq*t)).view(1,1,S*M).to(self.device) * input.to(self.device)
        # modulated_input = modulated_input.contiguous().view(N,P,M,S).permute(0,1,3,2)
        # print(input.shape)
        input_reshaped = input.unsqueeze(2).contiguous().view(N,P,1,S,M).to(self.device) 
        # 根据路径衰落生成路径距离
        # len = - 100 * np.log(cof).to(self.device) # N, P, L

        # delay1 = len / 3e8
        # 符号间的延迟差
        # ts = 0.5e-3 / 14
        # ts = 0
        # delay2= torch.linspace(0,(S-1) * ts,S).to(self.device)
        # delay2 = delay2.repeat(N,P,1)

        # delay = delay1.unsqueeze(3) + delay2.unsqueeze(2) # N,P,L,S
        
        
        

        # if self.opt.is_random_v:
        #     velocity = random.uniform(self.opt.v_min,self.opt.v_range)
        # else:
        #     velocity = float(self.opt.V)    


        

         # Calculate the maximum Doppler shift
        max_doppler_shift = velocity / 3e8 * carrier_freq   # N, 1

        # angles = torch.linspace(0, 2 * np.pi, self.opt.L).unsqueeze(0).repeat(N,1).to(self.device) #N,L
        angles = torch.rand(self.opt.L).unsqueeze(0).repeat(N,1).to(self.device) * 2 * np.pi 

        # angles = torch.linspace(0, 2 * np.pi, self.opt.L).to(self.device)
       
        # 将输入和系数张量的形状调整为匹配乘法的形状
        
        
        cof_reshaped = cof.view(N, P, self.opt.L, 1,1).to(self.device)  # (N, P, L, 1,1)

        # 将相位张量调整为匹配乘法的形状
        phases = (2 * np.pi * torch.cos(angles) * max_doppler_shift).unsqueeze(1).unsqueeze(-1)  # (N,1,L,1)
       
        phases += torch.tensor(np.random.uniform(low=0, high=2*np.pi, size=(N,P,self.opt.L,1))).to(self.device)
        # phases = (2 * np.pi * torch.cos(angles) * max_doppler_shift).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,L,1)


        phases = phases * t.to(self.device)  # (N,P,L, S)
        shift = torch.exp(1j * phases).unsqueeze(-1)  # (N,P, L, S,1)
        self.cof_fin = cof_reshaped * shift

        out = input_reshaped * self.cof_fin # (N,P, L, S,M)

        # out = torch.sum(out, dim=2)  # (N, P, M+K)
        # 
        # 不同多径的时延 
        # output = out.contiguous().view(N*P*self.opt.L,S,M).permute(0,2,1).contiguous().view(N*P*self.opt.L,-1)
        output = out.contiguous().view(N*P*self.opt.L,-1)
        
        output = batch_conv1d(output,self.eye)

        output = torch.sum(output.contiguous().view(N,P,self.opt.L,-1), dim=2)  # (N, P, MS)


        # 将输出调整为所需的形状
        # output = output.view(N, P, SMK)
        # demodulated_out = output * torch.tensor(np.exp(-2j*np.pi*carrier_freq*t)).to(self.device)
        # output = output.view(N,P,M,S).permute(0,1,3,2).contiguous().view(N,P,SMK)
        output = output.contiguous().view(N,P,SMK)
       

        H_t = torch.sum(self.cof_fin,2).squeeze()

        # H_t = torch.mean(self.cof_fin,-2).squeeze().unsqueeze(1)
        
        # cof_zp = torch.cat((H_t, torch.view_as_complex(torch.zeros((N,P,self.opt.M-self.opt.L,2))).to(self.device)), 2)  
        # cof_zp = torch.view_as_complex(cof_zp) 
        # print(H_t.shape)
        # H_t = torch.fft.fft(H_t, dim=-1)
        return output, H_t


# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    def __init__(self, opt, device, pilot_path):
        super(OFDM, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)
        self.device = device
        # Generate the pilot signal
        if(self.opt.pkt=='OTFS'):
            if not os.path.exists(pilot_path):
                bits = torch.randint(2, (opt.S,2))
                torch.save(bits,pilot_path)
                pilot = (2*bits-1).float()
                
            else:
                bits = torch.load(pilot_path)
                pilot = (2*bits-1).float()
        elif(self.opt.pkt=='OFDM'):
            if not os.path.exists(pilot_path):
                if self.opt.is_pb:
                    if opt.S>opt.M:
                        bits = torch.randint(2, (opt.S+opt.N_pilot,2))
                    else:
                        bits = torch.randint(2, (opt.M+opt.N_pilot,2))
                elif self.opt.is_pm:
                    
                    bits = torch.randint(2, (opt.M,2))
                else:
                    bits = torch.randint(2, (opt.S,2))
                torch.save(bits,pilot_path)
                pilot = (2*bits-1).float()
            else:
                bits = torch.load(pilot_path)
                pilot = (2*bits-1).float()
            # if self.opt.is_pm:
            #     bits = torch.load('./models/Pilot_bit%s.pt'%self.opt.M)
            # else:
                
            #     bits = torch.load('./models/Pilot_bit%s.pt'%self.opt.S)
            pilot = (2*bits-1).float()
    
        self.pilot = pilot.to(device)
        self.pilot = torch.view_as_complex(self.pilot)
        self.pilot = normalize(self.pilot, 1)
        # print(self.pilot)
        print("pilot shape",self.pilot.shape)
        #ISFFT
        if(self.opt.pkt=='OTFS'):

            self.pilot_is = self.pilot.view(1,self.opt.S,1).repeat(opt.P,1,opt.N_pilot)

        elif(self.opt.pkt=='OFDM'):
            # self.pilot_cp = add_cp_m(torch.fft.ifft(self.pilot), self.opt.K).repeat(opt.P, opt.N_pilot,1) 
            if opt.modulation == 'OFDM':
                if self.opt.is_pb:
                    self.pilot_s = self.pilot[:self.opt.S+opt.N_pilot].view(1,1,self.opt.S+opt.N_pilot,1).repeat(1,opt.P,1,opt.N_pilot)
                    self.pilot_m = self.pilot[:self.opt.M].view(1,1,1,self.opt.M).repeat(1,opt.P,opt.N_pilot,1)
                elif self.opt.is_pm:
                    self.pilot_cp = self.pilot.view(1,1,1,self.opt.M).repeat(1,opt.P,opt.N_pilot,1)
                else:
                    self.pilot_cp = self.pilot.view(1,1,self.opt.S,1).repeat(1,opt.P,1,opt.N_pilot)
            else:
                if opt.modulation == 'OTFS':
                    if self.opt.is_pm:
                        z = np.zeros((opt.S,opt.M))+np.zeros((opt.S,opt.M))*0j
                        
                    else:
                        z = np.zeros((opt.S,opt.N_pilot))+np.zeros((opt.S,opt.N_pilot))*0j
                    z[opt.S//2+1,0] = 4*(1+1*1j)
                    # print(z[0,:,:])
                    self.pilot_cp = torch.tensor(z).to(device).view(1,1,self.opt.S,-1).repeat(1,opt.P,1,1)
                    
                
            
        # self.pilot_otfs = torch.view_as_complex(torch.tensor(np.zeros((64, 6,2))))
        # self.pilot_otfs[30:30+2, 2:2+2] += np.random.rand(2, 2)*2+1j*np.random.rand(2, 2)*2
        # print(self.pilot_otfs)
        # self.pilot = self.pilot_otfs.to(device)
        # self.pilot_cp = self.pilot_otfs.view(1,1,64,6).to(device)

        # self.pilot_is = np.sqrt(self.pilot_is.shape[-2] / self.pilot_is.shape[-1]) * torch.fft.ifft(self.pilot_is,dim = -2)
        # print(self.pilot_is)
        # self.pilot_cp = add_cp(self.pilot_is, self.opt.K).repeat(opt.P, 1,1)
        
        
        # self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.opt.K).repeat(opt.P, opt.N_pilot,1)      
        # print(self.pilot_cp)  
        # print("pilot_cp",self.pilot_cp.shape)

        # self.eng = matlab.engine.start_matlab()
        # self.eng.addpath(r'/workspace/Deep-JSCC-for-images-with-OFDM')
        # self.eng.parpool(16)

    def forward(self, x, SNR, cof=None, batch_size=None,velocity=None):
        # Input size: NxPxSxM   The information to be transmitted
        # cof denotes given channel coefficients

        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)
        # print("x",x.shape)
        if not is_pilot:
            
            # Change to new complex representations
            N = x.shape[0]
            
            if(self.opt.pkt=='OTFS'):

                # ISFFT
                pilot = self.pilot_is.repeat(N,1,1,1) #  NxPxSx2
                
                x = torch.cat((pilot, x), 3)
                
                # print("xshape",x.shape)
                if self.opt.modulation == 'OTFS':
                    x = np.sqrt(x.shape[-2] / x.shape[-1]) * torch.fft.ifft(x,dim = -2)
                elif  self.opt.modulation == 'OFDM':
                    ## OFDM
                    x = np.sqrt(x.shape[-1]) * torch.fft.ifft(x, dim=-1)

                x = add_cp_s(x, self.opt.K)
                
            elif(self.opt.pkt=='OFDM'):
                # IFFT:                    NxPxSxM  => NxPxSxM
                # x = torch.fft.ifft(x, dim=-1)
                 # Add Cyclic Prefix:       NxPxSxM  => NxPxSx(M+K)
                

                # Add pilot:               NxPxSx(M+K)  => NxPx(S+2)x(M+K)

                

                if self.opt.modulation == 'OTFS':
                    # pilot = np.sqrt(pilot.shape[-2] / pilot.shape[-1]) * torch.fft.ifft(pilot, dim=-2)
                    pilot = self.pilot_cp.repeat(N,1,1,1)
                    
                    # x = pilot
                    if self.opt.is_pm:
                        x = torch.cat((x, pilot), 2)
                    else:

                        x = torch.cat((x, pilot), 3)
                    x = np.sqrt(x.shape[-2] / x.shape[-1]) * torch.fft.ifft(x,dim = -2)
                    # print(x[0,:])
                    
                elif  self.opt.modulation == 'OFDM':
                    ## OFDM
  

                    # pilot = np.sqrt(pilot.shape[-1]) * torch.fft.ifft(pilot, dim=-1)
                    # x = np.sqrt(x.shape[-1]) * torch.fft.ifft(x, dim=-1)
                
                    if self.opt.is_pb:
                        
                        x = torch.cat((x, self.pilot_m.repeat(N,1,1,1)), 2)
                        x = torch.cat((x, self.pilot_s.repeat(N,1,1,1)), 3)
                    elif self.opt.is_pm:
                        pilot = self.pilot_cp.repeat(N,1,1,1)
                        
                        x = torch.cat((x, pilot), 2)
                    else:
                        pilot = self.pilot_cp.repeat(N,1,1,1)
                        
                        x = torch.cat((x, pilot), 3)
                        
                    x = np.sqrt(x.shape[-1]) * torch.fft.ifft(x, dim=-1)
                x = add_cp_m(x, self.opt.K)
                
               
                      

            Ns = self.opt.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N,1,1,1)
            Ns = 0    
        # print("x",x.shape)
        if(self.opt.pkt=='OFDM'):
            if self.opt.is_pb:
                S = Ns+self.opt.N_pilot
                M = self.opt.M+self.opt.K+self.opt.N_pilot
            elif self.opt.is_pm:
                S = Ns+self.opt.N_pilot
                M = self.opt.M+self.opt.K
            else:
                S = Ns
                M = self.opt.M+self.opt.K+self.opt.N_pilot
            # x = x.permute(0,1,3,2).contiguous().view(N, self.opt.P, (S)*(M))
            # print(x.shape)
            x = x.contiguous().view(N, self.opt.P, (S)*(M))
        elif(self.opt.pkt=='OTFS'):
            S = Ns + self.opt.K 
            M = self.opt.M+self.opt.N_pilot
            x = x.view(N, self.opt.P, (S)*M)
       
        
      
        

        # papr = 0
        # papr_cp = 0

        # print("x.v",x.shape)
        # PAPR before clipping
        papr = PAPR(x)
        
        # Clipping (Optional):     NxPx(S+1)(M+K)  => NxPx(S+1)(M+K)
        if self.opt.is_clip:
            x = self.clip(x)
        
        # PAPR after clipping
        papr_cp = PAPR(x)
        
        
        # Pass through the Channel:        NxPx(S+1)(M+K)  =>  NxPx((S+1)(M+K))
        y, H_t = self.channel(x, cof, Ns,velocity)
        
        # print("channel diff",torch.sum(x[0,:].abs()**2)/torch.sum(y[0,:].abs()**2))

        # x = x.contiguous().view(-1,S*M).cpu().numpy()
        # y = np.array(self.eng.channel(x))



        
        # y = torch.from_numpy(y).contiguous().view(N,self.opt.P,M,S).permute(0,1,3,2).to(self.device)
        
        # print("y",y.shape)
        # Calculate the power of received signal        
        pwr = torch.mean(y.abs()**2, -1, True)

        noise_pwr = pwr*10**(-SNR/10)
        # # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise

        # NxPx((S+S')(M+K))  =>  NxPx(S+S')x(M+K)
        output = y_noisy.view(N, self.opt.P, S ,M )

        # SNR1 = 2
        # SNR2 = 11
        # SNR3 = 11
        # SNR4 = 2
        # n_pwr1 = pwr*10**(-SNR1/10)
        # n_pwr2 = pwr*10**(-SNR2/10)
        # n_pwr3 = pwr*10**(-SNR3/10)
        # n_pwr4 = pwr*10**(-SNR4/10)
        # noise1 = torch.sqrt(n_pwr1/2).view(N,1,1,1) * (torch.randn(N,self.opt.P,int(S/4),M) + 1j*torch.randn(N,self.opt.P,int(S/4),M)).to(self.device)
        # noise2 = torch.sqrt(n_pwr2/2).view(N,1,1,1)  * (torch.randn(N,self.opt.P,int(S/4),M) + 1j*torch.randn(N,self.opt.P,int(S/4),M)).to(self.device)
        # noise3 = torch.sqrt(n_pwr3/2).view(N,1,1,1)  * (torch.randn(N,self.opt.P,int(S/4),M) + 1j*torch.randn(N,self.opt.P,int(S/4),M)).to(self.device)
        # noise4 = torch.sqrt(n_pwr4/2).view(N,1,1,1)  * (torch.randn(N,self.opt.P,int(S/4),M) + 1j*torch.randn(N,self.opt.P,int(S/4),M)).to(self.device)
        # noise = torch.cat((noise1,noise2,noise3,noise4),-2)
        # output = y.view(N, self.opt.P, S ,M ) + noise
        # print(n_pwr1.shape)

        output = rm_cp_m(output, self.opt.K)
        if self.opt.modulation == 'OTFS':
            
            output = torch.fft.fft(output, dim=-2) * np.sqrt(x.shape[-1] / x.shape[-2])
           
        elif self.opt.modulation == 'OFDM':
            output = torch.fft.fft(output, dim=-1) / np.sqrt(output.shape[-1])
        if self.opt.is_pb:
            y_pilot_m = output[:,:,-self.opt.N_pilot:,:-self.opt.N_pilot]         # NxPxS'x(M+K)
            y_pilot_s = output[:,:,:-self.opt.N_pilot,-self.opt.N_pilot:]
            y_sig = output[:,:,:-self.opt.N_pilot,:-self.opt.N_pilot]           # NxPxSx(M+K)
            return y_pilot_m,y_pilot_s, y_sig, H_t, noise_pwr, papr, papr_cp
        elif self.opt.is_pm:
            y_pilot = output[:,:,-self.opt.N_pilot:,:]         # NxPxS'x(M+K)
            
            y_sig = output[:,:,:-self.opt.N_pilot,:]           # NxPxSx(M+K)
        else:
            y_pilot = output[:,:,:,-self.opt.N_pilot:]         # NxPxS'x(M+K)
            
            y_sig = output[:,:,:,:-self.opt.N_pilot]           # NxPxSx(M+K)
        # print((torch.abs(y_pilot)**2).sum()/y_pilot.numel())
        # print((torch.abs(y_sig)**2).sum()/y_sig.numel())
        return y_pilot, y_sig, H_t, noise_pwr, papr, papr_cp

        '''
        if(self.opt.pkt=='OTFS'):
            #SFFT
            output = rm_cp_s(output, self.opt.K)
            if self.opt.modulation == 'OTFS':
                output = np.sqrt(output.shape[-1] / output.shape[-2]) * torch.fft.fft(output,dim = -2)
            elif  self.opt.modulation == 'OFDM':
                output = torch.fft.fft(output,dim = -1)
            

            ## ofdm
            # output = torch.fft.fft(output,dim = -1)


            # print(output.shape)
            y_pilot = output[:,:,:,:self.opt.N_pilot]     
                
            y_sig = output[:,:,:,self.opt.N_pilot:] 
            # print("ypilot",y_pilot.shape)
            # print("ysig",y_sig.shape)          
        elif(self.opt.pkt=='OFDM'):
            if self.opt.is_pb:
                y_pilot_m = output[:,:,-self.opt.N_pilot:,:]         # NxPxS'x(M+K)
                y_pilot_s = output[:,:,:,-self.opt.N_pilot:]
                y_sig = output[:,:,:-self.opt.N_pilot,:]           # NxPxSx(M+K)
            elif self.opt.is_pm:
                y_pilot = output[:,:,-self.opt.N_pilot:,:]         # NxPxS'x(M+K)
                
                y_sig = output[:,:,:-self.opt.N_pilot,:]           # NxPxSx(M+K)
            else:
                y_pilot = output[:,:,:,-self.opt.N_pilot:]         # NxPxS'x(M+K)
                
                y_sig = output[:,:,:,:-self.opt.N_pilot]           # NxPxSx(M+K)
        if not is_pilot:

            if(self.opt.pkt=='OTFS'):
                info_pilot = y_pilot
                info_sig = y_sig
                # print("pilot diff",torch.sum(info_pilot.abs()-pilot.abs()))
            elif(self.opt.pkt=='OFDM'):
                if self.opt.is_pm:
                    y_pilot = rm_cp_m(y_pilot, self.opt.K)  

                y_sig = rm_cp_m(y_sig, self.opt.K)    
                if self.opt.modulation == 'OTFS':
                    info_pilot = np.sqrt(y_pilot.shape[-1] / y_pilot.shape[-2]) * torch.fft.fft(y_pilot,dim = -2)
                    info_sig = np.sqrt(y_sig.shape[-1] / y_sig.shape[-2]) * torch.fft.fft(y_sig,dim = -2)
                elif  self.opt.modulation == 'OFDM':
                    info_pilot = torch.fft.fft(y_pilot, dim=-1) / np.sqrt(y_pilot.shape[-1])
                    info_sig = torch.fft.fft(y_sig, dim=-1) / np.sqrt(y_sig.shape[-1])
                # Remove Cyclic Prefix:   
                # info_pilot = rm_cp_m(y_pilot, self.opt.K)    # NxPxS'xM

                # FFT:                     
                
                # print("INFOSIG",info_sig.shape)

                

            # H_t = None
            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = rm_cp(y_pilot, self.opt.K)    # NxPxS'xM
            info_pilot = torch.fft.fft(info_pilot, dim=-1)

            return info_pilot, H_t, noise_pwr
        '''


# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):
    
    def __init__(self, opt, device):
        super(PLAIN, self).__init__()
        self.opt = opt

        # Setup the channel layer
        self.channel = Channel(opt, device)

    def forward(self, x, SNR):

        # Input size: NxPxM   
        N, P, M = x.shape
        y = self.channel(x, None)
        
        # Calculate the power of received signal
        pwr = torch.mean(y.abs()**2, -1, True)      
        noise_pwr = pwr*10**(-SNR/10)
        
        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise                                    # NxPx(M+L-1)
        rx = y_noisy[:, :, :M, :]
        return rx 


if __name__ == "__main__":

    import argparse
    opt = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt.P = 1
    opt.S = 6
    opt.M = 64
    opt.K = 16
    opt.L = 8
    opt.decay = 4
    opt.N_pilot = 1
    opt.SNR = 10
    opt.is_clip = False

    ofdm = OFDM(opt, 0, './models/Pilot_bit.pt')

    input_f = torch.randn(32, opt.P, opt.S, opt.M) + 1j*torch.randn(32, opt.P, opt.S, opt.M)
    input_f = normalize(input_f, 1)
    input_f = input_f.cuda()

    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR, v=10)
    H_t = H_t.cuda()
    err = input_f*H_t.unsqueeze(0) 
    err = err - info_sig
    print(f'OFDM path error :{torch.mean(err.abs()**2).data}')


    info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp = ofdm(input_f, opt.SNR, v=1000)
    H_t = H_t.cuda()
    err = input_f*H_t.unsqueeze(0) 
    err = err - info_sig
    print(f'OFDM path error :{torch.mean(err.abs()**2).data}')

    from utils import ZF_equalization, MMSE_equalization, LS_channel_est, LMMSE_channel_est

    H_est_LS = LS_channel_est(ofdm.pilot, info_pilot)
    err_LS = torch.mean((H_est_LS.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LS channel estimation error :{err_LS.data}')

    H_est_LMMSE = LMMSE_channel_est(ofdm.pilot, info_pilot, opt.M*noise_pwr)
    err_LMMSE = torch.mean((H_est_LMMSE.squeeze()-H_t.squeeze()).abs()**2)
    print(f'LMMSE channel estimation error :{err_LMMSE.data}')
    
    rx_ZF = ZF_equalization(H_t.unsqueeze(0), info_sig)
    err_ZF = torch.mean((rx_ZF.squeeze()-input_f.squeeze()).abs()**2)
    print(f'ZF error :{err_ZF.data}')

    rx_MMSE = MMSE_equalization(H_t.unsqueeze(0), info_sig, opt.M*noise_pwr)
    err_MMSE = torch.mean((rx_MMSE.squeeze()-input_f.squeeze()).abs()**2)
    print(f'MMSE error :{err_MMSE.data}')







