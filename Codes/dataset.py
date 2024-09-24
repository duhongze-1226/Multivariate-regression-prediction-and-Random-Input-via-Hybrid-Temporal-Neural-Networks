import pandas as pd
import numpy as np
import torch
from math import ceil,floor

def getDataFromFile(FileName,paths,state):
    Datas = pd.read_excel(paths + FileName)
    Datas = Datas.drop(Datas.columns[0],axis=1)
    if state == False:
        Datas = Datas.drop(Datas.columns[0],axis=1)
    return Datas

def addGaussianNoise(Datas,Lambda,state):
    torch.manual_seed(123)
    Datas = Datas.clone()
    if state == False:
        noise = torch.randn_like(Datas)
        Datas += Lambda*noise
    else:
        noise = torch.randn_like(Datas[:,1:])
        Datas[:,1:]  += Lambda*noise
    return Datas

def addLaplaceNoise(Datas, Lambda, state):
    torch.manual_seed(123)
    Datas = Datas.clone()
    # 创建 Laplace 分布，location=0, scale=1
    laplace = torch.distributions.Laplace(0, 1)
    
    if state == False:
        # 生成与 Datas 大小相同的 Laplace 噪声
        noise = laplace.sample(Datas.shape)
        Datas += Lambda * noise
    else:
        # 生成与部分数据大小相同的 Laplace 噪声
        noise = laplace.sample(Datas[:, 1:].shape)
        Datas[:, 1:] += Lambda * noise
    
    return Datas

def addCauchyNoise(Datas, Lambda, state):
    torch.manual_seed(123)
    Datas = Datas.clone()
    # 创建 Cauchy 分布，location=0, scale=1
    cauchy = torch.distributions.Cauchy(0, 1)
    
    if state == False:
        # 生成与 Datas 大小相同的 Cauchy 噪声
        noise = cauchy.sample(Datas.shape)
        Datas += Lambda * noise
    else:
        # 生成与部分数据大小相同的 Cauchy 噪声
        noise = cauchy.sample(Datas[:, 1:].shape)
        Datas[:, 1:] += Lambda * noise
    
    return Datas

def getTensorFromFile(FileName,paths,state=False):
    Datas = pd.read_excel(paths + FileName)
    Datas = Datas.drop(Datas.columns[0],axis=1)
    if state==False:
        Datas = Datas.drop(Datas.columns[0],axis=1)
    return torch.tensor(Datas.values).float().unsqueeze(0)

def getDatas(params):
    Strains = []
    Loads = []
    if (len(params['inputs_FileName']) == len(params['outputs_FileName'])):
        FilesNum = len(params['inputs_FileName'])
        for num in range(FilesNum): #load data from each file
            strain = torch.tensor(getDataFromFile(params['inputs_FileName'][num],params['paths'],params['state']).values) #transfor to torch.tensor
            if params['Noise_type'] == 'Laplace':
                strain = addLaplaceNoise(strain,params['Noise_level'],params['state']) 
            elif params['Noise_type'] == 'Cauchy':
                strain = addCauchyNoise(strain,params['Noise_level'],params['state']) 
            else:
                strain = addGaussianNoise(strain,params['Noise_level'],params['state'])
            load = torch.tensor(getDataFromFile(params['outputs_FileName'][num],params['paths'],state=True).values)
            data_length = strain.size(0) # compute length of each exprimental data
            
            if ((data_length-params['seq_len'])/params['SampleStride'][num]+1).is_integer(): # determine whether it can be rounded
                
                for i in range(floor(((data_length-params['seq_len'])/params['SampleStride'][num]+1))):
                    Strains.append(strain[i*params['SampleStride'][num]:params['seq_len']+i*params['SampleStride'][num]])
                    Loads.append(load[i*params['SampleStride'][num]:params['seq_len']+i*params['SampleStride'][num]]) 
                    
            else:                                                         # for 100-1 file
                for i in range(floor(((data_length-params['seq_len'])/params['SampleStride'][num]+1))):
                    Strains.append(strain[i*params['SampleStride'][num]:params['seq_len']+i*params['SampleStride'][num]])
                    Loads.append(load[i*params['SampleStride'][num]:params['seq_len']+i*params['SampleStride'][num]]) 
                Strains.append(strain[-params['seq_len']:])                         # Take 100 data points from the end of the data as a sample
                Loads.append(load[-params['seq_len']:])
    else: 
        print ('Files Warning!!! The number of strain files and the number of load files are not equal')
    
    Strains = torch.stack(Strains).double()
    Loads = torch.stack(Loads).double()
    
    return Strains,Loads

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,params):
        super(MyDataset,self).__init__()
        self.Strains,self.Loads = getDatas(params)
        self.Strains = self.Strains.to(params['device'])
        self.Loads = self.Loads.to(params['device'])
    def __getitem__(self,idx):
        input_variable = self.Strains[idx]
        output_variable = self.Loads[idx]
        return input_variable,output_variable
    
    def __len__(self):
        if self.Strains.size(0) != self.Loads.size(0):
            print ("***Data Error! The lengths of the input variables and output variables are not equal***")
        return self.Strains.size(0)
    
def mask_input(input_seq,params):
    mask_index = torch.randint(low=0, high=params['input_size']+1, size=(input_seq.size(0),1)).to(params['device'])
    length = input_seq.size(1)
    for i in range(mask_index.size(0)):
        if params['state'] == False:
            mask = torch.ones(length, params['input_size']).to(params['device'])
        else:
            mask = torch.ones(length, params['input_size']+1).to(params['device'])
        index = mask_index[i].item()
        if params['state'] == False:
            if index == params['input_size']:
                input_seq[i] = input_seq[i]
            else:
                mask[:,index] = 0
                input_seq[i] = input_seq[i] * mask
        else:
            if index == params['input_size']:
                input_seq[i] = input_seq[i]
            else:
                mask[:,index+1] = 0
                input_seq[i] = input_seq[i] * mask
    return input_seq


'''def getDatas_noParams(Strains_FileName,Loads_FileName,SampleStride,seq_len,paths):
    Strains = []
    Loads = []
    if (len(Strains_FileName) == len(Loads_FileName)):
        FilesNum = len(Strains_FileName)
        for num in range(FilesNum): #load data from each file
            strain = torch.tensor(getDataFromFile(Strains_FileName[num],paths).values) #transfor to torch.tensor
            load = torch.tensor(getDataFromFile(Loads_FileName[num],paths).values)
            data_length = strain.size(0) # compute length of each exprimental data
            
            if ((data_length-seq_len)/SampleStride[num]+1).is_integer(): # determine whether it can be rounded
                
                for i in range(floor(((data_length-seq_len)/SampleStride[num]+1))):
                    Strains.append(strain[i*SampleStride[num]:seq_len+i*SampleStride[num]])
                    Loads.append(load[i*SampleStride[num]:seq_len+i*SampleStride[num]]) 
                    
            else:                                                         # for 100-1 file
                for i in range(floor(((data_length-seq_len)/SampleStride[num]+1))):
                    Strains.append(strain[i*SampleStride[num]:seq_len+i*SampleStride[num]])
                    Loads.append(load[i*SampleStride[num]:seq_len+i*SampleStride[num]]) 
                Strains.append(strain[-seq_len:])                         # Take 100 data points from the end of the data as a sample
                Loads.append(load[-seq_len:])
    else: 
        print ('Files Warning!!! The number of strain files and the number of load files are not equal')
    
    Strains = torch.stack(Strains).double()
    Loads = torch.stack(Loads).double()
    
    return Strains,Loads'''

'''def mask_input(input_seq,params):
    mask_index = torch.randint(low=0, high=params['input_size']+1, size=(input_seq.size(0),1)).to(params['device'])
    length = input_seq.size(1)
    for i in range(mask_index.size(0)):
        mask = torch.ones(length, params['input_size']).to(params['device'])
        index = mask_index[i].item()
        if index == params['input_size']:
            input_seq[i] = input_seq[i]
        else:
            mask[:,index] = 0
            input_seq[i] = input_seq[i] * mask
    return input_seq'''