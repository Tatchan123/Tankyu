import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
import random
import copy


def rmw(model,x,tobaoption):
    epsilon, complement, rmw_layer = [tobaoption[key] for key in ["epsilon", "complement", "rmw_layer"]]
    params = copy.deepcopy(model.params)
    batch_x = model.layers["toba"].forward(x,params)
    for idx in range(1,max(rmw_layer)+1):
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        
        rmlist = []
        complist = []
        difflist = []
        out = []
        for i in batch_x:
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #x_batch_y
        # out = np.transpose(out,(1,0,2))
        for i in range(0,len(out)-1):
            if i in complist:
                continue
            for j in range(i+1,len(out)):
                diff = out[i] - out[j]
                disp = np.var(diff,axis=0)
                if disp <= epsilon[idx-1]:
                    rmlist.append(i)
                    complist.append(j)
                    difflist.append(np.average(diff))
                    # difflist.append(np.average(diff,axis=0))
                    break
        if complement:
            difflist=np.asarray(difflist)
            scalar = np.ones(len(params["W"+str(idx)]))
            for n in range(len(rmlist)):
                scalar[int(complist[n])] += scalar[int(rmlist[n])]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            params["b"+str(idx)] += np.array([np.sum(difflist)]*len(params["b"+str(idx)]))
            # params["b"+str(idx)] += np.sum(difflist,axis=0)
            
        if idx == 1:
            params["init_remove"].append(rmlist)
            params["W1"] = np.delete(params["W1"],rmlist,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
            
        print("    layer"+str(idx),": delete",str(len(rmlist))+"nodes")

        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
                      
    return params



def random_rmw(model,x,tobaoption):
    rmw_layer, delete_n = [tobaoption[key] for key in ["rmw_layer", "delete_n"]]
    params = model.params
    for idx in rmw_layer:
        lst = random.sample(list(range(len(params["W"+str(idx)]))), delete_n[idx-1])
        if idx == 1:
            params["W1"] = np.delete(params["W1"],lst,axis=0)
            params["init_remove"].append(lst)
        
        else:    
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],lst,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],lst)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],lst,axis=0)
            
    return params



def count_rmw(model,x,tobaoption):
    complement, rmw_layer, delete_n = [tobaoption[key] for key in ["complement", "rmw_layer", "delete_n"]]
    params = model.params
    batch_x = model.layers["toba"].forward(x,params)
    for idx in range(1,max(rmw_layer)+1):
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        out = []
        for i in batch_x:
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #y_batch_x
        
        pair_list = []
        displist = []
        
        rmlist = []
        complist = []
        difflist = []
        
        for i in range(0,len(out)-1):
            for j in range(i+1,len(out)):
                diff = out[i] - out[j]
                disp = np.average(diff**2) - np.average(diff)**2
                pair_list.append([i,j])
                displist.append(disp)
        ziplist = zip(displist,pair_list)
        zipsort = sorted(ziplist)
        displist,pair_list = zip(*zipsort)
        rmpair = [pair_list[0]]
        cnt = 1
        while len(rmpair)<delete_n[idx-1]:
            for i in rmpair:
                if i[1] == pair_list[cnt][0] or i[0] == pair_list[cnt][0]:
                    cnt += 1
                    break
            else:
                rmpair.append(pair_list[cnt])
                cnt +=1
        
        rmlist = np.array(rmpair).T[0]
        complist = np.array(rmpair).T[1]
        difflist = out[rmlist] - out[complist]   
        difflist = np.mean(difflist,axis=1)    
        if complement:
            difflist=np.asarray(difflist)
            scalar = np.array([1]*len(params["W"+str(idx)]))
            for n in range(len(rmlist)):
                scalar[int(complist[n])] += scalar[int(rmlist[n])]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            diffs = np.sum(difflist)
            params["b"+str(idx)] += np.array([np.sum(difflist)]*len(params["b"+str(idx)]))
        if idx == 1:
            params["init_remove"].append(rmlist)
            params["W1"] = np.delete(params["W1"],rmlist,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
        
        print("    layer"+str(idx),": delete",str(len(rmlist))+"nodes")

        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
                      
    return params



def auto_epsilon_rmw(model,x,tobaoption):
    complement, rmw_layer = [tobaoption[key] for key in ["complement", "rmw_layer"]]
    params = model.params
    batch_x = model.layers["toba"].forward(x,params)
    for idx in range(1,max(rmw_layer)+1):
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        
        rmlist = []
        complist = []
        difflist = []
        out = []
        for i in batch_x:
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        #shape : batch, in_size, out_size
        try:
            s = len(params["W"+str(idx+1)])
            epsilon = 2 ** -(9 + 0.01*s)
            
            
        except:
            actual_out = np.sum(out,axis=1).reshape(-1) #batch*out_size
            Sout = np.average(actual_out ** 2) - np.average(actual_out)**2
            epsilon = 2 ** -(5*Sout**2 + 8)
            #次の層がsoftmaxのやつ用　出力の分散の逆数を使う
            
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #in_size,batch,out_size
        for i in range(0,len(out)-1):
            if i in complist:
                continue
            for j in range(i+1,len(out)):
                diff = out[i] - out[j]
                disp = np.average(diff**2) - np.average(diff)**2
                if disp <= epsilon:
                    rmlist.append(i)
                    complist.append(j)
                    difflist.append(np.average(diff))
                    break
        if complement:
            difflist=np.asarray(difflist)
            scalar = np.array([1]*len(params["W"+str(idx)]))
            for n in range(len(rmlist)):
                scalar[int(complist[n])] += scalar[int(rmlist[n])]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            params["b"+str(idx)] += np.array([np.sum(difflist)]*len(params["b"+str(idx)]))
            
        if idx == 1:
            params["init_remove"].append(rmlist)
            params["W1"] = np.delete(params["W1"],rmlist,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
            
        print("    layer"+str(idx-1),": delete",str(len(rmlist))+"nodes")
        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
                      
    return params
    
    
    
def corrcoref_rmw(model,x,tobaoption):
    epsilon, complement, rmw_layer = [tobaoption[key] for key in ["epsilon", "complement", "rmw_layer"]]
    params = copy.deepcopy(model.params)
    batch_x = model.layers["toba"].forward(x,params)
    for idx in range(1,max(rmw_layer)+1):
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        
        rmlist = []
        complist = []
        alist = []
        blist = []
        out = []
        for i in batch_x:
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #x_batch_y
        for i in range(0,len(out)-1):
            if i in complist:
                continue
            for j in range(i+1,len(out)):
                i_val = out[i]
                j_val = out[j]
                sxy = np.average(i_val*j_val) - np.average(i_val)*np.average(j_val)
                vari = (np.average(i_val**2)) - (np.average(i_val))**2
                varj = (np.average(j_val**2)) - (np.average(j_val))**2
                cor = sxy / (np.sqrt(vari)*np.sqrt(varj) + 0.00000001) #1e-10みたいな表記だとうまくいかないなぜ
                #print(sxy,vari,varj)
                #print("    ",cor)
                if np.isnan(cor): print(sxy,vari,varj,cor)
                if abs(cor) >= epsilon[idx-1]:
                    rmlist.append(i)
                    complist.append(j)                 # i=x , j=yとしてy=ax+bに近似
                    a = sxy/vari
                    b = np.average(j_val) - a*np.average(i_val)
                    alist.append(a)
                    blist.append(b)
                    break
                
        blist = np.array(blist)
        alist = np.array(alist)
        if complement:
            scalar = np.ones(len(params["W"+str(idx)]))
            for n in range(len(rmlist)):
                scalar[int(complist[n])] += alist[n]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            params["b"+str(idx)] = params["b"+str(idx)] + np.sum(blist)
            # params["b"+str(idx)] += np.sum(difflist,axis=0)
            
        if idx == 1:
            params["init_remove"].append(rmlist)
            params["W1"] = np.delete(params["W1"],rmlist,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
            
        print("    layer"+str(idx),": delete",str(len(rmlist))+"nodes")

        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
    return params