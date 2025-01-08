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
    if 1 in rmw_layer: batch_x = model.layers["toba"].forward(x,params)
    else: batch_x = x
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
    batch_x = model.layers["Toba"].forward(x,params)
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
    epsilon, complement, rmw_layer, delete_n = [tobaoption[key] for key in ["epsilon", "complement", "rmw_layer","delete_n"]]
    params = copy.deepcopy(model.params)
    if 1 in rmw_layer: batch_x = model.layers["Toba"].forward(x,params) #TODO昆布に対応
    else: batch_x = x
    for idx in range(1,max(rmw_layer)+1):  
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        
        rmlist = []
        complist = []
        corlist = []
        alist = []
        blist = []
        
        rmlist_s = []
        complist_s = []
        alist_s = []
        blist_s = []
        
        out = []
        for i in batch_x:            #ループなしにしたい
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #x_batch_y
        for i in range(0,len(out)-1):
            for j in range(i+1,len(out)):
                i_val = out[i]
                j_val = out[j]
                sxy = np.average(i_val*j_val) - np.average(i_val)*np.average(j_val)
                vari = (np.average(i_val**2)) - (np.average(i_val))**2
                varj = (np.average(j_val**2)) - (np.average(j_val))**2
                cor = sxy / (np.sqrt(vari)*np.sqrt(varj) + 0.00000001) #1e-10みたいな表記だとうまくいかないなぜ
                a = sxy/vari
                b = np.average(j_val) - a*np.average(i_val)
                
                corlist.append(abs(cor))
                rmlist.append(i)
                complist.append(j)
                alist.append(a)
                blist.append(b)                
        
        ziplist = zip(corlist,rmlist,complist,alist,blist)
        zipsort = sorted(ziplist,reverse=True)
        corlist,rmlist,complist,alist,blist = zip(*zipsort)
        cnt = 1
        while len(rmlist_s)<delete_n[idx-1]:
            if (rmlist[cnt] not in rmlist_s) and (rmlist[cnt] not in complist_s):
                rmlist_s.append(rmlist[cnt])
                complist_s.append(complist[cnt])
                alist_s.append(alist[cnt])
                blist_s.append(blist[cnt])
                print(corlist[cnt])
            cnt += 1
                
        blist_s = np.array(blist_s)
        alist_s = np.array(alist_s)
        if complement:
            scalar = np.ones(len(params["W"+str(idx)]))
            for n in range(len(rmlist_s)):
                scalar[int(complist_s[n])] += alist_s[n]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            params["b"+str(idx)] = params["b"+str(idx)] + np.sum(blist_s)
            # params["b"+str(idx)] += np.sum(difflist,axis=0)
            
        if idx == 1:
            params["init_remove"].append(rmlist_s)
            params["W1"] = np.delete(params["W1"],rmlist_s,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist_s,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist_s)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist_s,axis=0)
            
        print("    layer"+str(idx),": delete",str(len(rmlist_s))+"nodes")

        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist_s,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
    return params



def conv_corrcoef_toba(model,x,tobaoption):
    epsilon, complement, rmw_layer, delete_n = [tobaoption[key] for key in ["epsilon", "complement", "rmw_layer","delete_n"]]
    params = copy.deepcopy(model.params)
    batch_x = x
    conv_index = 1
    pc = 1
    rmw_index = 0
    print(batch_x.shape)
    while type(rmw_layer[rmw_index]) == str and "c" in rmw_layer[rmw_index]:
        if not "c"+str(conv_index) in rmw_layer:
            batch_x = model.layers["Conv2d"+str(conv_index)].forward(batch_x,params)
            
            try: batch_x = model.layers["BatchNorm"+str(conv_index)].forward(batch_x,params)
            except: pass
            
            try: batch_x = model.layers["ConvDrop"+str(conv_index)].forward(batch_x,params)
            except: pass

            batch_x = model.layers["ConvActivation"+str(conv_index)].forward(batch_x,params)
            conv_index += 1
            try:
                batch_x = model.layers["Maxpool"+str(pc)].forward(batch_x,params)
                pc += 1
            except:
                pass
            print(batch_x.shape)
            continue

        layer = model.layers["Conv2d"+str(conv_index)]
        pad = layer.P
        B,C,Ih,Iw = batch_x.shape
        F = params["F"+str(conv_index)]
        M,C,Fh,Fw = F.shape
        Oh = Ih + 2*pad -Fh + 1
        Ow = Iw + 2*pad -Fw + 1
        x = np.pad(batch_x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
        col = layer.im2col(x,B,C,Fh,Fw,Oh,Ow)
        col = col.reshape(C,Fh*Fw,B*Oh*Ow)
        F = np.transpose(F,(1,0,2,3)).reshape(C,M,-1)
        out = []
        for channel in range(C):
            y = np.dot(F[channel],col[channel]).reshape(-1)
            out.append(y)
        print("out calclated")
        rmlist = []
        complist = []
        corlist = []
        alist = []
        blist = []
        
        rmlist_s = []
        complist_s = []
        alist_s = []
        blist_s = []
    
        for i in range(0,len(out)-1):
            print(i)
            for j in range(i+1,len(out)):
                i_val = out[i]
                j_val = out[j]
                sxy = np.average(i_val*j_val) - np.average(i_val)*np.average(j_val)
                vari = (np.average(i_val**2)) - (np.average(i_val))**2
                varj = (np.average(j_val**2)) - (np.average(j_val))**2
                cor = sxy / (np.sqrt(vari)*np.sqrt(varj) + 0.00000001) #1e-10みたいな表記だとうまくいかないなぜ
                a = sxy/vari
                b = np.average(j_val) - a*np.average(i_val)
                
                corlist.append(abs(cor))
                rmlist.append(i)
                complist.append(j)
                alist.append(a)
                blist.append(b)                
        print("all pairs tested")
        ziplist = zip(corlist,rmlist,complist,alist,blist)
        zipsort = sorted(ziplist,reverse=True)
        corlist,rmlist,complist,alist,blist = zip(*zipsort)
        cnt = 0
        while len(rmlist_s)<delete_n[conv_index-1]:
            if (rmlist[cnt] not in rmlist_s) and (rmlist[cnt] not in complist_s) and (corlist[cnt] > epsilon[conv_index-1]):
                rmlist_s.append(rmlist[cnt])
                complist_s.append(complist[cnt])
                alist_s.append(alist[cnt])
                blist_s.append(blist[cnt])
                #print(corlist[cnt])
            cnt += 1
            if cnt == len(rmlist):break
                
        blist_s = np.array(blist_s)
        alist_s = np.array(alist_s)
        if complement:
            scalar = np.ones(len(params["F"+str(conv_index)]))
            for n in range(len(rmlist_s)):
                scalar[int(complist_s[n])] += alist_s[n]
            params["F"+str(conv_index)] = params["F"+str(conv_index)] * (scalar.reshape(-1,1,1,1))
            params["Cb"+str(conv_index)] = params["Cb"+str(conv_index)] + np.sum(blist_s)
        
        params["F"+str(conv_index-1)] = np.delete(params["F"+str(conv_index-1)],rmlist_s,axis=0)
        params["Cb"+str(conv_index-1)] = np.delete(params["Cb"+str(conv_index-1)],rmlist_s)
        params["F"+str(conv_index)] = np.delete(params["F"+str(conv_index)],rmlist_s,axis=1)

        print("    conv_layer"+str(conv_index)," : delete",str(len(rmlist_s))+"nodes")

        batch_x = model.layers["Conv2d"+str(conv_index)].forward(batch_x,params)

        try: batch_x = model.layers["BatchNorm"+str(conv_index)].forward(batch_x,params)
        except: pass

        try: batch_x = model.layers["ConvDrop"+str(conv_index)].forward(batch_x,params)
        except: pass

        batch_x = model.layers["ConvActivation"+str(conv_index)].forward(batch_x,params)
        conv_index += 1

        try:
            batch_x = model.layers["Maxpool"+str(pc)].forward(batch_x,params)
            pc += 1
        except:
            pass

        rmw_index += 1
        print(batch_x.shape)
    batch_x = model.layers["Flatten"].forward(batch_x,params)
    try:batch_x = model.layers["Toba"].forward(batch_x,params)
    except:pass



    
    for idx in range(1,rmw_layer[len(rmw_layer)-1]):  
        if idx not in rmw_layer:
            batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params)
            continue
        
        rmlist = []
        complist = []
        corlist = []
        alist = []
        blist = []
        
        rmlist_s = []
        complist_s = []
        alist_s = []
        blist_s = []
        
        out = []
        for i in batch_x:            #ループなしにしたい
            y = (i.reshape(-1,1))*params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #x_batch_y
        for i in range(0,len(out)-1):
            for j in range(i+1,len(out)):
                i_val = out[i]
                j_val = out[j]
                sxy = np.average(i_val*j_val) - np.average(i_val)*np.average(j_val)
                vari = (np.average(i_val**2)) - (np.average(i_val))**2
                varj = (np.average(j_val**2)) - (np.average(j_val))**2
                cor = sxy / (np.sqrt(vari)*np.sqrt(varj) + 0.00000001) #1e-10みたいな表記だとうまくいかないなぜ
                a = sxy/vari
                b = np.average(j_val) - a*np.average(i_val)
                
                corlist.append(abs(cor))
                rmlist.append(i)
                complist.append(j)
                alist.append(a)
                blist.append(b)                
        
        ziplist = zip(corlist,rmlist,complist,alist,blist)
        zipsort = sorted(ziplist,reverse=True)
        corlist,rmlist,complist,alist,blist = zip(*zipsort)
        cnt = 1
        while len(rmlist_s)<delete_n[rmw_index+idx-2]:
            if (rmlist[cnt] not in rmlist_s) and (rmlist[cnt] not in complist_s) and (corlist[cnt] > epsilon[rmw_index+idx-2]):
                rmlist_s.append(rmlist[cnt])
                complist_s.append(complist[cnt])
                alist_s.append(alist[cnt])
                blist_s.append(blist[cnt])
                print(corlist[cnt])
            cnt += 1
            if cnt == len(rmlist):break
                
        blist_s = np.array(blist_s)
        alist_s = np.array(alist_s)
        if complement:
            scalar = np.ones(len(params["W"+str(idx)]))
            for n in range(len(rmlist_s)):
                scalar[int(complist_s[n])] += alist_s[n]
            params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
            params["b"+str(idx)] = params["b"+str(idx)] + np.sum(blist_s)
            # params["b"+str(idx)] += np.sum(difflist,axis=0)
            
        if idx == 1:
            params["init_remove"].append(rmlist_s)
            params["W1"] = np.delete(params["W1"],rmlist_s,axis=0)
        else:
            params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist_s,axis=1)
            params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist_s)
            params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist_s,axis=0)
            
        print("    layer"+str(idx),": delete",str(len(rmlist_s))+"nodes")

        if idx == max(rmw_layer) : break
        batch_x = np.delete(batch_x,rmlist_s,axis=1)
        batch_x = model.layers["Affine"+str(idx)].forward(batch_x,params)
        batch_x = model.layers["Activation"+str(idx)].forward(batch_x,params) 
    return params


def half_predict(network,layer):
    x=netwotk.predict(

        
class Toba:
    def __init__ (self,model,x,tobaoption):
        self.model = model
        self.x = x
        
        self.params = copy.deepcopy(model.params)
        self.compare_nodes =eval(tobaoption[rmw_type])
        self.rmw_layer = tobaoption[rmw_layer]
        
    def rmw(self):
        for idx in self.rmw_layer:
            x = seld.half_predict(idx)
            rmlist, complist, scalar, bias = compare_nodes(x, idx, self.tobaoption)
            self.apply(idx,rmlist, complist, scalar, bias)
            
    def half_predict(self, stop_layer):
        a = self.model.predict(self.x,None,False,stop_layer)
        if stop_layer[0] == "C":
            layer = model.layers[stop_layer]
            pad = layer.P
            B,C,Ih,Iw = batch_x.shape
            F = self.params["F"+str(conv_index)]
            M,C,Fh,Fw = F.shape
            Oh = Ih + 2*pad -Fh + 1
            Ow = Iw + 2*pad -Fw + 1
            x = np.pad(batch_x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
            col = layer.im2col(x,B,C,Fh,Fw,Oh,Ow)
            col = col.reshape(C,Fh*Fw,B*Oh*Ow)
            F = np.transpose(F,(1,0,2,3)).reshape(C,M,-1)
            out = []
            for channel in range(C):
                y = np.dot(F[channel],col[channel]).reshape(-1)
                out.append(y)
        else :
            out = []
            for i in batch_x:            
                y = (i.reshape(-1,1))*self.params["W"+str(idx)]
                out.append(y)
            out=np.asarray(out)
            out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)
            
        return out
    
    def aplly(self, layer, dellst, complst, scalar, bias):
        if layer[0] == C:
            conv_index = layer[-1]
            self.params["F"+str(conv_index)] = self.params["F"+str(conv_index)] * (scalar.reshape(-1,1,1,1))
            self.params["Cb"+str(conv_index)] = self.params["Cb"+str(conv_index)] + bias
        
            self.params["F"+str(conv_index-1)] = np.delete(self.params["F"+str(conv_index-1)],rmlist_s,axis=0)
            self.params["Cb"+str(conv_index-1)] = np.delete(self.params["Cb"+str(conv_index-1)],rmlist_s)
            self.params["F"+str(conv_index)] = np.delete(self.params["F"+str(conv_index)],rmlist_s,axis=1)
        else:
            idx = layer[-1]
            self.params["W"+str(idx)] = self.params["W"+str(idx)] * (scalar.reshape(-1,1))
            self.params["b"+str(idx)] = self.params["b"+str(idx)] + bias
            
            self.params["W"+str(idx-1)] = np.delete(self.params["W"+str(idx-1)],rmlist_s,axis=1)
            self.params["b"+str(idx-1)] = np.delete(self.params["b"+str(idx-1)],rmlist_s)
            self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist_s,axis=0)

def corrcoef(out,layer,tobaoption):
    epsilon = tobaoption[epsilon]
    delete_n = tobaoption[epsilon]
    rmlist = complist = corlist = alist = blist = rmlist_s = complist_s = alist_s = blist_s = []
    
    for i in range(0,len(out)-1):
        for j in range(i+1,len(out)):
            i_val = out[i]
            j_val = out[j]
            sxy = np.average(i_val*j_val) - np.average(i_val)*np.average(j_val)
            vari = (np.average(i_val**2)) - (np.average(i_val))**2
            varj = (np.average(j_val**2)) - (np.average(j_val))**2
            cor = sxy / (np.sqrt(vari)*np.sqrt(varj) + 0.00000001) #1e-10みたいな表記だとうまくいかないなぜ
            a = sxy/vari
            b = np.average(j_val) - a*np.average(i_val)
                
            corlist.append(abs(cor))
            rmlist.append(i)
            complist.append(j)
            alist.append(a)
            blist.append(b)                
        
    ziplist = zip(corlist,rmlist,complist,alist,blist)
    zipsort = sorted(ziplist,reverse=True)
    corlist,rmlist,complist,alist,blist = zip(*zipsort)
    cnt = 1
    while len(rmlist_s)<delete_n[rmw_index+idx-2]:
        if (rmlist[cnt] not in rmlist_s) and (rmlist[cnt] not in complist_s) and (corlist[cnt] > epsilon[rmw_index+idx-2]):
            rmlist_s.append(rmlist[cnt])
            complist_s.append(complist[cnt])
            alist_s.append(alist[cnt])
            blist_s.append(blist[cnt])
            print(corlist[cnt])
        cnt += 1
        if cnt == len(rmlist):break
                
    blist_s = np.array(blist_s)
    alist_s = np.array(alist_s)
    scalar = np.ones(len(params[layer]))
    for n in range(len(rmlist_s)):
        scalar[int(complist_s[n])] += alist_s[n]
    bias = np.sum(blist_s)
    return rmlist, complist , scalar, bias

        
