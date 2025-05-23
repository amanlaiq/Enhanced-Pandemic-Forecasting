 #!/usr/bin/env python
# coding: utf-8

import time
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from math import ceil
from datetime import date, timedelta

import itertools
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import os

from models import MPNN
import json
# from preprocess import generate_new_features, generate_new_batches, read_meta_datasets, AverageMeter, generate_features
from llm_integration import generate_prompt, quantize_data  # Adjust the import if necessary
# from context_data_fetcher import fetch_contextual_data, enrich_features_with_context, process_text_data

from preprocess import (
    generate_new_features,
    generate_new_batches,
    read_meta_datasets,
    AverageMeter,
    generate_features,
)
from context_data_fetcher import generate_contextual_embeddings

def train( adj, features, y):
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(adj, features, y):    
    #output = model(adj, mob, ident)
    output = model(adj, features)
    loss_test = F.mse_loss(output, y)
    return output, loss_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    parser.add_argument('--meta-lr', default=0.01,#0.001,
                        help='')
    
    
    
    #---------------- Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    args = parser.parse_args()
    

        # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    country_keys = list(config['country_idx'].keys())  # Will give ["IT", "ES", "EN", "FR"]

    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window, config, country_keys)

    contextual_embeddings = generate_contextual_embeddings(
    country_keys,
    start_dates=[f"2020-{m:02d}-{d:02d}" for m, d in zip(config['country_start_month'], config['country_start_day'])],
    end_dates=[f"2020-{m:02d}-{d:02d}" for m, d in zip(config['country_end_month'], config['country_end_day'])]
)

    # Enrich each region's features in meta_features
    enriched_meta_features = []
    for i, region_features in enumerate(meta_features):
        enriched_region_features = [
            np.hstack((feature_array, contextual_embeddings[i].reshape(1, -1).repeat(feature_array.shape[0], axis=0)))
            for feature_array in region_features
        ]
        enriched_meta_features.append(enriched_region_features)

    combined_features = [
    np.vstack([np.hstack((core, context)) for core, context in zip(region_features, enriched_meta_features[i])])
    for i, region_features in enumerate(meta_features)
]

    # Normalize across all combined features for each region in the batch
    scaler = MinMaxScaler()
    normalized_features = [scaler.fit_transform(features) for features in combined_features]

    meta_features = enriched_meta_features

    for args.country in ["IT","ES","EN","FR"]:#,",

        if(args.country=="IT"):
            meta_train = [1,2,3]
            meta_test = 0

        elif(args.country=="ES"):
            meta_train = [0,2,3]
            meta_test = 1

        elif(args.country=="EN"):
            meta_train = [0,1,3]
            meta_test = 2
        else:
            meta_train = [0,1,2]
            meta_test = 3

        nfeat = meta_features[0][0].shape[1]

        model_eta = '../model_eta.pth.tar'
        model_theta = '../model_theta.pth.tar'

        
        #------------------------- 1 country, for 1 test sample and 1 shift, is 1 task
        #----------- tasks between targets
        
        norm_grad = 0
        for shift in list(range(0,args.ahead)):
            if 0 in meta_train:
                norm_grad += 63-args.start_exp-shift
            if 1 in meta_train:
                norm_grad += 47-args.start_exp-shift
            if 2 in meta_train:
                norm_grad += 48-args.start_exp-shift
            
        print("Meta Train")
        if not os.path.exists('../results'):
            os.makedirs('../results')
        fw = open("../results/results_"+args.country+"_tl_.csv","a")#results/
        fw.write("shift,loss,loss_std\n")
        
        #------ Initialize the model
        model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        torch.save({'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict(),}, model_eta)
        #pred_tables = {}
        
        for shift in list(range(0,args.ahead)):
            for train_idx in meta_train: # regularly reverse loop from train idx to shift
                labels = meta_labs[train_idx]
                gs = meta_graphs[train_idx]
                features = meta_features[train_idx]
                y = meta_y[train_idx]
                n_samples= len(gs)
                nfeat = features[0].shape[1]
                n_nodes = gs[0].shape[0]
            
             
                for test_sample in range(args.start_exp,n_samples - shift): 
                    idx_train = list(range(args.window-1, test_sample))
                    
                    adj_train, features_train, y_train = generate_new_batches(gs, features, y, idx_train, 1,shift,args.batch_size,device,test_sample)
                    
                    adj_test, features_test, y_test = generate_new_batches(gs, features, y,  [test_sample], 1, shift, args.batch_size, device,-1)

                    n_train_batches = ceil(len(idx_train)/args.batch_size)
                    n_test_batches = 1
                    
                    #-------------------- Meta-Train-Training
                    
                    #---- load eta
                    checkpoint = torch.load(model_eta) 
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    model.train()
                    train_loss = AverageMeter()

                    #------- Train for one epoch
                    for batch in range(n_train_batches):
                        output, loss = train(adj_train[batch], features_train[batch], y_train[batch])
                        train_loss.update(loss.data.item(), output.size(0))
                     
                    #----------- Backprop eta using the test sample
                    output, loss = train(adj_test[0], features_test[0], y_test[0])
                    print("meta train set "+str(train_idx)+" test sample "+str(test_sample)+" theta generalization=", '%03d'%loss.cpu().detach().numpy())
                 
			#------------ Take delta from the meta training 
                    w1 = model.conv1.lin.weight.grad.clone()
                    b1 = model.bn1.weight.grad.clone()
                    
                    w2 = model.conv2.lin.weight.grad.clone()
                    b2 = model.bn2.weight.grad.clone()
                    
                    f1 = model.fc1.weight.grad.clone()
                    f2 = model.fc2.weight.grad.clone()

                    model.eval()
                    

                    #----------- Update eta (one gradient per test sample)
                    checkpoint = torch.load(model_eta) 
                    model.load_state_dict(checkpoint['state_dict'])
                    model.conv1.lin.weight.data -= args.meta_lr*w1/norm_grad
                    model.bn1.weight.data -= args.meta_lr*b1/norm_grad
                    model.conv2.lin.weight.data -= args.meta_lr*w2/norm_grad
                    model.bn2.weight.data -= args.meta_lr*b2/norm_grad
                    model.fc1.weight.data -= args.meta_lr*f1/norm_grad
                    model.fc2.weight.data -= args.meta_lr*f2/norm_grad
                    
                    torch.save({
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            }, model_eta)

                    
        #-------------------------------------- Meta Test using pre-trained eta
        labels = meta_labs[meta_test]
        gs = meta_graphs[meta_test]
        features = meta_features[meta_test]
        y = meta_y[meta_test]
        nfeat = features[0].shape[1]
        #real_error = []
        n_samples = len(gs)
        result = []
        
        print("Meta Test")
        #pred_tables = {}
        for shift in list(range(0,args.ahead)):
            for test_sample in range(args.start_exp,n_samples-shift):#
                
                idx_train = list(range(args.window-1, test_sample-args.sep))
                idx_val = list(range(test_sample-args.sep,test_sample,2))
                idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))
                
                
                adj_train, features_train, y_train = generate_new_batches(gs, features, y, idx_train, 1, shift,args.batch_size,device,test_sample)
                adj_val, features_val, y_val = generate_new_batches(gs, features, y, idx_val, 1, shift,args.batch_size,device,test_sample)
                adj_test, features_test, y_test = generate_new_batches(gs, features, y,  [test_sample], 1, shift,args.batch_size, device,-1)

                n_train_batches = ceil(len(idx_train)/args.batch_size)
                n_val_batches = 1
                n_test_batches = 1 

                
                stop = False#
                stuck = False
                while(not stop):#
                    model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                    
                    #-------------------- Training
                    best_val_acc= 1e9
                    val_among_epochs = []

                    #----------------- Load the meta-trained model
                    if(not stuck):
                        checkpoint = torch.load(model_eta) 
                        model.load_state_dict(checkpoint['state_dict'])
                        #model.fc2 = nn.Linear(args.hidden, 1).to(device) # output layer still trained from scratch
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        #num_ftrs = model.fc2.in_features 

                        
                    for epoch in range(args.epochs):    
                        start = time.time()

                        model.train()
                        train_loss = AverageMeter()

                        # Train for one epoch
                        for batch in range(n_train_batches):
                            output, loss = train(adj_train[batch], features_train[batch], y_train[batch])
                            train_loss.update(loss.data.item(), output.size(0))

                        # Evaluate on validation set
                        model.eval()

                        output, val_loss = test(adj_val[0], features_val[0], y_val[0])
                        val_loss = int(val_loss.detach().cpu().numpy())


                        if(epoch%50==0):
                            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                        val_among_epochs.append(val_loss)

                        #--------- Stop if stuck                            
                        if(epoch<30 and epoch>10):
                            if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                                stuck= True
                                stop = False
                                break 
                                
                        if( epoch>args.early_stop):
                            if(len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):#
                                print("break")
                                stop = True 
                                break
                        stop = True

                        #--------- Remember best accuracy and save checkpoint
                        if val_loss < best_val_acc:
                            best_val_acc = val_loss
                            torch.save({
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            },model_theta) #------------------ store to meta_eta to share between test_samples

                        scheduler.step(val_loss)
                    
        
                #---------------- Testing
                test_loss = AverageMeter()

                checkpoint = torch.load(model_theta)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()

                output, loss = test(adj_test[0], features_test[0], y_test[0])
                o = output.cpu().detach().numpy()
                l = y_test[0].cpu().numpy()


                #------ Store to map plot                    
                error = np.sum(abs(o-l))
                df = pd.DataFrame({'n': labels.index,'o':o, 'l':l})
                if not os.path.exists('output'):
                    os.makedirs('output')
                df.to_csv("output/out_"+args.country+"_"+str(test_sample)+"_"+str(shift)+".csv",index=False) 

                n_nodes = adj_test[0].shape[0]
                #print(error/n_nodes)
                print(str(test_sample)+" "+args.country+" eta generalization=", "{:.5f}".format(error))
                result.append(error)
            
            print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(  np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))

            fw.write(str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")
        
        fw.close()

