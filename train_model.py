import torch
import torch.nn.functional as F
import os
from models import Siren, weights_init, HashGridINR, ResSiren, SkipSiren, SirenModulated
from utils import multilabel_sdm, evaluate, visualize_differences
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import json



import argparse
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='INR-SSM')

    #Data
    parser.add_argument('--dataset', type=str, default='JSRT', help='Select dataset (JSRT, RING, DISC or path to custom dataset)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to use. Default: -1 (all)')
    
    #Training
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_epochs', type=int, default=10, help='Validation epochs')
    parser.add_argument('--save_epochs', type=int, default=3000, help='Save epochs')
    parser.add_argument('--seg', action='store_true', help='Directly fit segmentation instead of SDM')

    #INR Model
    parser.add_argument('--model', type=str, default='Siren', help='Model type')
    parser.add_argument('--model_weights_init', type=int, default=20, help='INR Initialization scale')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')

    #INR Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='INR Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')

    #Codebook
    parser.add_argument('--codebook_sz', type=int, default=14, help='Codebook size')
    parser.add_argument('--codebook_lr', type=float, default=1e-3, help='Codebook learning rate')
    parser.add_argument('--codebook_optimizer', type=str, default='Adam', help='Codebook optimizer')
    parser.add_argument('--codebook_scheduler', type=str, default='LambdaLR', help='Codebook scheduler')
    parser.add_argument('--codebook_warmup', type=int, default=0, help='Codebook warmup')

    #Misc
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--seed', type=int, default=None, help='Seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    parser.add_argument('--version', type=str, default='v0')


    args = parser.parse_args()

    #set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #initialize wandbsave
    if args.no_wandb:
        wandb.init(project='INR-SSM',config=vars(args), mode='disabled')
    else:
        wandb.init(project='INR-SSM',config=vars(args))


    #define experiment name based wandb id
    rearrange_string = lambda s: '-'.join(s.split('-')[-1:] + s.split('-')[:-1])
    exp_name = rearrange_string(wandb.run.name)
    print('Experiment:',exp_name)

    args.save_path = os.path.join(args.save_path,exp_name)
    os.makedirs(args.save_path,exist_ok=True)




    ### PREPARE DATA
    if args.dataset == 'JSRT':
        data = torch.load('data/JSRT_img0_lms_seg_shifted.pth')
        #img0 = data['JSRT_img0']
        #seg0 = data['JSRT_seg']
        #lms0 = data['JSRT_lms']
        seg = data['JSRT_seg_shifted']
        #translations = data['translations']
    elif args.dataset == 'RING':
        seg = torch.load('data/ring_example_img.pth')
    elif args.dataset == 'DISC':
        seg = torch.load('data/disc_example_img.pth')
    else:
        seg = torch.load(args.dataset) # should be a tensor of shape (N,C,H,W)
    
    if args.num_samples > 0:
        seg = seg[:args.num_samples]
        print('Using only the first ',args.num_samples,'samples of dataset.')

    N,C,H,W = seg.shape
    args.C = C
        #save args as json
    with open(f'{args.save_path}/args.json', 'w') as f:
        json.dump(vars(args), f)


    #Create SDMs
    
    ml_sdm = []

    if not args.seg:
        for i in trange(seg.shape[0]):
            ml_sdm.append(multilabel_sdm(seg[i]))
        ssms = torch.stack(ml_sdm).permute(0,2,3,1).view(N,-1,C)
        print('SDMs created')
    else:
        for i in trange(seg.shape[0]):
            ml_sdm.append(seg[i]*2.0-1.0)
        ssms = torch.stack(ml_sdm).permute(0,2,3,1).view(N,-1,C)
        print('Directly use segmentations instead of SDM')


    #initialize model
    if args.model == 'Siren':
        model = Siren(in_features=2+args.codebook_sz,out_features=C, hidden_ch=args.channels,num_layers=args.layers)
        model.apply(partial(weights_init, scale=args.model_weights_init))
        model.to(device)
    elif args.model == 'Hashgrid':
        model = HashGridINR(in_features=2+args.codebook_sz,out_features=C,H=H, hidden_ch=args.channels,num_layers=args.layers)
        model.to(device)
    elif args.model == 'ResSiren':
        model = ResSiren(in_features=2+args.codebook_sz,out_features=C, hidden_ch=args.channels,num_layers=args.layers)
        model.apply(partial(weights_init, scale=args.model_weights_init))
        model.to(device)
    elif args.model == 'SkipSiren':
        model = SkipSiren(in_features=2+args.codebook_sz,out_features=C, hidden_ch=args.channels,num_layers=args.layers)
        model.apply(partial(weights_init, scale=args.model_weights_init))
        model.to(device)
    elif args.model == 'SirenModulated':
        model = SirenModulated(in_features=2+args.codebook_sz,out_features=C, hidden_ch=args.channels,num_layers=args.layers)
        model.apply(partial(weights_init, scale=args.model_weights_init))
        model.to(device)
    else:
        raise NotImplementedError

    #initialize codebook
    latent_codes = torch.randn(N, args.codebook_sz).to(device)*1e-8
    latent_codes.requires_grad = True
 
    ##itialize optimizer
    if args.optimizer == 'Adam':
        optimizer_model = torch.optim.Adam(model.parameters(),lr=args.lr)
    else:
        raise NotImplementedError
    
    if args.codebook_optimizer == 'Adam':
        optimizer_codes = torch.optim.Adam([latent_codes],lr=args.codebook_lr)
    else:
        raise NotImplementedError
    
    if args.codebook_scheduler == 'LambdaLR':
        if args.codebook_warmup > 0:
            code_scheduler_lambda = lambda epoch: 1 if epoch>args.displacement_warmup else epoch/args.displacement_warmup
        else:
            code_scheduler_lambda = lambda epoch: 1 if epoch>(args.epochs//4) else epoch/(args.epochs//4)
            
        code_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_codes, lr_lambda=code_scheduler_lambda)
    else:
        raise NotImplementedError
    


    #Define Dataset
    dataset = TensorDataset(ssms, latent_codes)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    mesh = F.affine_grid(torch.eye(2,3).unsqueeze(0).cuda(),(1,1,H,W), align_corners=True)


    ##TRAINING
    losses = []

    for epoch in trange(args.epochs):
        epoch_loss = 0.0

        for batch in data_loader:  
            optimizer_model.zero_grad()
            optimizer_codes.zero_grad()

            ssms_batch, latent_codes_batch = batch
            in_loop_batch_sz = ssms_batch.shape[0]
        
            mesh_batch = mesh.repeat(in_loop_batch_sz,1,1,1).view(in_loop_batch_sz,-1,2).to(device)
            input_ = torch.cat([mesh_batch, latent_codes_batch.unsqueeze(1).repeat(1,H*W,1)], dim=-1)
            out = model(input_)
            #loss = F.mse_loss(torch.tanh(out), ssms_batch.to(device))
            loss = F.mse_loss(out, ssms_batch.to(device))
            loss.backward()
            optimizer_model.step()
            optimizer_codes.step()
            code_scheduler.step()

            epoch_loss += loss.item()
        losses.append(epoch_loss)
        wandb.log({"loss": epoch_loss})




        if epoch % args.val_epochs == 0 or epoch == args.epochs-1:
            tqdm.write(f'Epoch {epoch} Loss: {epoch_loss}')
            model.eval()
            with torch.no_grad():
                out = model(torch.cat([mesh[0].view(-1,2), latent_codes[0].unsqueeze(0).repeat(H*W,1)], dim=-1)).cpu().detach()
                diff = visualize_differences(out.view(H,W,-1)<0, ssms[0].view(H,W,-1)<0)
                wandb.log({"diff": [wandb.Image(diff.squeeze(), caption=f"Diff at epoch {epoch}")]})


        if epoch % args.save_epochs == 0 or epoch == args.epochs-1:

            if epoch > 0:
                torch.save({'model':model.state_dict(),
                            'codebook':latent_codes.detach().cpu(),
                            'optimizer_model':optimizer_model.state_dict(),
                            'optimizer_codes':optimizer_codes.state_dict(),
                            'epoch':epoch,
                            'losses':losses},
                            f'{args.save_path}/model_{epoch}.pth')

    print('Training finished')
    ##EVALUATION
    HDs, dices = evaluate(model, mesh, latent_codes, ssms, H, W, C, mode='greater' if args.seg else 'less')
    #save results
    torch.save({'HDs':HDs,
                'dices':dices},
                f'{args.save_path}/result_metrics.pth')
    print('HDs:',HDs.mean(0))
    print('Dices:',dices.mean(0))
    print('Comment:',args.comment)
    wandb.log({"HDs": HDs.mean(0), "dices": dices.mean(0), "model": args.model, "seg": args.seg, "comment": args.comment})
    #disconnect wandb
    wandb.finish()