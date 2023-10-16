import argparse
import os
import os.path as osp
import random
from unicodedata import digit
from spreadloss import SpreadLoss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import tool
import loss
import utils

@torch.no_grad()
def get_embedding(args, tgt_loader, model, cat_data=False, aug=False):
    model.eval()
    
    pred_bank = torch.zeros([len(tgt_loader.dataset), args.class_num]).cuda()
    emb_bank = torch.zeros([len(tgt_loader.dataset), args.bottleneck_dim]).cuda()

    for batch_idx, (data, target, idx) in enumerate(tgt_loader):
        data, target = data.cuda(), target.cuda()

        fea, out = model(data)
        emb_bank[idx] = fea
        pred_bank[idx] = out

    return pred_bank, emb_bank

def data_load_list(args, p_list, t_list):

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    weak_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    init_source_set = utils.ObjectImage('', args.s_dset_path, train_transform)
    source_set = utils.ObjectImage_list(p_list, train_transform)
    target_set = utils.ObjectImage_mul_list(t_list, [train_transform, weak_transform])
    test_set = utils.ObjectImage("", args.test_dset_path, test_transform)

    dset_loaders = {}
    dset_loaders["mid"] = torch.utils.data.DataLoader(
        source_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )

    dset_loaders["target"] = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )

    dset_loaders["test"] = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source"] = torch.utils.data.DataLoader(init_source_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=False)
    return dset_loaders

def list2txt(list, name):
    """save the list to txt file"""
    file = name     
    if os.path.exists(file):
        os.remove(file)
    for (path, label) in list:
        with open(file,'a+') as f:
            f.write(path+' '+ str(label)+'\n')

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=0.1, power=1.0):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss

@torch.no_grad()
def data_split(args, base_network):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_set = utils.ObjectImage_mul('', args.t_dset_path, train_transform)
    split_loaders = {}
    split_loaders["split"] = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.worker, drop_last=False)

    NUMS = args.midnum  # mid samples of each class

    if(args.skip_split):
        filename_e = './data/{}/mid_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS) 
        filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
        mid_path = utils.make_dataset('', filename_e)
        hard_path = utils.make_dataset('', filename_h)
        print('load txt from ' + filename_e + ' and ' + filename_h )             
        args.out_file.write('load txt from ' + filename_e + 'and' + filename_h  + '\n')             
        args.out_file.flush()
    else:
        mid_path, hard_path, mid_idx, hard_idx = [], [], [], []

        base_network.eval()
        """ the full (path, label) list """
        img = utils.make_dataset('', args.t_dset_path)

        # with torch.no_grad():
        """ extract the prototypes """
        for name, param in base_network.named_parameters():
            if('fc.weight' in name):
                prototype = param

        _, features_bank = get_embedding(args, split_loaders["split"], base_network)
        features_bank = F.normalize(features_bank) # len * 256
        prototype = F.normalize(prototype) # cls * 256
        dists = prototype.mm(features_bank.t())  # cls * len

        sort_idxs = torch.argsort(dists, dim=1, descending=True) #cls * len
        fault = 0.

        for i in range(args.class_num):
            ## check if the repeated index in the list
            for _ in range(NUMS):
                idx = sort_idxs[i, s_idx]

                while idx in mid_idx:
                    s_idx += 1
                    idx = sort_idxs[i, s_idx]

                assert idx not in mid_idx

                mid_idx.append(idx)
                mid_path.append((img[idx][0], i))

                if not img[idx][1] == i:
                    fault += 1
                s_idx += 1


        for id in range(len(img)):
            if id not in mid_idx:
                hard_path.append(img[id])
                hard_idx.append(id)


        acc = 1 - fault / (args.class_num*NUMS)

        print('Splited data list label Acc:{}'.format(acc))
        args.out_file.write('Splited data list label Acc:{}'.format(acc) + '\n')
        args.out_file.flush()

        if args.save_middle:
            filename_e = './data/{}/mid_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
            filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
            list2txt(mid_path, filename_e)
            list2txt(hard_path, filename_h)
            print('Splited data list saved in ' + filename_e + ' and ' + filename_h )
            args.out_file.write('Splited data list saved in ' + filename_e + 'and' + filename_h  + '\n')
            args.out_file.flush()

    return  mid_path, hard_path

def KLD(sfm, sft):
    return -torch.mean(torch.sum(sfm.log() * sft, dim=1))

def momentum_update_key_encoder(netG, net_G_tea):
    moco_m = 0.999
    for param_q, param_k in zip(netG.parameters(), net_G_tea.parameters()):
        param_k.data = param_k.data * moco_m + param_q.data * (1 - moco_m)

def mse_loss(pred, target):
    N = pred.size(0)
    pred_norm = nn.functional.normalize(pred, dim=1)
    target_norm = nn.functional.normalize(target, dim=1)
    loss = 1 - 2 * (pred_norm * target_norm).sum() / N
    return loss

def train(args):

    class_num = args.class_num
    class_weight_src = torch.ones(class_num, ).cuda()

    if args.net == 'resnet101':
        netG = utils.ResBase101().cuda()
        net_G_tea = utils.ResBase101().cuda()
    elif args.net == 'resnet50':
        netG = utils.ResBase50().cuda()  
        net_G_tea = utils.ResBase50().cuda()

    elif args.net == 'resnet34':
        netG = utils.ResBase34().cuda()  

    netF = utils.ResClassifier(class_num=class_num, feature_dim=netG.in_features, 
        bottleneck_dim=args.bottleneck_dim, type = args.cls_type, ltype = args.layer_type).cuda()
    netP = utils.ResProjector(feature_dim=netG.in_features, head = 'mlp').cuda()  #投影层    
    base_network = nn.Sequential(netG, netF)

    for name, param in base_network.named_parameters():
        if('fc' in name):
            param.requires_grad = False

    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr )    
    optimizer_f = optim.SGD(netF.parameters(), lr = args.botlr )
    optimizer_p = optim.SGD(netP.parameters(), lr = args.lr )


    for name, param in netF.named_parameters():
        if not param.requires_grad:
            print(name + ' fixed.')
            args.out_file.write(name + ' fixed.' + '\n')
            args.out_file.flush()

    base_network.load_state_dict(torch.load(args.ckpt))
    for param_q, param_k in zip(netG.parameters(), net_G_tea.parameters()): #tea网络参数初始化
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    print('load source model from '+ args.ckpt + '\n')
    args.out_file.write('load source model from '+ args.ckpt + '\n')
    args.out_file.flush()

    mid_path, hard_path = data_split(args, base_network)

    ## set dataloaders

    dset_loaders = data_load_list(args, mid_path, mid_path + hard_path)
    max_len =  len(dset_loaders["target"])
    args.max_iter = args.max_epoch * max_len
    # eval_iter = args.val_num

    ## Memory Bank
    if args.pl.endswith('mem'):
        mem_fea = torch.rand(len(dset_loaders["target"].dataset), args.bottleneck_dim).cuda()
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    middle_loader_iter = iter(dset_loaders["middle"])
    target_loader_iter = iter(dset_loaders["target"])

    list_acc = []
    best_ent = 100
    c_n = 0

    for iter_num in range(1, args.max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr , iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.botlr  , iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_p, init_lr=args.lr  , iter_num=iter_num, max_iter=args.max_iter)
        
        if c_n < args.cycle_num:
            if iter_num % int(args.eval_epoch*max_len*args.cycle_ratio) == 0: # 迭代进行
                mid_path, hard_path = data_split(args, base_network)
                dset_loaders = data_load_list(args, mid_path, mid_path + hard_path)
                middle_loader_iter = iter(dset_loaders["middle"])
                target_loader_iter = iter(dset_loaders["target"])
                c_n = c_n + 1

        try:
            inputs_middle, labels_middle = middle_loader_iter.next()
        except:
            middle_loader_iter = iter(dset_loaders["middle"])
            inputs_middle, labels_middle = middle_loader_iter.next()
        inputs_middle, labels_middle = inputs_middle.cuda(),  labels_middle.cuda()

        try:
            inputs_target_all, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target_all, _, idx = target_loader_iter.next()

        inputs_target = inputs_target_all[0].cuda()
        inputs_target_aug = inputs_target_all[1].cuda()

        features_target, outputs_target = base_network(inputs_target)

        total_loss = torch.tensor(0.).cuda()

        eff = iter_num / args.max_iter

        if args.CIDCVCL_ratio:
            _, outputs_source = base_network(inputs_middle)

            src_ = loss.CrossEntropyLabelSmooth(reduction=None,num_classes=class_num, epsilon=args.smooth)(outputs_source, labels_middle)

            weight_src = class_weight_src[labels_middle].unsqueeze(0)
            src_cls = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
            # print('src:',src_cls)
            total_loss += src_cls * args.CIDCVCL_ratio


        softmax_out = nn.Softmax(dim=1)(outputs_target)

        if args.pl.endswith('mem'):
            dis = -torch.mm(features_target.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)

            w = torch.zeros(features_target.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(args.K):
                    w[wi][p1[wi, wj]] = 1/ args.K

            sft_label = w.mm(mem_cls)

        else:
            raise RuntimeError('pseudo label error')



        mix_cls_loss = torch.tensor(0.).cuda()

        if args.mix_ratio:
            rho = np.random.beta(args.alpha, args.alpha)
            mix_img = inputs_target * rho + inputs_middle*(1-rho)

            _, mix_out = base_network(mix_img)
            weight_src = class_weight_src[labels_middle].unsqueeze(0)

            targets_s = torch.zeros(args.batch_size, args.class_num).cuda().scatter_(1, labels_middle.view(-1,1), 1)
            
            mix_target = sft_label * rho + targets_s * (1-rho)
            mix_cls_loss += eff * (KLD(nn.Softmax(dim=1)(mix_out), mix_target) )

        remix_reg_loss = torch.tensor(0.).cuda()


        if args.CVCL_ratio:
            inputs_t = inputs_target
            inputs_t2 = inputs_target_aug

            # 得到q和ek,分别来自G的两个不同view
            q, _ = netP(netG(inputs_t)) # q, feat_q
            ek, _ = netP(netG(inputs_t2)) # ek, feat_ek

            with torch.no_grad():  # no gradient 
                momentum_update_key_encoder(netG, net_G_tea)  # update the momentum encoder
                eq, feat_eq = netP(net_G_tea(inputs_t))
                k, feat_k = netP(net_G_tea(inputs_t2))


            # f1_loss = mse_loss(q.squeeze(-1), feat_k.clone().detach().squeeze(-1))
            # f2_loss = mse_loss(ek.squeeze(-1), feat_eq.clone().detach().squeeze(-1))
            # f_loss = f1_loss + f2_loss
            f1_loss = mse_loss(q.squeeze(-1), k.clone().detach().squeeze(-1))
            f2_loss = mse_loss(ek.squeeze(-1), eq.clone().detach().squeeze(-1))
            f_loss = f1_loss + f2_loss
            # print('f:',f_loss)

            total_loss += f_loss * args.CVCL_ratio

        
        if args.spread_ratio:
            spread = SpreadLoss(256, args.class_num, mem_fea)
            spread_loss = spread(features_target, idx)
            if iter_num % int(args.eval_epoch*max_len) == 0:
                print(spread_loss)
            total_loss += args.spread_ratio * spread_loss
        
        if args.CVCL2_ratio:
            # features_target_s, outputs_target_s = base_network(inputs_target_aug)
            prob_tu_w = torch.softmax(outputs_target, dim=1)
            # prob_tu_s = torch.softmax(outputs_target_s, dim=1)
            # with torch.no_grad():
            features_target_s, outputs_target_s = netF(net_G_tea(inputs_target_aug))
            prob_tu_s = torch.softmax(outputs_target_s, dim=1)

            L_con_cls = contras_cls(prob_tu_w, prob_tu_s)

            if args.ot_ratio:
                target_source, _ = base_network(inputs_middle)
                ot_loss = ot_loss_2(target_source, features_target, features_target_s, args)
                print('ot_loss:',ot_loss)
                total_loss += args.ot_ratio * ot_loss
                
            if iter_num % int(args.eval_epoch*max_len) == 0:
                print(L_con_cls)
            total_loss += args.CVCL2_ratio * L_con_cls
            
            
        total_loss += mix_cls_loss * args.mix_ratio + remix_reg_loss * args.reg_ratio  

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_p.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        optimizer_p.step()

        
        if args.pl.endswith('mem'):
            base_network.eval() 
            with torch.no_grad():
                features_target, outputs_target = base_network(inputs_target)
                features_target = features_target / torch.norm(features_target, p=2, dim=1, keepdim=True)
                softmax_out = nn.Softmax(dim=1)(outputs_target)

                if args.pl == 'fw_na':
                    # print(3)
                    sfm_out = softmax_out ** 2 / softmax_out.sum(dim=0)
                    outputs_target = sfm_out / sfm_out.sum(dim=1, keepdim=True)

                else:
                    raise RuntimeError('pseudo label error')

            mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
            mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if iter_num % int(args.eval_epoch*max_len) == 0:
            base_network.eval()
            if args.dset == 'VISDA-C':
                acc, py, score, y, tacc = utils.cal_acc_visda(dset_loaders["test"], base_network)
                args.out_file.write(tacc + '\n')
                args.out_file.flush()

                _ent = loss.Entropy(score)
                mean_ent = 0
                for ci in range(args.class_num):
                    mean_ent += _ent[py==ci].mean()
                mean_ent /= args.class_num

            else:
                acc, py, score ,y = utils.cal_acc(dset_loaders["test"], base_network)
                # acc, py, score, y = utils.cal_acc_tsne(dset_loaders["test"], base_network)
                mean_ent = torch.mean(loss.Entropy(score))

            list_acc.append(acc * 100)
            if best_ent > mean_ent:
                best_ent = mean_ent
                val_acc = acc * 100
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num, args.max_iter, acc*100, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
    
    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()

    torch.save(base_network.state_dict(), osp.join(args.output_dir, args.log + ".pt"))

    return best_y.cpu().numpy().astype(np.int64)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation Methods')
    parser.add_argument('--method', type=str, default='srconly')
    parser.add_argument('--dset', type=str, default='office-home', choices=['IMAGECLEF', 'VISDA-C', 'office', 'office-home','DomainNet126'], help="The dataset or source dataset used")

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--tname', type=str, default=None, help="target")
    parser.add_argument('--Nofinetune', action='store_true')
    parser.add_argument('--test_on_src', action='store_true')
    parser.add_argument('--pl', type=str, default='fw_na',choices=['mixmatch', 'fw', 'mixmatch_na','remixmatch_na','fw_na','atdoc_na'])
    parser.add_argument('--split', type=str, default='proto',choices=['proto', 'ent','rand'])

    parser.add_argument('--mix_ratio', type=float, default=1)
    parser.add_argument('--reg_ratio', type=float, default=0)
    parser.add_argument('--CIDCVCL_ratio', type=float, default=0.01)
    parser.add_argument('--CVCL_ratio', type=float, default=1)
    parser.add_argument('--im_ratio', type=float, default=0)
    parser.add_argument('--spread_ratio', type=float, default=0)
    parser.add_argument('--CVCL2_ratio', type=float, default=0.1)
    parser.add_argument('-ot_ratio', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--cycle_ratio', type=float, default=0)
    parser.add_argument('--cycle_num', type=float, default=0)

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_epoch', type=str, default=None)
    parser.add_argument('--output', type=str, default='tmp/',required=True)

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")

    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument("--lambda_u", default=100, type=float)

    parser.add_argument('--midnum', type=int, default=5) 
    parser.add_argument('--skip_split', action='store_true')
    parser.add_argument('--save_middle', action='store_true')
    
    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet50", "resnet101",'resnet34'])
    parser.add_argument('--cls_type', type=str, default='ori')
    parser.add_argument('--layer_type', type=str, default='linear')
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--botlr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=1)

    args = parser.parse_args()
    args.output = 'logs/'+ args.output
    args.output = args.output.strip()

    args.momentum = 1.0

    args.eval_epoch = args.max_epoch / 100

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'IMAGECLERF':
        names = ['c','i','p']
        args.class_num = 12


    if(args.ckpt is None):
        if(args.ckpt_epoch is not None):
            args.ckpt = './logs/source_' + args.ckpt_epoch + '/' + args.dset + '/' + names[args.s][0].upper() + names[args.s][0].upper() + '/srconly.pt' 
        else:
            args.ckpt = './logs/source/' + args.dset + '/' + names[args.s][0].upper() + names[args.s][0].upper() + '/srconly.pt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.s_dset_path = './data/' + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = './data/' + args.dset + '/' + names[args.t] + '_list.txt'

    if(args.tname is not None):
        args.t_dset_path = './data/' + args.dset + '/' + args.tname + '_list.txt'
    args.test_dset_path = args.t_dset_path
    args.sname = names[args.s]

    if(args.test_on_src):
        args.test_dset_path = args.s_dset_path
        args.t_dset_path = args.s_dset_path

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.log = args.method
    args.out_file = open(osp.join(args.output_dir, "{:}.txt".format(args.log)), "w")

    utils.print_args(args)
    
    label = train(args)
    
