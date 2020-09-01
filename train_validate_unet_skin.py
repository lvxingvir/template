# Modified for using ynet to seg the skin and classify.
# v0: only unet for skin.


import os
from torch import optim
from time import gmtime, strftime
from tensorboardX import SummaryWriter

os.sys.path.append(r'C:\Users\Xing\Projects\SDFY\mammo\projects\SDFY_project')
from SDFY_project.unet_model.unet_mass_light_v2 import *   # for customed unet
# from SDFY_project.efficientunet.efficientynet import *   # for efficientynet
from SDFY_project.uneunet_factoryt_factory.unet import * # for unet factory
from SDFY_project.data_generator_skin_ynet import *
from SDFY_project.tools import *
from lr_cosine import CosineAnnealingWarmUpRestarts
from SDFY_project.sampler import BalancedBatchSampler
from SDFY_project.losses import dice_loss
from SDFY_project.Unet_metrics.unet_metircs import *

from SDFY_project.auto_augment.auto_augment import AutoAugment, Cutout

import warnings
warnings.filterwarnings(action='ignore')



if __name__ == '__main__':
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    # result_path = r'E:\Data\HeronTech\SDFY\ori_marigin_crop\model_train'
    result_path = r'E:\Xing\SDFY_skin\Train_log'
    descrip = 'July20_resunext50_light_384_imgaug'
    model_save_path = os.path.join(result_path, descrip, time_string, 'save')
    tb_save_path = os.path.join(result_path, descrip,time_string, 'tb')
    os.makedirs(model_save_path)
    os.makedirs(tb_save_path)
    writer = SummaryWriter(log_dir=tb_save_path)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    # torch.cuda.set_device(1)

    # model = get_unetx_reduce_fpn_v1().cuda()
    # model = resnet.resnet10(in_channels=8, drop_rate=0.3, sample_size=64, sample_duration=16, shortcut_type='B', num_classes=1).cuda()
    # model = resnet_v2.resnet10(in_channels=8, drop_rate=0.3, sample_size=64, sample_duration=16, shortcut_type='B',
    #                         num_classes=1).cuda()
    num_classes = 1
    # model = ResNet(dataset='calc', depth=50, num_classes=num_classes).cuda()
    # model = UNet(n_channels=3,n_classes=num_classes,height=384,width=384).cuda()
    # model = get_efficientynet_b0(out_channels=1, n_classes=num_classes, concat_input=True, pretrained=True).cuda()

    model = unet_resnet(resnet_type='resnext50_32x4d',in_channels=3,out_channels=1).cuda()

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model).cuda()

    criterion = nn.BCELoss().cuda()
    # criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=0.01, factor=0.3)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=1e-3, T_up=1, gamma=0.5)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AutoAugment(),
        transforms.ToTensor()
    ])

    trainconfig = {"dataset": 'mammo_calc',"subset": '0'}
    train_config = dataconfig(**trainconfig)
    # training_data = DataGenerator(train_config,transform= train_transform)
    training_data = DataGenerator(train_config, transform= transforms.ToTensor())
    # train_loader = DataLoader(training_data, num_workers=4, batch_size=16, shuffle= True)
    train_loader = DataLoader(training_data,num_workers=4,
                              sampler=BalancedBatchSampler(training_data, type='single_label'),
                              batch_size = 6, drop_last= True)

    valconfig = {"dataset": "calc","subset": '1'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config,transform= transforms.ToTensor())
    val_loader = DataLoader(validation_data, num_workers=4,shuffle=True)

    print('data loader finished')

    Train_C_flag = False
    epoch_len = 200

    bst_acc = 0
    bst_loss = 1e10
    bst_tsh = 0.1

    if Train_C_flag == True:
        model_load_path = r'E:\Xing\mass0508\Train_log\June12_ynet_light_bridgel\Fri12Jun2020-204633\save'
        model_name = r'\best_model.pth'

        checkpoint = torch.load(model_load_path + model_name)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        Epoch = checkpoint['epoch']
    else:
        Epoch = 0


    for epoch in range(Epoch,Epoch+epoch_len):
        model.train()
        losses = AverageMeter()
        losses_dice = AverageMeter()
        losses_cls = AverageMeter()
        accuracies = AverageMeter()

        for i, (images,masks,labels) in enumerate(train_loader):
            targets = labels.cuda()
            masks = masks.cuda()

            est_maps = model(images.cuda())
            # print('outputs: ', outputs.data.cpu().numpy().tolist(), 'targets: ', targets.data.cpu().numpy().tolist())
            est_maps = F.sigmoid(est_maps)
            est_tags = torch.ones(labels.shape[0]).cuda()
            loss_dice = dice_loss(est_maps, masks)

            loss_cls = criterion(est_maps,masks)

            loss = 0.5*loss_dice + 0.5*loss_cls

            acc = calculate_accuracy_binary(est_tags, targets)
            losses.update(loss.item(), targets.size(0))
            losses_dice.update(loss_dice.item(), targets.size(0))
            losses_cls.update(loss_cls.item(), targets.size(0))
            accuracies.update(acc, targets.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if len(est_maps.shape) >3:
                est_maps = torch.squeeze(est_maps,1)
            if (epoch) % 50==0 and i % 20 == 0:
                # _, predict = torch.max(est_tags, 1)
                predict = est_tags
                add_image_3d(images, predict, targets, writer, subset='train', epoch=epoch, name= str(i)+'_image')
                add_image_3d(torch.unsqueeze(masks,1), predict, targets, writer, subset='train', epoch=epoch,name = str(i)+'_masks')
                add_image_3d(torch.unsqueeze(est_maps,1), predict, targets, writer, subset='train', epoch=epoch,name = str(i)+'_est_maps')

            #print(i,loss)

        losses_val = AverageMeter()
        losses_val_dice = AverageMeter()
        losses_val_cls = AverageMeter()
        accuracies_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for j, (inputs_val,masks_val, targets_val) in enumerate(val_loader):
                targets_val = targets_val.cuda()
                masks_val = masks_val.cuda()
                outputs_val = model(inputs_val.cuda())
                outputs_val = F.sigmoid(outputs_val)

                est_tags_val = torch.ones(targets_val.shape[0]).cuda()

                loss_dice_val = dice_loss(outputs_val, masks_val)
                loss_cls_val = criterion(outputs_val, masks_val)
                loss_val = 0.5*loss_dice_val  + 0.5*loss_cls_val

                acc_val = calculate_accuracy_binary(est_tags_val, targets_val)
                losses_val.update(loss_val.item(), targets_val.size(0))
                losses_val_dice.update(loss_dice_val.item(), targets_val.size(0))
                losses_val_cls.update(loss_cls_val.item(), targets_val.size(0))
                accuracies_val.update(acc_val, targets_val.size(0))
                if len(outputs_val.shape) > 3:
                    outputs_val = torch.squeeze(outputs_val,1)
                if (epoch ) % 20 == 0 and j % 10 == 0:
                    print(j, loss_val)
                    # _,predict = torch.max(est_tags_val,1)
                    predict = est_tags_val
                    add_image_3d(inputs_val, predict, targets_val, writer, subset='val', epoch=epoch, name = str(j)+'_images')
                    add_image_3d(torch.unsqueeze(masks_val,1), predict, targets_val, writer, subset='val', epoch=epoch, name = str(j)+'_masks')
                    add_image_3d(torch.unsqueeze(outputs_val,1), predict, targets_val, writer, subset='val', epoch=epoch, name = str(j)+'_est_maps')
        # scheduler.step(losses_val.avg)
        scheduler.step()

        print('epoch: ', epoch+1, 'train_loss: ', losses.avg, 'train_acc: ', accuracies.avg,
              'val_loss: ', losses_val.avg, 'val_acc: ', accuracies_val.avg)


        # if bst_loss >= losses_val.avg or abs(bst_loss - losses_val.avg)<=bst_tsh:
        if bst_loss >= losses_val.avg:
            # if bst_acc <= accuracies_val.avg:

            save_file_path = os.path.join(model_save_path, 'best_model.pth')
            states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(states, save_file_path)
            better_epoch = epoch

            bst_acc = accuracies_val.avg
            bst_loss = losses_val.avg

        print('better model found at epoch {} with val_loss {} and val_acc {}'.format(better_epoch,bst_loss,bst_acc))

        # Save model and print something in the tensorboard
        # Save model and print something in the tensorboard
        writer.add_scalars('loss/epoch',
                           {'train loss': losses.avg, 'validation loss': losses_val.avg,
                            'train dice loss':losses_dice.avg,'valid dice loss':losses_val_dice.avg,
                            'train cls loss':losses_cls.avg,'valid cls loss':losses_val_cls.avg,}, epoch + 1)
        # writer.add_scalars('acc/epoch',
        #                    {'train accuracy': accuracies.avg, 'validation accuracy': accuracies_val.avg}, epoch + 1)
        writer.add_scalars('Learning Rate/epoch',
                           {'train accuracy': optimizer.param_groups[0]['lr']}, epoch + 1)

        # if (epoch + 1) % 10 == 0:
        #     save_file_path = os.path.join(model_save_path, 'save_{}.pth'.format(epoch+1))
        #     states = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(states, save_file_path)


        # if (epoch + 1) % 50 == 0:
        #     add_image(inputs, outputs, targets, names, writer, subset='train', epoch=epoch + 1)


