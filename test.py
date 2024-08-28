import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import *
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score

from nets.tacunet.TAC_UNet import TACUNet


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    pre_pred = precision_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    recall_pred = recall_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    acc_pred = accuracy_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    f1_score_pred = f1_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    fpr, tpr, _ = roc_curve(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))

    # pre_pred = precision_score()
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(2000,2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    return [dice_pred, iou_pred, pre_pred, recall_pred, f1_score_pred, acc_pred], fpr, tpr


def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    metrics_li, fpr, tpr = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return metrics_li, fpr, tpr



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name == "GlaS":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    elif config.task_name == "cvcdb":
        test_num = 61
        model_type = config.model_name
        # model_path = "./CSF/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
        model_path = "/root/autodl-tmp/weights/cvcdb/unet/Test_session_08.28_11h35/models/best_model-unet1.pth.tar"


    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'TAC_UNet':
        config_vit = config.get_config_tacunet()
        model = TACUNet(config_vit)

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_li = list()
    iou_li = list()
    pre_li = list()
    recall_li = list()
    acc_li = list()
    f1_score_li = list()
    fpr_li = list()
    tpr_li = list()

    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            metrics_li, fpr, tpr = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i))

            dice_li.append(metrics_li[0])
            iou_li.append(metrics_li[1])
            pre_li.append(metrics_li[2])
            recall_li.append(metrics_li[3])
            acc_li.append(metrics_li[5])
            fpr_li.append(fpr)
            tpr_li.append(tpr)

            torch.cuda.empty_cache()
            pbar.update()

    avg_dice = np.mean(dice_li, axis=0)
    avg_iou = np.mean(iou_li, axis=0)
    avg_pre = np.mean(pre_li, axis=0)
    avg_recall = np.mean(recall_li, axis=0)
    avg_acc = np.mean(acc_li, axis=0)
    avg_fpr = np.mean(fpr_li, axis=0)
    avg_tpr = np.mean(tpr_li, axis=0)

    # Calculate AUC for this network
    roc_auc = auc(avg_fpr, avg_tpr)

    print("dice_pred", avg_dice)
    print("iou_pred", avg_iou)
    print("precision", avg_pre)
    print("recall", avg_recall)
    print("acc", avg_acc)
    print("roc_auc", roc_auc)
    # print ("dice_pred",dice_pred/test_num)
    # print ("iou_pred",iou_pred/test_num)

    # network_names = ["unet"]
    # names = ["unet"]
    # plt.figure()
    # for i in range(len(network_names)):
    #     plt.plot(avg_fpr, avg_tpr, lw=2, label=f'{names[i]} (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='best')
    # plt.show()



