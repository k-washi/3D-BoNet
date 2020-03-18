import os
import pickle


def train(net, data, eval=False, start_ep=0, ep_num=51):
    l_ls_psemce = []
    l_ls_bbvert_all = []
    l_ls_bbvert_l2 = []
    l_ls_bbvert_ce = []
    l_ls_bbvert_iou = []
    l_ls_bbscore = []
    l_ls_pmask = []

    for ep in range(start_ep, start_ep + ep_num, 1):
        l_rate = max(0.0005 / (2 ** (ep // 20)), 0.00001)

        data.shuffle_train_files(ep)
        total_train_batch_num = data.total_train_batch_num
        print('total train batch nums:', total_train_batch_num)
        for i in range(total_train_batch_num):
            ###### training
            bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
            # print(bat_pc.shape, bat_psem_onehot.shape, bat_bbvert.shape, bat_pmask.shape)

            _, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask = net.sess.run(
                [
                    net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce,
                    net.bbvert_loss_iou, net.bbscore_loss, net.pmask_loss],
                feed_dict={net.X_pc: bat_pc[:, :, 0:6], net.Y_bbvert: bat_bbvert, net.Y_pmask: bat_pmask,
                           net.Y_psem: bat_psem_onehot, net.lr: l_rate, net.is_train: True})

            if i % 200 == 0:
                sum_train = net.sess.run(net.sum_merged,
                                         feed_dict={net.X_pc: bat_pc[:, :, 0:6], net.Y_bbvert: bat_bbvert,
                                                    net.Y_pmask: bat_pmask, net.Y_psem: bat_psem_onehot, net.lr: l_rate,
                                                    net.is_train: False})
                net.sum_writer_train.add_summary(sum_train, ep * total_train_batch_num + i)

            if i % 10 == 0:
                print('zep', ep, 'i', i, 'psemce', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce',
                      ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)

            l_ls_psemce.append(ls_psemce)
            l_ls_bbvert_all.append(ls_bbvert_all)
            l_ls_bbvert_l2.append(ls_bbvert_l2)
            l_ls_bbvert_ce.append(ls_bbvert_ce)
            l_ls_bbvert_iou.append(ls_bbvert_iou)
            l_ls_bbscore.append(ls_bbscore)
            l_ls_pmask.append(ls_pmask)

            ###### random testing
            if i % 200 == 0:
                bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_test_next_batch_random()
                ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask, sum_test, pred_bborder = net.sess.run(
                    [
                        net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou,
                        net.bbscore_loss, net.pmask_loss, net.sum_merged, net.pred_bborder],
                    feed_dict={net.X_pc: bat_pc[:, :, 0:6], net.Y_bbvert: bat_bbvert, net.Y_pmask: bat_pmask,
                               net.Y_psem: bat_psem_onehot, net.is_train: False})
                net.sum_write_test.add_summary(sum_test, ep * total_train_batch_num + i)
                print('ep', ep, 'i', i, 'test psem', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce',
                      ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)
                print('test pred bborder', pred_bborder)

            ###### saving model
            if i == total_train_batch_num - 1 or i == 0:
                net.saver.save(net.sess, save_path=net.train_mod_dir + 'model.cptk')
                print("ep", ep, " i", i, " model saved!")
            if ep % 5 == 0 and i == total_train_batch_num - 1:
                net.saver.save(net.sess, save_path=net.train_mod_dir + 'model' + str(ep).zfill(3) + '.cptk')

            ###### full eval, if needed
            if eval:
                if ep % 5 == 0 and i == total_train_batch_num - 1:
                    from main_eval import Evaluation
                    result_path = './log/test_res/' + str(ep).zfill(3) + '_' + data.test_areas[0] + '/'
                    Evaluation.ttest(net, data, result_path, test_batch_size=20)
                    Evaluation.evaluation(data.dataset_path, data.train_areas, result_path)
                    print('full eval finished!')

    with open('./log/ls_psemce.pickle', 'wb') as f:
        pickle.dump(l_ls_psemce, f)

    with open('./log/ls_bbvert.pickle', 'wb') as f:
        pickle.dump(l_ls_bbvert_all, f)

    with open('./log/ls_bbscore.pickle', 'wb') as f:
        pickle.dump(l_ls_bbscore, f)

    with open('./log/ls_pmask.pickle', 'wb') as f:
        pickle.dump(l_ls_pmask, f)

    net.saver.save(net.sess, save_path=net.train_mod_dir + 'model_fin.cptk')

############
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='./qbs_train_data', help='path to input directory: pickle形式')
    parser.add_argument('-s', default=0, type=int, help='start episode number')
    parser.add_argument('-e', default=51, type=int, help='num of episode')
    args = parser.parse_args()
    dataset_path = args.i

    ep_start = args.s
    num_ep = args.e

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

    from main_3D_BoNet import BoNet
    from qbs_data_helper import Data_Configs as Data_Configs

    configs = Data_Configs()
    net = BoNet(configs=configs)
    net.creat_folders(name='log', re_train=False)

    net.build_graph()

    ####
    from qbs_data_helper import DATA_QBS as Data

    train_areas = [1, 2]
    test_areas = [0]


    data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)
    train(net, data, start_ep=ep_start, ep_num=num_ep)