import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='graphics card')
parser.add_argument('--data', type=str, default='data/Milan400', help='data path')
parser.add_argument('--adjdata', type=str, default='data/Milan400/adj_mx.pkl', help='adj data path')
parser.add_argument('--seq_length', type=int, default=12, help='prediction length')
parser.add_argument('--nhid', type=int, default=60, help='')                                 # intermediate filters
# parser.add_argument('--nhid', type=int, default=40, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=400, help='number of nodes')
# parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
# parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
# parser.add_argument('--dropout', type=float, default=0.12, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')                 # DST-Blocks
parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--top_k', type=int, default=4, help='top-k sampling')                   # top-k
parser.add_argument('--print_every', type=int, default=12, help='')
parser.add_argument('--save', type=str, default='./garage/metr-la', help='save path')
parser.add_argument('--seed', type=int, default=530302, help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()
print(args)


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU


def main():
    setup_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    adj_mx, adj_mx_ = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).cuda() for i in adj_mx]
    # # adj_mx = [util.construct_adj(i,3) for i in adj_mx]
    # adj_mx = [util.construct_adj(adj_mx[0],3)]
    # adj_mx = adj_mx + adj_mx
    supports_ = util.construct_adj(adj_mx_, 3).cuda()
    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = util.load_hadj(args.adjdata, args.top_k)

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)

    H_a = H_a.cuda()
    H_b = H_b.cuda()
    G0 = torch.tensor(G0).cuda()
    G1 = torch.tensor(G1).cuda()
    H_T_new = torch.tensor(H_T_new).cuda()
    lwjl = lwjl.cuda()
    indices = indices.cuda()

    G0_all = torch.tensor(G0_all).cuda()
    G1_all = torch.tensor(G1_all).cuda()

    engine = trainer(args.batch_size, scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, supports, supports_, H_a, H_b, G0, G1, indices,
                     G0_all, G1_all, H_T_new, lwjl, args.clip, args.lr_decay_rate)

    print("start training...", flush=True)
    g_train_loss = []
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        print('***** Epoch: %03d START *****' % i)
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        # np.savez('train_data.npz', data=dataloader['train_loader'].ys[:, 0, :, :])
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).cuda()
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).cuda()
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_mse.append(metrics[3])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train MAE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], train_mse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        engine.scheduler.step()

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda()
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).cuda()
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mse = np.mean(train_mse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        g_train_loss.append(mtrain_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train MSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_mse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        print('***** Epoch: %03d END *****' % i)
        print('\n')

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        # torch.load("./garage/milan400_exp1_best_0.01.pth"))
        torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda()
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).cuda()
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    # print(yhat.shape)
    yhat = yhat[:realy.size(0), ...]
    # print(yhat.shape)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    print("Best model epoch:", str(bestid + 1))

    amae = []
    amape = []
    armse = []
    for i in range(12):
        # print(yhat.shape, realy.shape)
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        # print(pred.shape, real.shape)
        metrics = util.metric(pred, real)
        # print(metrics)
        prediction = pred.cpu().numpy()
        ground_truth = real.cpu().numpy()
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    np.savez('train_loss.npz', loss=g_train_loss)
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    # print("prediction:", prediction)
    # print("ground_truth:", ground_truth)
    # torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    # np.savez('predict_24_06_29-400.npz', prediction=prediction, ground_truth=ground_truth)
    # data = np.load('predict_call_in_07_15_3.npz')
    # print(data.files)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
