import argparse
from utils import psd_safe_cholesky
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dist
from fastprogress import master_bar, progress_bar
from torch.utils import data as tdata
#from convcnp.models import ConvCNP1d
from nmdp_1d_model import NeuralMixtureDensityProcess,AttentiveNMDP
from convcnp.gp import oracle_gp
from convcnp.dataset import Synthetic1D
from convcnp.visualize import plot_all, convert_tfboard
from convcnp.dataset.generate_1d_data import MixtureGPDataset
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.set_default_dtype(torch.float64)
import warnings
import time
# 忽略UserWarning

def train(model, dataloader, optimizer,epoch):
    model.train()
    total_loss, total_iw_nll, total_prop_loss, total_kl_reg = 0, 0, 0, 0
    num_batches = len(dataloader)
    # 计算一个随epoch变化的kl权重
    kl_weight = min(1.0, epoch / 100) # WARMUP_EPOCHS, e.g., 50
    for i, (xc, yc, xt, yt) in enumerate(progress_bar(dataloader, parent=args.mb)):
        xc, yc, xt, yt = xc.to(args.device), yc.to(args.device), xt.to(args.device), yt.to(args.device)

        optimizer.zero_grad()

        with torch.autograd.detect_anomaly():
            loss,iw_nll, proposal_loss, kl_reg = model(xc, yc, xt, yt)

            # 损失计算保持不变
            #loss = iw_nll + 0.05 * proposal_loss + 0.1 * kl_reg * kl_weight
            loss = loss + kl_reg * kl_weight
            # 添加梯度监控
            if i % 10 == 0:
                grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                avg_grad = sum(grads) / len(grads) if grads else 0
                args.writer.add_scalar('train/avg_gradient', avg_grad, epoch * num_batches + i)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # max_norm是可调超参

        optimizer.step()

        # 记录各个损失分量
        total_loss += loss.item()
        total_iw_nll += iw_nll.item()
        total_prop_loss += proposal_loss.item()
        total_kl_reg += kl_reg.item()

    num_batches = len(dataloader)
    return (total_loss / num_batches, total_iw_nll / num_batches,
            total_prop_loss / num_batches, total_kl_reg / num_batches)


import torch.distributions as dist


# 在你的训练脚本中

def train_1(model, dataloader, optimizer, epoch):
    model.train()
    # 我们现在主要关心总损失，但仍然可以记录报告的组件
    total_loss, total_iw_nll, total_prop_loss, total_kl_reg = 0, 0, 0, 0
    kl_final_weight = 0.01  # 可以调整的超参数
    kl_warmup_epochs = 100
    kl_weight = kl_final_weight * min(1.0, epoch / kl_warmup_epochs)
    kl_budget_start = 0.1
    kl_budget_end = 5.0
    kl_budget_warmup = 150
    kl_budget = min(kl_budget_end, kl_budget_start + (kl_budget_end - kl_budget_start) * (epoch / kl_budget_warmup))
    for i, (xc, yc, xt, yt) in enumerate(progress_bar(dataloader, parent=args.mb)):
        xc, yc, xt, yt = xc.to(args.device), yc.to(args.device), xt.to(args.device), yt.to(args.device)

        optimizer.zero_grad()

        # 模型现在返回一个统一的loss，和三个用于报告的值
        loss,reported_iw_nll, reported_prop_loss, reported_kl_reg = model(xc, yc, xt, yt)
        # --- 使用 Free Bits 计算KL损失 ---
        #kl_loss = torch.relu(reported_kl_reg - kl_budget)
        # loss = reported_iw_nll + 0.05 * reported_prop_loss + 0.1 * kl_loss * kl_weight
        #loss = loss + kl_loss * kl_weight
        # 直接使用这个统一的loss进行反向传播
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_iw_nll += reported_iw_nll.item()
        total_prop_loss += reported_prop_loss.item()
        total_kl_reg += reported_kl_reg.item()

    num_batches = len(dataloader)
    return (total_loss / num_batches, total_iw_nll / num_batches,
            total_prop_loss / num_batches, total_kl_reg / num_batches)


# 假设你在训练脚本的主文件里

def validate_2(model, dataloader):
    model.eval()

    # 从数据加载器中获取一个批次用于可视化和评估
    # 我们只取第一个，因为验证通常在单一样本上进行以进行可视化
    try:
        xc, yc, xt, yt = next(iter(dataloader))
    except StopIteration:
        print("Validation dataloader is empty.")
        return 0, 0, None  # 返回默认值

    # 定义一个更宽的范围用于绘图
    out_range_xt = torch.linspace(-4, 4, 401).reshape(1, -1, 1)

    with torch.no_grad():
        # 使用模型的 predict 方法
        pred_mean, pred_std = model.predict(xc, yc, xt, num_samples=200)
        out_range_pred_mean, out_range_pred_std = model.predict(xc, yc, out_range_xt, num_samples=200)

    # 将数据移到CPU以便计算损失和绘图
    pred_mean, pred_std = pred_mean.cpu(), pred_std.cpu()
    out_range_pred_mean, out_range_pred_std = out_range_pred_mean.cpu(), out_range_pred_std.cpu()
    yt_cpu = yt.cpu()

    # 创建预测分布 (独立的Normal)
    pred_dist = torch.distributions.Normal(pred_mean, pred_std)
    out_range_pred_dist = torch.distributions.Normal(out_range_pred_mean, out_range_pred_std)

    # 计算验证损失和RMSE
    # 计算每个点的对数似然，然后对所有点求和，再对批次求平均
    log_likelihood = pred_dist.log_prob(yt_cpu).sum(dim=(1, 2)).mean()
    loss = -log_likelihood

    rmse = (pred_mean - yt_cpu).pow(2).mean().sqrt()

    # 生成可视化图像 (需要一个绘图函数 plot_all)
    # plot_all 函数需要能处理 Normal 分布对象
    try:
        # 获取 Oracle GP 预测作为对比
        from convcnp.gp import oracle_gp  # 假设这个函数可用
        gp_pred = oracle_gp(xc, yc, xt)
        out_range_gp_pred = oracle_gp(xc, yc, out_range_xt)

        # 假设 plot_all 函数存在并能处理Normal分布
        image = plot_all(xc, yc, xt, yt, xt, gp_pred, pred_dist)
        image = convert_tfboard(image)
        out_range_image = plot_all(xc, yc, xt, yt, out_range_xt, out_range_gp_pred,
                                   out_range_pred_dist, support=True)
        out_range_image = convert_tfboard(out_range_image)
    except ImportError:
        print("Warning: Visualization libraries (convcnp) not found. Skipping plot generation.")
        image, out_range_image = None, None

    return loss.item(), rmse.item(), image, out_range_image


# 在您的主训练脚本的开头部分

def create_fixed_validation_set(test_dataset, num_functions=100, batch_size=1):
    """创建一个固定的、可重复的验证函数列表"""
    print("Creating a fixed validation set...")
    fixed_val_data = []
    for _ in range(num_functions):
        # 从数据集中采样一个函数
        # test_dataset.sample 会生成一个新函数
        # 我们把它存储起来
        xc, yc, xt, yt = test_dataset[0]  # 调用 __getitem__
        # 确保它们有批次维度
        fixed_val_data.append((xc.unsqueeze(0), yc.unsqueeze(0), xt.unsqueeze(0), yt.unsqueeze(0)))
    return fixed_val_data


# 确保这些库已导入


# 假设您的主训练脚本中已经有了 create_fixed_validation_set 函数
# 和 fixed_validation_data 变量

def validate_on_fixed_set(model, fixed_val_data, args):
    """
    在固定的验证集上评估模型，并对其中一个样本进行可视化。

    Args:
        model: 要评估的模型。
        fixed_val_data (list): 一个包含 (xc, yc, xt, yt) 元组的列表，代表固定的验证函数。
        args: 包含配置参数的对象。

    Returns:
        tuple: (平均验证损失, 平均RMSE, TensorBoard格式的图像, None)
    """
    model.eval()
    total_ll = 0
    total_rmse = 0
    num_functions = len(fixed_val_data)
    device = next(model.parameters()).device  # 获取模型所在的设备

    if num_functions == 0:
        print("Warning: Fixed validation set is empty.")
        return 0, 0, None, None

    with torch.no_grad():
        # 1. 在整个固定验证集上计算平均指标
        for xc, yc, xt, yt in fixed_val_data:
            # 将当前批次的数据移动到正确的设备
            xc_dev, yc_dev, xt_dev = xc.to(device), yc.to(device), xt.to(device)

            # 使用模型的 predict 方法
            pred_mean, pred_std = model.predict(xc_dev, yc_dev, xt_dev, num_samples=200)

            # 将结果移回CPU进行计算
            pred_mean, pred_std = pred_mean.cpu(), pred_std.cpu()

            # 创建预测分布
            pred_dist = torch.distributions.Normal(pred_mean, pred_std)

            # 计算并累加对数似然
            log_likelihood = pred_dist.log_prob(yt).sum(dim=(1, 2)).mean()
            total_ll += log_likelihood.item()

            # 计算并累加RMSE
            rmse = (pred_mean - yt).pow(2).mean().sqrt()
            total_rmse += rmse.item()

    # 计算平均损失和RMSE
    avg_ll = total_ll / num_functions
    avg_rmse = total_rmse / num_functions

    # 2. 选择一个固定的样本进行可视化 (例如，第一个样本)
    vis_xc, vis_yc, vis_xt, vis_yt = fixed_val_data[0]

    # 将可视化样本移动到设备
    vis_xc_dev, vis_yc_dev, vis_xt_dev = vis_xc.to(device), vis_yc.to(device), vis_xt.to(device)

    # 对可视化样本进行预测
    with torch.no_grad():
        vis_pred_mean, vis_pred_std = model.predict(vis_xc_dev, vis_yc_dev, vis_xt_dev, num_samples=200)

    vis_pred_mean, vis_pred_std = vis_pred_mean.cpu(), vis_pred_std.cpu()
    vis_pred_dist = torch.distributions.Normal(vis_pred_mean, vis_pred_std)

    # 3. 生成可视化图像
    image = None
    try:
        # 获取 Oracle GP 预测作为对比
        gp_pred = oracle_gp(vis_xc, vis_yc, vis_xt)

        # 调用绘图函数
        # 注意：这里的 vis_xt, vis_yt 是作为 "Ground Truth" 传入的
        # 而 vis_pred_dist 是模型的预测
        plot = plot_all(vis_xc, vis_yc, vis_xt, vis_yt, vis_xt, gp_pred, vis_pred_dist)

        # 转换为TensorBoard格式
        image = convert_tfboard(plot)
    except Exception as e:
        print(f"Warning: Visualization failed with error: {e}. Skipping plot generation.")

    # 返回平均指标和生成的图像
    return -avg_ll, avg_rmse, image, None  # 最后一个None是为了与旧函数签名保持一致


# 假设你在训练脚本的主文件里

def validate_2(model, dataloader):
    model.eval()

    # 从数据加载器中获取一个批次用于可视化和评估
    # 我们只取第一个，因为验证通常在单一样本上进行以进行可视化
    try:
        xc, yc, xt, yt = next(iter(dataloader))
    except StopIteration:
        print("Validation dataloader is empty.")
        return 0, 0, None  # 返回默认值

    # 定义一个更宽的范围用于绘图
    out_range_xt = torch.linspace(-4, 4, 401).reshape(1, -1, 1)

    with torch.no_grad():
        # 使用模型的 predict 方法
        pred_mean, pred_std = model.predict(xc, yc, xt, num_samples=200)
        out_range_pred_mean, out_range_pred_std = model.predict(xc, yc, out_range_xt, num_samples=200)

    # 将数据移到CPU以便计算损失和绘图
    pred_mean, pred_std = pred_mean.cpu(), pred_std.cpu()
    out_range_pred_mean, out_range_pred_std = out_range_pred_mean.cpu(), out_range_pred_std.cpu()
    yt_cpu = yt.cpu()

    # 创建预测分布 (独立的Normal)
    pred_dist = torch.distributions.Normal(pred_mean, pred_std)
    out_range_pred_dist = torch.distributions.Normal(out_range_pred_mean, out_range_pred_std)

    # 计算验证损失和RMSE
    # 计算每个点的对数似然，然后对所有点求和，再对批次求平均
    log_likelihood = pred_dist.log_prob(yt_cpu).sum(dim=(1, 2)).mean()
    loss = -log_likelihood

    rmse = (pred_mean - yt_cpu).pow(2).mean().sqrt()

    # 生成可视化图像 (需要一个绘图函数 plot_all)
    # plot_all 函数需要能处理 Normal 分布对象
    try:
        # 获取 Oracle GP 预测作为对比
        from convcnp.gp import oracle_gp  # 假设这个函数可用
        gp_pred = oracle_gp(xc, yc, xt)
        out_range_gp_pred = oracle_gp(xc, yc, out_range_xt)

        # 假设 plot_all 函数存在并能处理Normal分布
        from convcnp.visualize import plot_all, convert_tfboard
        image = plot_all(xc, yc, xt, yt, xt, gp_pred, pred_dist)
        image = convert_tfboard(image)
        out_range_image = plot_all(xc, yc, xt, yt, out_range_xt, out_range_gp_pred,
                                   out_range_pred_dist, support=True)
        out_range_image = convert_tfboard(out_range_image)
    except ImportError:
        print("Warning: Visualization libraries (convcnp) not found. Skipping plot generation.")
        image, out_range_image = None, None

    return loss.item(), rmse.item(), image, out_range_image


def validate(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    # 假设dataloader的batch_size为1
    xc, yc, xt, yt = next(iter(dataloader))
    out_range_xt = torch.linspace(-4, 4, 401).reshape(1, -1, 1)

    device = args.device  # 假设args.device在作用域内
    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
    out_range_xt = out_range_xt.to(device)

    # 获取Oracle GP的预测作为基准
    gp_pred = oracle_gp(xc, yc, xt)
    out_range_gp_pred = oracle_gp(xc, yc, out_range_xt)

    with torch.no_grad():
        #*****************************
        alpha, _ = model.prior_network(xc, yc)
        num_samples = 200
        z_samples = model.sample_from_dirichlet(alpha, num_samples)  # [S, B, C]

        # 获取模型预测
        means, scale_trils = model.mixture_network(xt)
        # out_range_means, out_range_scale_trils = model.mixture_network(out_range_xt)

        # 计算目标点范围内预测
        # 加权均值和协方差
        z_mean = z_samples.mean(dim=0)  # [B, C]
        weighted_mean = (z_mean.unsqueeze(-1) * means).sum(dim=1)  # [B, N]

        # 计算加权协方差
        diff = means - weighted_mean.unsqueeze(1)  # [B, C, N]
        cov_components = torch.matmul(diff.unsqueeze(-1), diff.unsqueeze(-2))  # [B, C, N, N]
        weighted_cov = (z_mean.unsqueeze(-1).unsqueeze(-1) * (cov_components + torch.matmul(scale_trils, scale_trils.transpose(-1, -2))).sum(dim=1))

        # 添加抖动确保正定
        weighted_cov += torch.eye(weighted_cov.size(-1), device = device) *1e-4
        scale_tril_out = torch.linalg.cholesky(weighted_cov)

        # 5. 创建最终分布
        pred_dist = dist.MultivariateNormal(
            loc=weighted_cov,
            scale_tril=scale_tril_out
        )
        # out_range_pred_dist = dist.MultivariateNormal(
        #     loc=weighted_cov,
        #     scale_tril=scale_tril_out
        # )

    loss = -pred_dist.log_prob(yt.squeeze(-1)).mean()
    rmse = (pred_dist.mean - yt.squeeze(-1)).pow(2).mean().sqrt()

    # 7. 生成可视化图像
    image = plot_all(xc, yc, xt, yt, xt, gp_pred, pred_dist)
    image = convert_tfboard(image)
    # out_range_image = plot_all(xc, yc, xt, yt, out_range_xt, out_range_gp_pred,
    #                            out_range_pred_dist, support=True)
    # out_range_image = convert_tfboard(out_range_image)

    return loss.item(), rmse.item(), image #, out_range_image
    # 解决plot_all中4D切片的问题：为scale_tril增加一个虚拟的“组件”维度
    # 这使得[B, N, N] -> [B, 1, N, N]，从而兼容plot_all中的切片操作
    # pred_dist.scale_tril = pred_dist.scale_tril.unsqueeze(1)
    # out_range_pred_dist.scale_tril = out_range_pred_dist.scale_tril.unsqueeze(1)
    #
    # # 使用创建的分布计算损失和评估指标
    # loss = -pred_dist.log_prob(yt.squeeze(-1)).mean()
    # rmse = (pred_dist.mean - yt.squeeze(-1)).pow(2).mean().sqrt()
    #
    # # --- 修改后对plot_all的调用 ---
    # # 现在 `pred_dist` 和 `out_range_pred_dist` 对象拥有 `plot_all` 函数所需的 `.mean` 和 `.scale_tril` 属性
    # image = plot_all(xc, yc, xt, yt, xt, gp_pred, pred_dist)
    # image = convert_tfboard(image)
    # out_range_image = plot_all(xc, yc, xt, yt, out_range_xt, out_range_gp_pred,
    #                            out_range_pred_dist, support=True)
    # out_range_image = convert_tfboard(out_range_image)
    #
    # return loss.item(), rmse.item(), image, out_range_image


def main():
    train_dataset = MixtureGPDataset(kernels=['eq', 'matern', 'periodic'], train=True, random_params=True)
    test_dataset = MixtureGPDataset(kernels=['eq', 'matern', 'periodic'], train=False, random_params=True)
    # train_dataset = Synthetic1D('matern',train=True)
    # test_dataset = Synthetic1D('matern',train=False)
    fixed_validation_data = create_fixed_validation_set(test_dataset, num_functions=100)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    warnings.filterwarnings("ignore", category=UserWarning)
    # 实例化NMDP模型
    # model = NeuralMixtureDensityProcess(
    #     x_dim=1,
    #     y_dim=1,
    #     num_components=args.num_components,
    #     hidden_dim=128,
    #     latent_dim=args.latent_dim,
    #     num_particles=200
    # ).to(args.device)
    model = AttentiveNMDP(
        x_dim=1,
        y_dim=1,
        num_components=args.num_components,
        h_dim=256,
        r_dim=256,
        #num_particles=200
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=1e-5,eps=1e-7)
    scheduler = ReduceLROnPlateau(optimizer, mode ='min',factor=0.5, patience=5)  # 动态降学习率
    args.mb = master_bar(range(1, args.epochs + 1))

    print("Starting training for NMDP model...")
    for epoch in args.mb:
        avg_train_loss, avg_iw_nll, avg_prop_loss, avg_kl_reg = train_1(model, trainloader, optimizer,epoch)
        #valid_loss, rmse, image, out_range_image = validate_on_fixed_set(model, testloader)
        valid_loss, rmse, image,out_range_image = validate_on_fixed_set(model, fixed_validation_data,args)
        scheduler.step(valid_loss)
        # 记录NMDP相关的损失
        args.writer.add_scalar('train/total_loss', avg_train_loss, epoch)
        args.writer.add_scalar('train/iw_nll', avg_iw_nll, epoch)
        args.writer.add_scalar('train/proposal_loss', avg_prop_loss, epoch)
        args.writer.add_scalar('train/kl_reg', avg_kl_reg, epoch)

        # 记录验证指标
        args.writer.add_scalar('validate/likelihood', valid_loss, epoch)
        args.writer.add_scalar('validate/rmse', rmse, epoch)
        args.writer.add_image('validate/image', image, epoch)
        #args.writer.add_image('validate/out_range_image', out_range_image, epoch)

        args.mb.write(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f},IW nll={avg_iw_nll:.4f},Proposal loss={avg_prop_loss:.4f},KL Reg={avg_kl_reg:.4f}'
                      f'Valid LL={valid_loss:.4f}, RMSE={rmse:.4f}')

    torch.save(model.state_dict(), args.filename)
    print(f"Model saved to {args.filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=32)
    parser.add_argument('--learning-rate', '-LR', type=float, default=1e-4)
    parser.add_argument('--epochs', '-E', type=int, default=200)
    # NMDP特定参数
    parser.add_argument('--num-components', type=int, default=9, help='Number of mixture components in NMDP')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension for context representation')

    args = parser.parse_args()

    args.filename = f'nmdp_model_C{args.num_components}_without_kl_0812.pth.gz'

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    args.writer = SummaryWriter(log_dir=f'logs/nmdp_C{args.num_components}_0811')
    main()
    args.writer.close()