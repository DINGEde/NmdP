import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import time
from utils import psd_safe_cholesky


class Attention(nn.Module):
    """Dot-product attention module."""

    def __init__(self, h_dim):
        super().__init__()
        self.scale = h_dim ** -0.5
        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)

    def forward(self, query, context):
        """
        query: (B, N_q, D) - 例如，目标点的表示
        context: (B, N_c, D) - 例如，上下文点的表示
        """
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(dots, dim=-1)

        return torch.matmul(attn_weights, v)

class ContextEncoder(nn.Module):
    """
    编码 (x, y) 上下文对。
    将(x,y)对映射到表示r。
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super().__init__()
        upgraded_h_dim = 256
        layers = [
            nn.Linear(x_dim + y_dim, upgraded_h_dim), nn.GELU(),
            nn.Linear(upgraded_h_dim, upgraded_h_dim), nn.GELU(),
            nn.Linear(upgraded_h_dim, r_dim)
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, xc, yc):
        # xc: [B, N_c, x_dim], yc: [B, N_c, y_dim]
        input_pairs = torch.cat([xc, yc], dim=-1)
        return self.encoder(input_pairs) # [B, N_c, r_dim]



class AttentiveMixtureDecoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim, num_components):
        super().__init__()
        self.num_components = num_components

        # 1. 将目标x点编码到与r相同的维度
        self.target_encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.GELU(),
            nn.Linear(h_dim, r_dim)
        )

        # 2. 注意力模块
        self.attention = Attention(r_dim)
        upgraded_h_dim = 256
        # 3. 混合专家MLP
        # 输入维度: r_dim (来自目标x) + r_dim (来自注意力)
        self.expert_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(r_dim * 2, upgraded_h_dim), nn.GELU(),
                nn.Linear(upgraded_h_dim, upgraded_h_dim), nn.GELU(),
                nn.Linear(upgraded_h_dim, upgraded_h_dim), nn.GELU(),
                nn.Linear(upgraded_h_dim, y_dim * 2)  # mu, sigma logits
            ) for _ in range(num_components)
        ])

        self.mean_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, context_r, context_x, target_x):
        """
        context_r: 编码后的上下文表示 [B, N_c, r_dim]
        context_x: 上下文点的x坐标 [B, N_c, x_dim] (用于注意力机制的key)
        target_x: 目标点的x坐标 [B, N_t, x_dim]
        """
        # 1. 将目标x编码为"查询(query)"
        query = self.target_encoder(target_x)  # [B, N_t, r_dim]

        # 2. 使用注意力机制，为每个目标点计算一个定制的上下文向量
        # 注意力的context是context_r
        attentive_context = self.attention(query, context_r)  # [B, N_t, r_dim]

        # 3. 拼接输入
        # 每个目标点的输入 = 自身表示 + 注意力上下文
        combined_input = torch.cat([query, attentive_context], dim=-1)  # [B, N_t, r_dim * 2]

        # 4. 通过所有专家网络
        all_mu = []
        all_sigma = []
        for i in range(self.num_components):
            output = self.expert_networks[i](combined_input)  # [B, N_t, y_dim*2]
            mu_i, pre_sigma_i = torch.chunk(output, 2, dim=-1)
            mu_i = torch.tanh(mu_i) * self.mean_scale
            sigma_i = 0.1 + 0.9 * F.softplus(pre_sigma_i)
            all_mu.append(mu_i)
            all_sigma.append(sigma_i)

        means = torch.stack(all_mu, dim=1)  # [B, C, N_t, y_dim]
        sigmas = torch.stack(all_sigma, dim=1)  # [B, C, N_t, y_dim]

        return means.squeeze(-1), sigmas.squeeze(-1)


class AttentiveNMDP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, num_components=8, h_dim=128, r_dim=128):
        super().__init__()
        self.num_components = num_components

        # 1. 上下文编码器
        self.context_encoder = ContextEncoder(x_dim, y_dim, h_dim, r_dim)

        # 2. 专家混合解码器
        self.decoder = AttentiveMixtureDecoder(x_dim, y_dim, r_dim, h_dim, num_components)

        # 3. 狄利克雷先验/提议网络 (这里的SelfAttentionEncoder是独立的)
        self.prior_network = DirichletPriorNetwork(x_dim, y_dim, num_components, h_dim, latent_dim=r_dim)
        self.proposal_network = DirichletPriorNetwork(x_dim, y_dim, num_components, h_dim, latent_dim=r_dim)

        # 根据您的实验，可以决定是否保留粒子数
        self.num_particles = 20

    def compute_mixture_logprob(self, target_y, means, sigmas, z_samples):
        # target_y: [B, N_t]
        # means, sigmas: [S, B, C, N_t]
        # z_samples: [S, B, C]

        # 1. 创建独立高斯分布
        dists = torch.distributions.Normal(loc=means, scale=sigmas)

        # 2. 计算每个点在每个专家下的对数似然
        # 输入 target_y 扩展为 [1, B, 1, N_t]
        log_probs_components = dists.log_prob(target_y.unsqueeze(0).unsqueeze(2))  # [S, B, C, N_t]

        # 3. 对所有目标点的对数似然求和，得到每个专家的总似然
        log_probs_components = log_probs_components.sum(dim=-1)  # [S, B, C]

        # 4. 与混合权重结合
        log_z = torch.log(z_samples + 1e-9)
        log_joint = log_z + log_probs_components

        # 5. LogSumExp
        mixture_log_probs = torch.logsumexp(log_joint, dim=-1)  # [S, B]

        return mixture_log_probs
    def sample_from_dirichlet(self, alpha, num_samples):
        """从Dirichlet分布采样混合权重"""
        dirichlet = dist.Dirichlet(alpha)
        z_samples = dirichlet.rsample((num_samples,))  # [S, B, C]
        return z_samples
    def forward(self, xc, yc, xt, yt):
        # 编码
        context_r = self.context_encoder(xc, yc)

        # 狄利克雷部分
        prior_alpha, _ = self.prior_network(xc, yc)
        all_x = torch.cat([xc, xt], dim=1)
        all_y = torch.cat([yc, yt], dim=1)
        proposal_alpha, _ = self.proposal_network(all_x, all_y)

        z_samples = self.sample_from_dirichlet(proposal_alpha, self.num_particles)
        z_samples = torch.clamp(z_samples, min=1e-6, max=1 - 1e-6)
        z_samples = z_samples / z_samples.sum(dim=-1, keepdim=True)

        log_prior = dist.Dirichlet(prior_alpha).log_prob(z_samples)
        log_proposal = dist.Dirichlet(proposal_alpha).log_prob(z_samples)

        # 解码
        # 为了节约计算，我们可以先不扩展 z_samples
        means_components, sigmas_components = self.decoder(context_r, xc, xt)  # [B, C, N_t]

        # 扩展以匹配粒子数 S
        means = means_components.unsqueeze(0).expand(self.num_particles, -1, -1, -1)
        sigmas = sigmas_components.unsqueeze(0).expand(self.num_particles, -1, -1, -1)

        # 计算似然
        log_likelihood = self.compute_mixture_logprob(yt.squeeze(-1), means, sigmas, z_samples)

        # IWAE 损失计算 (与之前相同)
        # 1. 计算未经 detach 的 log importance weights
        log_weights = log_likelihood + log_prior - log_proposal

        # 2. 计算 Importance Weighted ELBO (IWAE ELBO)
        # logsumexp(log_weights) 约等于 log(sum(weights))
        # 这是IWAE的目标函数，它对所有参数（包括提议网络）都是可微的。
        log_iw_elbo = torch.logsumexp(log_weights, dim=0) - torch.log(
            torch.tensor(self.num_particles, dtype=torch.float64))

        # 3. 最终的、单一的损失函数
        # 我们要最大化 ELBO，所以最小化它的负值
        loss = -log_iw_elbo.mean()

        # 4. (可选但推荐) 为了监控，计算并报告各个分量
        # 注意：这里的计算使用了 detach，因此不会影响梯度，仅用于日志记录
        with torch.no_grad():
            normalized_weights = F.softmax(log_weights, dim=0)
            # 重新计算 iw_nll 用于报告 (与我们之前定义的 iw_nll 相同)
            reported_iw_nll = -(normalized_weights * (log_likelihood + log_prior)).sum(dim=0).mean()
            reported_proposal_loss = -(normalized_weights * log_proposal).sum(dim=0).mean()
            reported_kl_reg = dist.kl_divergence(dist.Dirichlet(proposal_alpha), dist.Dirichlet(prior_alpha)).mean()

        return loss, reported_iw_nll, reported_proposal_loss, reported_kl_reg

    def predict(self, xc, yc, xt, num_samples=200):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            xc, yc, xt = map(lambda t: t.to(device), [xc, yc, xt])

            # 编码
            context_r = self.context_encoder(xc, yc)

            # 使用先验进行预测
            prior_alpha, _ = self.prior_network(xc, yc)
            z_samples = self.sample_from_dirichlet(prior_alpha, num_samples)  # [S, B, C]

            # 解码
            means_components, sigmas_components = self.decoder(context_r, xc, xt)  # [B, C, N_t]

            # 加权平均均值
            pred_means_per_sample = torch.einsum('sbc,bcn->sbn', z_samples, means_components)
            pred_mean = pred_means_per_sample.mean(dim=0)

            # 使用全方差定律计算标准差
            # Var(Y) = E[Var(Y|z)] + Var(E[Y|z])
            # E[Var(Y|z)]
            weighted_var = torch.einsum('sbc,bcn->sbn', z_samples, sigmas_components ** 2)
            expected_var_y = weighted_var.mean(dim=0)
            # Var(E[Y|z])
            var_of_mean_y = pred_means_per_sample.var(dim=0)

            pred_var = expected_var_y + var_of_mean_y
            pred_std = torch.sqrt(pred_var + 1e-6)

            return pred_mean.unsqueeze(-1), pred_std.unsqueeze(-1)
# --- 自注意力编码器 (保持不变) ---
class SelfAttentionEncoder(nn.Module):
    """
    使用Transformer Encoder来聚合上下文信息，生成全局任务表示。
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.input_projection = nn.Linear(x_dim + y_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, context_x, context_y):
        device = next(self.parameters()).device
        context_x, context_y = context_x.to(device), context_y.to(device)
        context_input = torch.cat([context_x, context_y], dim=-1)
        projected_input = self.input_projection(context_input)
        encoded_output = self.transformer_encoder(projected_input)
        mu = encoded_output.mean(dim=1)
        return mu


class DirichletPriorNetwork(nn.Module):
    """Dirichlet先验推理网络"""

    def __init__(self, x_dim=1, y_dim=1, num_components=3, hidden_dim=128, latent_dim=128):
        super().__init__()
        self.num_components = num_components
        self.context_encoder = SelfAttentionEncoder(x_dim, y_dim, hidden_dim=latent_dim)
        self.alpha_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_components)
        )

    def forward(self, context_x, context_y):
        mu = self.context_encoder(context_x, context_y)
        alpha_logits = self.alpha_network(mu)
        alpha = F.softplus(alpha_logits) + 1e-4
        return alpha, mu


# Transformer架构，输出MVN参数 ---
class TransformerMVNGenerator(nn.Module):
    """
    使用Transformer架构的生成器，为每个专家网络输出一个多元正态分布(MVN)的参数。
    它输出一个均值向量和一个Cholesky分解因子L，以确保协方差矩阵是正定的。
    """

    def __init__(self, x_dim=1, y_dim=1, num_components=3, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.num_components = num_components
        self.y_dim = y_dim  # 假设 y_dim=1
        #self.positive_transform = nn.Softmax()
        # Transformer编码器用于处理目标点序列
        self.input_projection = nn.Linear(x_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 线性层用于从Transformer的输出生成均值和Cholesky因子
        # 输出维度：均值(num_points) + Cholesky因子((num_points * (num_points + 1)) // 2)
        # 乘以 num_components 是为所有专家一次性生成
        self.mean_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_components)
        ])
        self.cov_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_components)
        ])

        #self.cholesky_head = nn.Linear(hidden_dim, num_components * hidden_dim)
        # 新输出: diagonal_log_values (只需要N参数)
        self.positive_activation = nn.Softplus()
        #self.scale_param = nn.Parameter(torch.zeros(1))
        # 添加一个缩放参数，让模型可以学习输出的范围
        #self.mean_scale = nn.Parameter(torch.tensor(3.0))  # 假设y的范围大致在-3到3之间
    def forward(self, x):
        """
        Args:
            x: 输入目标点，形状为 [B, N, D_x]
        Returns:
            means: [B, C, N] - L个均值向量
            scale_trils: [B, C, N, N] - L个Cholesky因子L
        """
        # print(f"输入形状: {x.shape}")  # 应为 [B, N, input_dim]
        batch_size, num_points, _ = x.shape
        device = next(self.parameters()).device
        x = x.to(device)

        # 1. Transformer处理输入序列
        projected_x = self.input_projection(x)
        transformer_output = self.transformer_encoder(projected_x)  # [B, N, H]
        #context_features = transformer_output.mean(dim=1)  # [B, H]
        # 2. 生成均值向量
        # [B, N, H] -> [B, N, C] -> [B, C, N]
        #means = self.mean_head(transformer_output).permute(0, 2, 1)
        all_means = []
        all_diag_log_values = []

        for i in range(self.num_components):
            # 将 transformer_output 分别送入每个专家的头
            mean_i = self.mean_heads[i](transformer_output)  # [B, N, 1]
            diag_log_i = self.cov_heads[i](transformer_output)  # [B, N, 1]
            all_means.append(mean_i)
            all_diag_log_values.append(diag_log_i)
        #means = self.mean_head(transformer_output)  # [B, output_dim * num_components]
        means = torch.cat(all_means,dim = -1)
        diag_log_values = torch.cat(all_diag_log_values,dim = -1)
        #means = means.view(batch_size, self.num_components, self.y_dim)  # [B, C, D]
        #means = means.view(batch_size, self.num_components, num_points)

        means = means.permute(0, 2, 1)
        # --- 对均值输出进行归一化和缩放 ---
        # means = torch.tanh(means) * self.mean_scale
        diag_log_values = diag_log_values.permute(0, 2, 1)
        #diag_log_values = diag_log_values.view(batch_size, self.num_components, num_points)  # [B, C, D]
        # 6. 应用正约束确保对角元素>0
        diag_values = self.positive_activation(diag_log_values)
        diag_values = torch.clamp(diag_values, min=1e-4,max=5.0)  # 保持方差下限

        # 7. 创建对角协方差矩阵的Cholesky因子
        # 对角矩阵的Cholesky因子是开平方的对角矩阵
        #scale_trils = torch.diag_embed(torch.sqrt(diag_values))
        identity = torch.eye(num_points, device=x.device).expand(batch_size, self.num_components, num_points,
                                                                 num_points)
        # 对每个组件应用不同的对角值
        scale_trils = identity * torch.sqrt(diag_values).unsqueeze(-1)
        # print(f"均值形状: {means.shape}")  # 应为 [B, C, N]
        # print(f"协方差形状: {scale_trils.shape}")  # 应为 [B, C, N, N]
        return means, scale_trils

# # --- NMDP 主模型 ---
# class NeuralMixtureDensityProcess(nn.Module):
#     """NMDP主模型，现在输出MVN分布"""
#
#     def __init__(self, x_dim=1, y_dim=1, num_components=3, hidden_dim=128, latent_dim=128, num_particles=10):
#         super().__init__()
#         self.num_components = num_components
#         self.y_dim = y_dim
#         self.num_particles = num_particles
#         self.positive_transform = nn.Softplus()
#         self.prior_network = DirichletPriorNetwork(x_dim, y_dim, num_components, hidden_dim, latent_dim)
#         self.proposal_network = DirichletPriorNetwork(x_dim, y_dim, num_components, hidden_dim, latent_dim)
#         # --- 核心修改 2: 使用新的Transformer生成器 ---
#         self.mixture_network = TransformerMVNGenerator(x_dim, y_dim, num_components, hidden_dim)
#         self._initialize_weights()
#     def _initialize_weights(self):
#         # 初始化协方差头，避免初始值过大
#         for m in self.mixture_network.cov_heads.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.constant_(m.bias, -2.0)  # 初始小协方差
#     def sample_from_dirichlet(self, alpha, num_samples):
#         """从Dirichlet分布采样混合权重"""
#         dirichlet = dist.Dirichlet(alpha)
#         z_samples = dirichlet.rsample((num_samples,))  # [S, B, C]
#         return z_samples
#
#     # --- 核心修改 3: 修改对数似然计算以适应MVN ---
#     def compute_mixture_logprob(self, target_y, means, scale_trils, z_samples):
#         """计算混合MVN分布的对数似然 p(y|x, z)"""
#         num_samples, batch_size, _ = z_samples.shape
#         num_target_points = target_y.size(1)
#         assert means.size(2) == num_target_points, \
#             f"目标点数量不一致: 模型输出{means.size(2)} vs 输入{num_target_points}"
#         target_y = target_y.squeeze(-1)  # 假设y_dim=1, [B, N]
#         # target_y: [B, N] -> [1, B, 1, N]
#         target_y_exp = target_y.unsqueeze(0).expand(num_samples, -1, -1)
#         #target_y_expanded = target_y_exp.unsqueeze(2).expand(-1, -1, self.num_components, -1)
#         # means: [B, C, N] -> [S, B, C, N]
#         # scale_trils: [B, C, N, N] -> [S, B, C, N, N]
#         #使用repeat替代expand
#         means_exp = means.unsqueeze(0).repeat(num_samples, 1, 1, 1)  # [S, B, C, N]
#         scale_trils_exp = scale_trils.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)  # [S, B, C, N, N]
#         mvn_dists = dist.MultivariateNormal(loc=means_exp, scale_tril= scale_trils_exp)
#         # 1. 创建仅包含下三角部分的掩码
#         # 计算对数概率
#         # log_prob的输入需要是[..., N]，所以target_y_exp的形状是[1, B, 1, N]
#         # mvn_dists的batch_shape是[B, C]，event_shape是[N]
#         # log_prob输出形状[1, B, C]
#         #log_probs_components = mvn_dists.log_prob(target_y_exp.unsqueeze(2))
#
#         # 扩展对数概率以匹配样本维度 [1, B, C] -> [S, B, C]
#         #log_probs_components_exp = log_probs_components.expand(num_samples, -1, -1)
#         # 计算对数概率 [S, B, C]
#         log_probs_components = mvn_dists.log_prob(target_y_exp.unsqueeze(2))
#         log_z = torch.log(z_samples+1e-9)
#         log_joint = log_z+log_probs_components
#         # === 新增稳定性处理 ===
#         # max_log = log_joint.max(dim=-1, keepdim=True)[0]
#         # stable_log_joint = log_joint - max_log
#         mixture_log_probs = torch.logsumexp(log_joint, dim=-1)
#
#         # 约束极端值
#         mixture_log_probs = torch.clamp(mixture_log_probs, min=-50, max=50)
#         # 使用LogSumExp技巧进行混合: log(sum(z_i * p_i))
#         # log(z)的形状: [S, B, C]
#         #mixture_log_probs = torch.logsumexp(torch.log(z_exp + 1e-9) + log_probs_components_exp, dim=-1)  # [S, B]
#         #mixture_log_probs = torch.logsumexp(log_joint, dim=-1)
#         return mixture_log_probs  # 返回每个样本的总对数似然
#
#     # 在 NeuralMixtureDensityProcess 类的 forward 方法中
#
#     def forward(self, context_x, context_y, target_x, target_y):
#         device = next(self.parameters()).device
#         context_x, context_y, target_x, target_y = map(lambda t: t.to(device),
#                                                        [context_x, context_y, target_x, target_y])
#
#         all_x = torch.cat([context_x, target_x], dim=1)
#         all_y = torch.cat([context_y, target_y], dim=1)
#
#         prior_alpha, _ = self.prior_network(context_x, context_y)
#         proposal_alpha, _ = self.proposal_network(all_x, all_y)
#
#         z_samples = self.sample_from_dirichlet(proposal_alpha, self.num_particles)
#         z_samples = torch.clamp(z_samples, min=1e-6, max=1 - 1e-6)
#         z_samples = z_samples / z_samples.sum(dim=-1, keepdim=True)
#
#         log_prior = dist.Dirichlet(prior_alpha).log_prob(z_samples)
#         log_proposal = dist.Dirichlet(proposal_alpha).log_prob(z_samples)
#
#         means, scale_trils = self.mixture_network(target_x)
#         log_likelihood = self.compute_mixture_logprob(target_y, means, scale_trils, z_samples)
#
#         # ==================== 全新的损失计算逻辑 ====================
#
#         # 1. 计算未经 detach 的 log importance weights
#         log_weights = log_likelihood + log_prior - log_proposal
#
#         # 2. 计算 Importance Weighted ELBO (IWAE ELBO)
#         # logsumexp(log_weights) 约等于 log(sum(weights))
#         # 这是IWAE的目标函数，它对所有参数（包括提议网络）都是可微的。
#         log_iw_elbo = torch.logsumexp(log_weights, dim=0) - torch.log(
#             torch.tensor(self.num_particles, device=device, dtype=torch.float64))
#
#         # 3. 最终的、单一的损失函数
#         # 我们要最大化 ELBO，所以最小化它的负值
#         loss = -log_iw_elbo.mean()
#
#         # 4. (可选但推荐) 为了监控，计算并报告各个分量
#         # 注意：这里的计算使用了 detach，因此不会影响梯度，仅用于日志记录
#         with torch.no_grad():
#             normalized_weights = F.softmax(log_weights, dim=0)
#             # 重新计算 iw_nll 用于报告 (与我们之前定义的 iw_nll 相同)
#             reported_iw_nll = -(normalized_weights * (log_likelihood + log_prior)).sum(dim=0).mean()
#             reported_proposal_loss = -(normalized_weights * log_proposal).sum(dim=0).mean()
#             reported_kl_reg = dist.kl_divergence(dist.Dirichlet(proposal_alpha), dist.Dirichlet(prior_alpha)).mean()
#
#         # 返回单一的损失和用于报告的组件
#         return loss, reported_iw_nll, reported_proposal_loss, reported_kl_reg
#
#
#     def predict(self, context_x, context_y, target_x, num_samples=100):
#         self.eval()
#         with torch.no_grad():
#             device = next(self.parameters()).device
#             context_x, context_y, target_x = map(lambda t: t.to(device), [context_x, context_y, target_x])
#
#             alpha, _ = self.prior_network(context_x, context_y)
#             z_samples = self.sample_from_dirichlet(alpha, num_samples)  # [S, B, C]
#             # 添加稳定性处理
#             z_samples = torch.clamp(z_samples, min=1e-6, max=1 - 1e-6)
#             z_samples = z_samples / z_samples.sum(dim=-1, keepdim=True)
#             means, scale_trils = self.mixture_network(target_x)  # means: [B, C, N], scale_trils: [B, C, N, N]
#
#             # 对每个样本加权平均预测
#             pred_means_per_sample = torch.einsum('sbc,bcn->sbn', z_samples, means)
#
#             # 最终预测均值
#             pred_mean = pred_means_per_sample.mean(dim=0)  # [B, N]
#
#             # 预测标准差使用样本标准差
#             pred_std = pred_means_per_sample.std(dim=0)  # [B, N]
#
#             return pred_mean.unsqueeze(-1), pred_std.unsqueeze(-1)
#
#     # 在您的 NeuralMixtureDensityProcess 类中修改/增加此方法
#     def sample_posterior(self, context_x, context_y, target_x, num_samples=100):
#         """从后验分布中采样完整的函数，以用于验证和可视化。"""
#         self.eval()
#         with torch.no_grad():
#             device = next(self.parameters()).device
#             context_x, context_y, target_x = map(lambda t: t.to(device), [context_x, context_y, target_x])
#
#             alpha, _ = self.prior_network(context_x, context_y)
#             z_samples = self.sample_from_dirichlet(alpha, num_samples)  # [S, B, C]
#             # 添加稳定性处理
#             z_samples = torch.clamp(z_samples, min=1e-6, max=1 - 1e-6)
#             z_samples = z_samples / z_samples.sum(dim=-1, keepdim=True)
#             means, scale_trils = self.mixture_network(target_x)  # means: [B, C, N], scale_trils: [B, C, N, N]
#
#             # 为每个z样本选择一个专家分布进行采样
#             # 这是一个简化的混合采样方法，更准确的方法是加权求和，但更复杂
#             # 此处我们为每个z样本采样一个专家类别
#             # 更高效的选择专家方法
#             expert_indices = torch.distributions.Categorical(z_samples).sample()  # [S, B]
#             batch_indices = torch.arange(z_samples.size(1), device=device)[None, :]
#
#             selected_means = means[batch_indices, expert_indices]  # [S, B, N]
#             selected_scale_trils = scale_trils[batch_indices, expert_indices]  # [S, B, N, N]
#
#             # 从选定的MVN分布中采样
#             mvn_dists = dist.MultivariateNormal(loc=selected_means, scale_tril=selected_scale_trils)
#             posterior_samples = mvn_dists.sample()  # [S, B, N]
#
#             return posterior_samples.permute(1, 0, 2).unsqueeze(-1)  # [B, S, N, 1]
