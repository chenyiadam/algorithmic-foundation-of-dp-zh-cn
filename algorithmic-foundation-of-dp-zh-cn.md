## 6 查询的提升算法

在前面的章节中，我们专注于私有查询发布问题，在该问题中，我们坚持对所有查询的最坏情况误差进行界定。如果我们改为仅要求在给定查询的某种分布下平均误差较低，我们的问题会更容易解决吗？在本节中，我们会发现答案是否定的：给定一个能够在查询的任何分布下以较低平均误差解决查询发布问题的机制，我们可以将其“提升”为一个能够解决查询发布问题并达到最坏情况误差的机制。这既揭示了私有查询发布的难度，也为我们设计私有查询发布算法提供了一个新工具。

提升算法是一种通用且广泛使用的提高学习算法准确性的方法。给定一组带标签的训练示例

$$
\left\{  {\left( {{x}_{1},{y}_{1}}\right) ,\left( {{x}_{2},{y}_{2}}\right) ,\ldots ,\left( {{x}_{m},{y}_{m}}\right) }\right\}  ,
$$

其中每个 ${x}_{i}$ 是从全域 $\mathcal{U}$ 上的基础分布 $\mathcal{D}$ 中抽取的，并且每个 ${y}_{i} \in  \{  + 1, - 1\}$ ，一个学习算法会产生一个假设 $h : \mathcal{U} \rightarrow  \{  + 1, - 1\}$ 。理想情况下，$h$ 不仅会“描述”给定样本上的标签，还会具有泛化能力，为从基础分布中抽取的其他元素提供一种相当准确的分类方法。提升算法的目标是将一个弱的基础学习器（其产生的假设可能仅比随机猜测略好）转换为一个强学习器，该强学习器能为根据 $\mathcal{D}$ 抽取的样本提供非常准确的预测器。许多提升算法都具有以下基本结构。首先，在样本集上施加一个初始（通常是均匀的）概率分布。然后计算按轮次进行。在每一轮 $t$ 中：

1. 在当前分布（记为 ${\mathcal{D}}_{t}$ ）上运行基础学习器，产生一个分类假设 ${h}_{t}$ ；并且

2. 使用假设 ${h}_{1},\ldots ,{h}_{t}$ 对样本进行重新加权，定义一个新的分布 ${\mathcal{D}}_{t + 1}$ 。

该过程在预定的轮数之后或者当确定假设的适当组合足够准确时停止。因此，给定一个基础学习器，提升算法的设计决策是（1）如何从一轮到下一轮修改概率分布，以及（2）如何组合假设 ${\left\{  {h}_{t}\right\}  }_{t = 1,\ldots ,T}$ 以形成最终的输出假设。

在本节中，我们将对查询使用提升算法——也就是说，对于提升算法而言，全域 $\mathcal{U}$ 是一组查询 $\mathcal{Q}$ ——以获得一个用于回答大量任意低敏感度查询的离线算法。该算法比中位数机制所需的空间更少，并且根据基础学习器的不同，其时间效率也可能更高。

该算法围绕着一个有点神奇的事实（引理 6.5）：如果我们能找到一个对少数选定查询提供准确答案的概要，那么实际上这个概要能为大多数查询提供准确答案！我们将这个事实应用于基础学习器，该学习器从 $\mathcal{Q}$ 上的分布中采样，并输出一个“弱”概要，该概要为 $\mathcal{Q}$ 中大部分权重的查询提供“好”的答案，以差分隐私的方式进行提升，从而获得一个对所有 $\mathcal{Q}$ 都适用的概要。

尽管提升操作是在查询上进行的，但隐私保护仍然是针对数据库的行。查询提升中的隐私挑战源于数据库中的每一行都会影响所有查询的答案这一事实。这将体现在查询的重新加权上：相邻的数据库可能会导致截然不同的重新加权，这将在生成的 ${h}_{t}$ 中体现出来，这些 ${h}_{t}$ 共同构成概要。

提升过程的运行时间与查询数量 $\left| \mathcal{Q}\right|$ 和基础概要生成器的运行时间近似线性相关，而与数据全域大小 $\left| \mathcal{X}\right|$ 无关。这为构建高效且准确的隐私保护机制提供了一条新途径，类似于机器学习文献中提升算法所采用的方法：算法设计者可以处理构建一个弱隐私保护基础概要生成器这一（可能容易得多）的任务，并自动获得一个更强的机制。

### 6.1 查询增强算法

我们将使用第2节中概述的数据库行表示法，即将数据库视为行的多重集，也就是 $\mathcal{X}$ 的元素。固定数据库大小 $n$、数据全域 $\mathcal{X}$ 和查询集 $\mathcal{Q} = \{ q$：$\left. {{\mathcal{X}}^{ * } \rightarrow  \mathbb{R}}\right\}$ 为敏感度至多为 $\rho$ 的实值查询。

我们假设存在一个基础概要生成器（在6.2节中我们将看到如何构建这些生成器）。接下来阐述的基础生成器所需的性质是，对于查询集 $\mathcal{Q}$ 上的任何分布 $\mathcal{D}$，基础生成器的输出可用于为大部分查询计算准确答案，其中“大部分”是根据 $\mathcal{D}$ 给出的权重来定义的。基础生成器由以下参数化：$k$，即要采样的查询数量；$\lambda$，其输出的准确性要求；$\eta$，描述“大部分”查询含义的“大”的度量；以及 $\beta$，失败概率。

定义6.1（$(\left( {k,\lambda ,\eta ,\beta }\right)$ - 基础概要生成器）。对于固定的数据库大小 $n$、数据全域 $\mathcal{X}$ 和查询集 $\mathcal{Q}$，考虑一个概要生成器 $\mathcal{M}$，它从 $\mathcal{Q}$ 上的分布 $\mathcal{D}$ 中独立采样 $k$ 个查询并输出一个概要。我们称 $\mathcal{M}$ 是一个 $\left( {k,\lambda ,\eta ,\beta }\right)$ - 基础概要生成器，如果对于 $\mathcal{Q}$ 上的任何分布 $\mathcal{D}$，除了 $\mathcal{M}$ 抛硬币的概率为 $\beta$ 的情况外，$\mathcal{M}$ 输出的概要 $\mathcal{S}$ 对于由 $\mathcal{D}$ 加权的 $\mathcal{Q}$ 的 $\left( {1/2 + \eta }\right)$ - 部分质量是 $\lambda$ - 准确的：

$$
\mathop{\Pr }\limits_{{q \sim  \mathcal{D}}}\left\lbrack  {\left| {q\left( \mathcal{S}\right)  - q\left( x\right) }\right|  \leq  \lambda }\right\rbrack   \geq  1/2 + \eta . \tag{6.1}
$$

查询增强算法可用于任何类型的查询和任何差分隐私基础概要生成器。运行时间继承自基础概要生成器。增强器会投入额外的时间，该时间与 $\left| \mathcal{Q}\right|$ 呈拟线性关系，特别是其运行时间并不直接取决于数据全域的大小。

为了指定增强算法，我们需要指定一个停止条件、一个聚合机制以及一个用于更新 $\mathcal{Q}$ 上当前分布的算法。

停止条件。我们将算法运行固定的 $T$ 轮数——这将是我们的停止条件。选择 $T$ 是为了确保足够的准确性（具有非常高的概率）；正如我们将看到的，$\log \left| \mathcal{Q}\right| /{\eta }^{2}$ 轮就足够了。

更新分布。尽管分布在输出中从未直接披露，但基础概要 ${\mathcal{A}}_{1},{\mathcal{A}}_{2},\ldots ,{\mathcal{A}}_{T}$ 会被披露，并且原则上每个 ${\mathcal{A}}_{i}$ 都可能泄露有关在构建 ${\mathcal{A}}_{i}$ 时从 ${\mathcal{D}}_{i}$ 中选择的查询的信息。因此，我们需要限制在相邻数据库上获得的概率分布之间的最大散度。这在技术上具有挑战性，因为给定 ${\mathcal{A}}_{i}$ 时，数据库在构建 ${\mathcal{D}}_{i + 1}$ 中起着非常重要的作用。

初始分布 ${\mathcal{D}}_{1}$ 将在 $\mathcal{Q}$ 上均匀分布。更新 ${\mathcal{D}}_{t}$ 的一种标准方法是，对于处理效果不佳的元素（在我们的例子中，即满足 $\left| {q\left( x\right)  - q\left( {A}_{t}\right) }\right|  > \lambda$ 的查询），将其权重增加一个固定因子，例如 $e$ ；对于处理效果良好的元素，将其权重减少相同的因子。（然后对权重进行归一化处理，使其总和为 1。）为了感受一下难度，设 $x = y \cup  \{ \xi \}$ ，并假设当数据库为 $y$ 时，${\mathcal{A}}_{t}$ 能很好地处理所有查询 $q$ ，但加入 $\xi$ 后，例如，会导致 1/10 的查询处理失败；也就是说，对于所有查询 $q$ 有 $\left| {q\left( y\right)  - q\left( {A}_{t}\right) }\right|  \leq  \lambda$ ，但对于某些 $\left| \mathcal{Q}\right| /{10}$ 查询有 $\left| {q\left( x\right)  - q\left( {A}_{t}\right) }\right|  > \lambda$ 。请注意，由于即使数据库为 $x$ 时，${\mathcal{A}}_{t}$ 仍能对 $9/{10}$ 的查询“处理得很好”，所以无论 $x,y$ 中哪个是真实数据集，都可以从基础清理器中返回 ${\mathcal{A}}_{t}$ 。我们关注的是更新的影响：当数据库为 $y$ 时，所有查询都能得到很好的处理，并且（归一化后）没有重新加权；但当数据库为 $x$ 时，会进行重新加权：十分之一的查询权重增加，其余十分之九的查询权重减少。这种重新加权的差异可能会在下次迭代中通过 ${\mathcal{A}}_{t + 1}$ 被检测到，${\mathcal{A}}_{t + 1}$ 是可观测的，并且根据数据库是 $x$ 还是 $y$ ，它将由从截然不同的分布中抽取的样本构建而成。

例如，假设我们从均匀分布 ${\mathcal{D}}_{1}$ 开始。那么 ${\mathcal{D}}_{2}^{\left( y\right) } = {\mathcal{D}}_{1}^{\left( y\right) }$ ，这里 ${\mathcal{D}}_{i}^{\left( z\right) }$ 表示当数据库为 $z$ 时第 $i$ 轮的分布。这是因为每个查询的权重都减少了 $e$ 倍，而这在归一化过程中会被消除。因此，在 ${\mathcal{D}}_{2}^{\left( y\right) }$ 中，每个 $q \in  \mathcal{Q}$ 被分配的权重为 $1/\left| \mathcal{Q}\right|$ 。相比之下，当数据库为 $x$ 时，“不满意”的查询具有归一化后的权重

$$
\frac{\frac{e}{\left| \mathcal{Q}\right| }}{\frac{9}{10}\frac{1}{\left| \mathcal{Q}\right| }\frac{1}{e} + \frac{1}{10}\frac{e}{\left| \mathcal{Q}\right| }}.
$$

考虑任何这样一个“不满意”的查询 $q$ 。比率 ${\mathcal{D}}_{2}^{\left( x\right) }\left( q\right) /{\mathcal{D}}_{2}^{\left( y\right) }\left( q\right)$ 由下式给出

$$
\frac{{\mathcal{D}}_{2}^{\left( x\right) }\left( q\right) }{{\mathcal{D}}_{2}^{\left( y\right) }\left( q\right) } = \frac{\frac{\frac{e}{\left| \mathcal{Q}\right| }}{\frac{9}{10}\frac{1}{\left| \mathcal{Q}\right| }\frac{1}{e} + \frac{1}{10}\frac{e}{\left| \mathcal{Q}\right| }}}{\frac{1}{\left| \mathcal{Q}\right| }}
$$

$$
 = \frac{10}{1 + \frac{9}{{e}^{2}}}\overset{\text{ def }}{ = }F \approx  {4.5085}.
$$

现在，$\ln F \approx  {1.506}$ ，并且即使基础生成器在第 2 轮中使用的查询选择没有明确公开，但它们可能可以从公开的结果 ${\mathcal{A}}_{2}$ 中检测到。因此，每个查询最多存在 1.506 的潜在隐私损失（当然，我们期望有抵消情况；我们只是试图解释问题的根源）。通过确保基础生成器使用的样本数量相对较少，可以部分解决这个问题，尽管我们仍然存在这样的问题：在多次迭代中，即使在相邻的数据库上，分布 ${\mathcal{D}}_{t}$ 也可能会有非常不同的演变。

解决方案是减弱重新加权过程。我们不再总是使用固定比例来增加权重（当答案“准确”时）或减少权重（当答案不准确时），而是为“准确性” $\left( \lambda \right)$ 和“不准确性” $(\lambda  + \mu$ 设置单独的阈值，对于一个适当选择的、随基础生成器输出的比特大小而缩放的 $\mu$ ；见下面的引理6.5）。误差低于或高于这些阈值的查询，其权重分别以 $e$ 为因子减小或增大。对于误差介于这两个阈值之间的查询，我们对权重变化的自然对数进行线性缩放： $1 - 2\left( {\left| {q\left( x\right)  - q\left( {A}_{t}\right) }\right|  - \lambda }\right) /\mu$ ，因此误差幅度超过 $\lambda  + \mu /2$ 的查询权重增加，而误差幅度小于 $\lambda  + \mu /2$ 的查询权重减小。

衰减缩放减少了任何个体对任何查询重新加权的影响。这是因为一个个体只能对查询的真实答案产生很小的影响，从而也只能对基础概要生成器输出 $q\left( {A}_{t}\right)  -$ 的准确性产生很小的影响，并且衰减将这个影响量除以一个参数 $\mu$ ，该参数将被选择以补偿在提升算法执行过程中从 $T$ 个分布中选择的（总共） ${kT}$ 个样本。这有助于确保隐私性。直观地说，我们将这些 ${kT}$ 个样本中的每一个都视为一个“微型机制”。我们首先界定任何一轮采样的隐私损失（声明6.4），然后通过组合定理界定累积损失。

“准确”和“不准确”阈值之间的差距 $\left( \mu \right)$ 越大，每个个体对查询权重的影响就越小。这意味着更大的差距对隐私性更有利。然而，对于准确性而言，大的差距是不利的。如果不准确性阈值很大，我们只能保证基础概要生成器非常不准确的查询在重新加权期间其权重会大幅增加。这会降低提升算法的准确性保证：误差大致等于“不准确性”阈值 $\left( {\lambda  + \mu }\right)$ 。

聚合。对于 $t \in  \left\lbrack  T\right\rbrack$ ，我们将运行基础生成器以获得概要 ${\mathcal{A}}_{t}$ 。这些概要将通过取中位数进行聚合：给定 ${\mathcal{A}}_{1},\ldots ,{\mathcal{A}}_{T}$ ，通过使用每个 ${\mathcal{A}}_{i}$ 计算 $q\left( x\right)$ 的 $T$ 个近似值，然后计算它们的中位数来估计量 $q\left( x\right)$ 。使用这种聚合方法，我们可以通过论证大多数 ${\mathcal{A}}_{i},1 \leq  i \leq  T$ 为查询 $q$ 提供 $\lambda  + \mu$ 的准确性（或更好）来证明查询 $q$ 的准确性。这意味着 $q\left( x\right)$ 的 $T$ 个近似值的中位数将在真实值的 $\lambda  + \mu$ 范围内。

## 符号说明。

1. 在算法的整个运行过程中，我们（显式或隐式地）跟踪几个变量。由 $q \in  \mathcal{Q}$ 索引的变量保存与查询集中的查询 $q$ 相关的信息。由 $t \in  \left\lbrack  T\right\rbrack$ 索引的变量，通常在第 $t$ 轮计算，将用于构建在时间段 $t + 1$ 用于采样的分布 ${\mathcal{D}}_{t + 1}$ 。

2. 对于谓词 $P$ ，如果谓词为真，我们使用 $\left\lbrack  \left\lbrack  P\right\rbrack  \right\rbrack$ 表示1；如果谓词为假，则表示0。

3. 算法中使用了一个最终的调优参数 $\alpha$。它的值将被选定（见下面的推论 6.3）为

$$
\alpha  = \alpha \left( \eta \right)  = \left( {1/2}\right) \ln \left( \frac{1 + {2\eta }}{1 - {2\eta }}\right) .
$$

该算法如图 6.1 所示。步骤 2(2b) 中的量 ${u}_{t,q}$ 是查询的新的、未归一化的权重。目前，我们令 $\alpha  = 1$（这样我们就可以忽略任何 $\alpha$ 因子）。设 ${a}_{j,q}$ 为第 $j,1 \leq  j \leq  t$ 轮权重变化的自然对数，则新的权重由下式给出：

$$
{u}_{t,q} \leftarrow  \exp \left( {-\mathop{\sum }\limits_{{j = 1}}^{t}{a}_{j,q}}\right) 
$$

因此，在上一步结束时，未归一化的权重为 ${u}_{t - 1,q} = \exp \left( {-\mathop{\sum }\limits_{{j = 1}}^{{t - 1}}{a}_{j,q}}\right)$，更新对应于乘以 ${e}^{-{a}_{j,t}}$。当和 $\mathop{\sum }\limits_{{j = 1}}^{t}{a}_{j,q}$ 很大时，权重很小。每当一个概要对 $q\left( x\right)$ 给出非常好的近似时，我们就给这个和加 1；如果近似只是中等程度的好（在 $\lambda$ 和

<!-- Media -->

---

查询的提升算法 $\left( {k,\lambda ,\eta ,\rho ,\mu ,T}\right)$

给定：数据库 $x \in  {\mathcal{X}}^{n}$，查询集 $\mathcal{Q}$，其中每个 $q \in  \mathcal{Q}$ 是一个函数 $q : {X}^{n} \rightarrow  \mathbb{R}$，其敏感度至多为

$\rho$ .

将 ${\mathcal{D}}_{1}$ 初始化为 $\mathcal{Q}$ 上的均匀分布。

对于 $t = 1,\ldots ,T$：

1. 从 ${\mathcal{D}}_{t}$ 中独立随机抽取一个包含 $k$ 个样本的序列 ${S}_{t} \subseteq  \mathcal{Q}$。

运行基础概要生成器，计算一个概要 ${\mathcal{A}}_{t} : \mathcal{Q} \rightarrow  \mathbb{R}$，该概要以高概率（w.h.p.）对

${\mathcal{D}}_{t}$ 中至少 $1/2 + \eta$ 的质量是准确的。

2. 对查询重新加权。对于每个 $q \in  \mathcal{Q}$：

(a) 如果 ${\mathcal{A}}_{t}$ 是 $\lambda$ -准确的，则 ${a}_{t,q} \leftarrow  1$

如果 ${\mathcal{A}}_{t}$ 是 $\left( {\lambda  + \mu }\right)$ -不准确的，则 ${a}_{t,q} \leftarrow   - 1$

否则，设 ${d}_{q,t} = \left| {q\left( x\right)  - {\mathcal{A}}_{t}\left( q\right) }\right|$ 为 ${\mathcal{A}}_{t}$ 在 $q$ 上的误差（介于 $\lambda$ 和 $\lambda  + \mu$ 之间）：

$$
{a}_{t,q} \leftarrow  1 - 2\left( {{d}_{t,q} - \lambda }\right) /\mu 
$$

(b) ${u}_{t,q} \leftarrow  \exp \left( {-\alpha  \cdot  \mathop{\sum }\limits_{{j = 1}}^{t}{a}_{j,q}}\right)$，其中 $\alpha  = \left( {1/2}\right) \ln \left( {\left( {1 + {2\eta }}\right) /\left( {1 - {2\eta }}\right) }\right)$。

3. 重新归一化：

$$
{Z}_{t} \leftarrow  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{u}_{t,q}
$$

$$
{D}_{t + 1}\left\lbrack  q\right\rbrack   = {u}_{t,q}/{Z}_{t}
$$

输出最终答案的数据结构 $\mathcal{A} = \left( {{\mathcal{A}}_{1},\ldots ,{\mathcal{A}}_{T}}\right)$ 。对于 $q \in  \mathcal{Q}$ ：

$$
\mathcal{A}\left( q\right)  = \operatorname{median}\left\{  {{\mathcal{A}}_{1}\left( q\right) ,\ldots ,{\mathcal{A}}_{T}\left( q\right) }\right\}  
$$

---

图 6.1：查询的提升算法。

<!-- Media -->

$\lambda  + \mu /2$ 时，我们添加一个正值，但小于 1。相反，当概要非常差（准确率低于 $\lambda  + \mu$ ）时，我们减去 1；当它勉强可接受（介于 $\lambda  + \mu /2$ 和 $\lambda  + \mu$ 之间）时，我们减去一个较小的值。

在下面的定理中，我们看到由 ${\varepsilon }_{\text{sample }}$ 表示的因采样导致的隐私损失与准确和不准确阈值之间的差距 $\mu$ 呈反比关系。

定理 6.1。设 $\mathcal{Q}$ 是一个敏感度至多为 $\rho$ 的查询族。对于适当的参数设置，经过 $T = \log \left| \mathcal{Q}\right| /{\eta }^{2}$ 轮迭代，图 6.1 中的算法是一种准确且具有差分隐私性的查询提升算法：

1. 当使用 $\left( {k,\lambda ,\eta ,\beta }\right)$ -基概要生成器实例化时，提升算法的输出以至少 $1 - {T\beta }$ 的概率为 $\mathcal{Q}$ 中的所有查询提供 $\left( {\lambda  + \mu }\right)$ -准确的答案，其中

$$
\mu  \in  O\left( {\left( {\left( {{\log }^{3/2}\left| Q\right| }\right) \sqrt{k}\sqrt{\log \left( {1/\beta }\right) }\rho }\right) /\left( {{\varepsilon }_{\text{sample }} \cdot  {\eta }^{3}}\right) }\right) . \tag{6.2}
$$

2. 如果基概要生成器是 $\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ -差分隐私的，那么提升算法是 $\left( {{\varepsilon }_{\text{sample }} + T \cdot  {\varepsilon }_{\text{base }},{\delta }_{\text{sample }} + }\right.$ $T{\delta }_{\text{base }}$ )-差分隐私的。

允许常数 $\eta$ 被纳入大 O 符号中，为简单起见取 $\rho  = 1$ ，我们得到 $\mu  = O\left( \left( {\left( {{\log }^{3/2}\left| Q\right| }\right) \sqrt{k}}\right) \right.$ $\left. \sqrt{\log \left( {1/\beta }\right) }\right) /{\varepsilon }_{\text{sample }}$ 。因此，我们看到减少基清理器所需的输入查询数量 $k$ 可以提高输出质量。类似地，从定理的完整表述中，我们看到提高基清理器的泛化能力，这对应于 $\eta$ 具有更大的值（更大的“强多数”），也可以提高准确性。

定理 6.1 的证明。我们首先证明准确性，然后证明隐私性。

我们引入符号 ${a}_{t,q}^{ - }$ 和 ${a}_{t,q}^{ + }$ ，满足

1. ${a}_{t,q}^{ - },{a}_{t,q}^{ + } \in  \{  - 1,1\}$ ；以及

2. ${a}_{t,q}^{ - } \leq  {a}_{t,q} \leq  {a}_{t,q}^{ + }$ .

回想一下， ${a}_{t,q}$ 越大，表示概要 ${\mathcal{A}}_{t}$ 对 $q\left( x\right)$ 的近似质量越高。

1. 如果 ${\mathcal{A}}_{t}$ 在 $q$ 上是 $\lambda$ -准确的，则 ${a}_{t,q}^{ - }$ 为 1，否则为 -1。为了验证 ${a}_{t,q}^{ - } \leq  {a}_{t,q}$ ，注意如果 ${a}_{t,q}^{ - } = 1$ ，那么 ${\mathcal{A}}_{t}$ 对于 $q$ 是 $\lambda$ -准确的，因此根据定义 ${a}_{t,q} = 1$ 也成立。如果相反我们有 ${a}_{t,q}^{ - } =  - 1$ ，那么由于我们总是有 ${a}_{t,q} \in  \left\lbrack  {-1,1}\right\rbrack$ ，我们就完成了证明。

我们将使用${a}_{t,q}^{ - }$来对基础生成器输出质量的一个度量进行下界估计。根据基础生成器的承诺，对于${\mathcal{D}}_{t}$的至少$1/2 + \eta$比例的质量，${\mathcal{A}}_{t}$是$\lambda$ - 准确的。因此，

$$
{r}_{t} \triangleq  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{\mathcal{D}}_{t}\left\lbrack  q\right\rbrack   \cdot  {a}_{t,q}^{ - } \geq  \left( {1/2 + \eta }\right)  - \left( {1/2 - \eta }\right)  = {2\eta }. \tag{6.3}
$$

2. 如果对于$q$，${\mathcal{A}}_{t}$是$\left( {\lambda  + \mu }\right)$ - 不准确的，则${a}_{t,q}^{ + }$为 - 1；否则为 1。为了验证${a}_{t,q} \leq  {a}_{t,q}^{ + }$，注意到如果${a}_{t,q}^{ + } =  - 1$，那么对于$q$，${\mathcal{A}}_{t}$是$\left( {\lambda  + \mu }\right)$ - 不准确的，所以根据定义${a}_{t,q} =  - 1$也成立。如果相反${a}_{t,q}^{ + } = 1$，那么由于我们始终有${a}_{t,q} \in  \left\lbrack  {-1,1}\right\rbrack$，我们就完成了验证。

因此，当且仅当对于$q$，${\mathcal{A}}_{t}$至少具有最低限度的足够准确性时，${a}_{t,q}^{ + }$才为正。我们将使用${a}_{t,q}^{ + }$来证明聚合的准确性。当我们对${a}_{t,q}^{ + }$的值求和时，当且仅当大多数${\mathcal{A}}_{t}$提供了可接受的——即，对$q\left( x\right)$的误差在$\lambda  + \mu$范围内的近似值——时，我们得到一个正数。在这种情况下，中位数将在$\lambda  + \mu$范围内。

引理 6.2。经过$T$轮提升后，除了${T\beta }$的概率外，除了$\exp \left( {-{\eta }^{2}T}\right)$比例的查询外，所有查询的答案都是$\left( {\lambda  + \mu }\right)$ - 准确的。

证明。在提升的最后一轮，我们有：

$$
{\mathcal{D}}_{T + 1}\left\lbrack  q\right\rbrack   = \frac{{u}_{T,q}}{{Z}_{T}}. \tag{6.4}
$$

由于${a}_{t,q} \leq  {a}_{t,q}^{ + }$，我们有：

$$
{u}_{T,q}^{ + } \triangleq  {e}^{-\alpha \mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t,q}^{ + }} \leq  {e}^{-\alpha \mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t,q}} = {u}_{T,q}. \tag{6.5}
$$

（上标“ + ”提醒我们，这个未加权的值是使用项${a}_{t,q}^{ + }$计算得到的。）注意，我们始终有${u}_{T,q}^{ + } \geq  0$。结合方程(6.4)和(6.5)，对于所有$q \in  \mathcal{Q}$：

$$
{\mathcal{D}}_{T + 1}\left\lbrack  q\right\rbrack   \geq  \frac{{u}_{T,q}^{ + }}{{Z}_{T}}. \tag{6.6}
$$

回顾一下，$\left\lbrack  \left\lbrack  P\right\rbrack  \right\rbrack$表示一个布尔变量，当且仅当谓词$P$为真时，其值为 1。我们现在来考察值$\left\lbrack  \left\lbrack  {\mathcal{A}\text{is}\left( {\lambda  + \mu }\right) \text{-inaccurate for}q}\right\rbrack  \right\rbrack$。如果这个谓词为 1，那么必然是大多数${\left\{  {\mathcal{A}}_{j}\right\}  }_{j = 1}^{T}$是$\left( {\lambda  + \mu }\right)$ - 不准确的，否则它们的中位数将是$\left( {\lambda  + \mu }\right)$ - 准确的。

从我们对$\mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t,q}^{ + }$符号意义的讨论中，我们有：

$$
\mathcal{A}\text{is}\left( {\lambda  + \mu }\right) \text{-inaccurate for}q \Rightarrow  \mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t,q}^{ + } \leq  0
$$

$$
 \Leftrightarrow  {e}^{-\alpha \mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t,q}^{ + }} \geq  1
$$

$$
 \Leftrightarrow  {u}_{T,q}^{ + } \geq  1
$$

由于${u}_{T,q}^{ + } \geq  0$，我们得出结论：

$$
\left\lbrack  \left\lbrack  {\mathcal{A}\text{ is }\left( {\lambda  + \mu }\right) \text{-inaccurate for }q}\right\rbrack  \right\rbrack   \leq  {u}_{T,q}^{ + }
$$

将此与方程(6.6)结合起来得到：

$$
\frac{1}{\left| \mathcal{Q}\right| } \cdot  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}\left\lbrack  \left\lbrack  {\mathcal{A}\text{ is }\left( {\lambda  + \mu }\right) \text{-inaccurate for }q}\right\rbrack  \right\rbrack   \leq  \frac{1}{\left| \mathcal{Q}\right| } \cdot  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{u}_{T,q}^{ + }
$$

$$
 \leq  \frac{1}{\left| \mathcal{Q}\right| } \cdot  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{\mathcal{D}}_{T + 1}\left\lbrack  q\right\rbrack   \cdot  {Z}_{T}
$$

$$
 = \frac{{Z}_{T}}{\left| \mathcal{Q}\right| }\text{.}
$$

因此，以下断言完成了证明：

断言 6.3。在提升的第$t$轮，除了${t\beta }$的概率外：

$$
{Z}_{t} \leq  \exp \left( {-{\eta }^{2} \cdot  t}\right)  \cdot  \left| \mathcal{Q}\right| 
$$

证明。根据基本概要生成器的定义，除了概率为 $\beta$ 的情况外，所生成的概要对于分布 ${\mathcal{D}}_{t}$ 的至少 $\left( {1/2 + \eta }\right)$ 比例的质量是 $\lambda$ -准确的。回顾一下，当且仅当 ${\mathcal{A}}_{t}$ 在 $q$ 上是 $\lambda$ -准确时，${a}_{t,q}^{ - } \in  \{  - 1,1\}$ 为 1，并且 ${a}_{t,q}^{ - } \leq  {a}_{t,q}$，再回顾一下方程 (6.3) 中定义的量 ${r}_{t} \triangleq  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{\mathcal{D}}_{t}\left\lbrack  q\right\rbrack   \cdot  {a}_{t,q}^{ - }$。如上所述，${r}_{t}$ 衡量了第 $t$ 轮中基本概要生成器的“成功”情况，这里的“成功”指的是更严格的 $\lambda$ -准确性概念。如方程 (6.3) 所总结的，如果 ${\mathcal{D}}_{t}$ 的 $\left( {1/2 + \eta }\right)$ 比例的质量以 $\lambda$ -准确性计算，那么 ${r}_{t} \geq  {2\eta }$。现在还需注意，对于 $t \in  \left\lbrack  T\right\rbrack$，假设基本清理器在第 $t$ 轮没有失败：

$$
{Z}_{t} = \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{u}_{t,q}
$$

$$
 = \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{u}_{t - 1,q} \cdot  {e}^{-\alpha  \cdot  {a}_{t,q}}
$$

$$
 = \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{Z}_{t - 1} \cdot  {\mathcal{D}}_{t}\left\lbrack  q\right\rbrack   \cdot  {e}^{-\alpha  \cdot  {a}_{t,q}}
$$

$$
 \leq  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{Z}_{t - 1} \cdot  {\mathcal{D}}_{t}\left\lbrack  q\right\rbrack   \cdot  {e}^{-\alpha  \cdot  {a}_{t,q}^{ - }}
$$

$$
 = {Z}_{t - 1} \cdot  \mathop{\sum }\limits_{{q \in  \mathcal{Q}}}{\mathcal{D}}_{t}\left\lbrack  q\right\rbrack   \cdot  \left( {\left( \frac{1 + {a}_{t,q}^{ - }}{2}\right)  \cdot  {e}^{-\alpha } + \left( \frac{1 - {a}_{t,q}^{ - }}{2}\right)  \cdot  {e}^{\alpha }}\right) 
$$

(情况分析)

$$
 = \frac{{Z}_{t - 1}}{2}\left\lbrack  {\left( {{e}^{\alpha } + {e}^{-\alpha }}\right)  + {r}_{t}\left( {{e}^{-\alpha } - {e}^{\alpha }}\right) }\right\rbrack  
$$

$$
 \leq  \frac{{Z}_{t - 1}}{2}\left\lbrack  {\left( {{e}^{\alpha } + {e}^{-\alpha }}\right)  + {2\eta }\left( {{e}^{-\alpha } - {e}^{\alpha }}\right) }\right\rbrack  \left( {{r}_{t} \geq  {2\eta }\text{ and }\left( {{e}^{-\alpha } - {e}^{\alpha }}\right)  \leq  0}\right) 
$$

通过简单的微积分，我们可以看到 $\left( {{e}^{\alpha } + {e}^{-\alpha }}\right)  + {2\eta }\left( {{e}^{-\alpha } - {e}^{\alpha }}\right)$ 被最小化

当

$$
\alpha  = \left( {1/2}\right) \ln \left( \frac{1 + {2\eta }}{1 - {2\eta }}\right) .
$$

将其代入递推式，我们得到

$$
{Z}_{t} \leq  {\left( \sqrt{1 - 4{\eta }^{2}}\right) }^{t}\left| \mathcal{Q}\right|  \leq  \exp \left( {-2{\eta }^{2}t}\right) \left| \mathcal{Q}\right| .
$$

这就完成了引理 6.2 的证明。

该引理意味着，通过设置可以同时实现所有查询的准确性

$$
T > \frac{\ln \left| \mathcal{Q}\right| }{{\eta }^{2}}
$$

隐私性。我们将证明，在保留差分隐私的同时可以输出整个序列 $\left( {{S}_{1},{\mathcal{A}}_{1},\ldots ,{S}_{T},{\mathcal{A}}_{T}}\right)$。请注意，这比我们所需的更强 —— 我们实际上并不输出集合 ${S}_{1},\ldots ,{S}_{T}$。根据我们的自适应组合定理，每个 ${\mathcal{A}}_{i}$ 的隐私性将由基本概要生成器的隐私保证以及 ${S}_{i - 1}$ 以差分隐私的方式计算这一事实来保证。因此，只需证明在 $\left( {{S}_{1},{\mathcal{A}}_{1},\ldots ,{S}_{i},{\mathcal{A}}_{i}}\right)$ 是差分隐私的情况下，${S}_{i + 1}$ 也是差分隐私的即可。然后，我们可以使用我们的组合定理来组合隐私参数，以计算最终的保证。

引理 6.4。设 ${\varepsilon }^{ * } = \frac{4\alpha T\rho }{\mu }$。对于所有 $i \in  \left\lbrack  T\right\rbrack$，一旦 $\left( {{S}_{1},{\mathcal{A}}_{1},\ldots ,{S}_{i},{\mathcal{A}}_{i}}\right)$ 固定，${S}_{i + 1}$ 中每个元素的计算都是 $\left( {{\varepsilon }^{ * },0}\right)$ -差分隐私的。

证明。固定${\mathcal{A}}_{1},\ldots ,{\mathcal{A}}_{i}$，对于每个$j \leq  i$，量${d}_{q,j}$的敏感度为$\rho$，因为${\mathcal{A}}_{j}\left( q\right)$与数据库无关（因为${\mathcal{A}}_{j}$是固定的），并且每个$q \in  \mathcal{Q}$的敏感度都有界于$\rho$。因此，对于每个$j \leq  i$，根据构造可知${a}_{j,q}$的敏感度为${2\rho }/\mu$，所以

$$
{g}_{i}\left( q\right) \overset{\text{ def }}{ = }\mathop{\sum }\limits_{{j = 1}}^{i}{a}_{j,q}
$$

其敏感度至多为${2i\rho }/\mu  \leq  {2T\rho }/\mu$。那么$\Delta {g}_{i}\overset{\text{ def }}{ = }{2T\rho }/\mu$是${g}_{i}$敏感度的一个上界。

为了论证隐私性，我们将证明为${S}_{i + 1}$选择查询是指数机制的一个实例。将$- {g}_{i}\left( q\right)$视为在第$i + 1$轮选择过程中查询$q$的效用。指数机制表明，为了实现$\left( {{\varepsilon }^{ * },0}\right)$-差分隐私，我们应该以与

$$
\exp \left( {-{g}_{i}\left( q\right) \frac{{\varepsilon }^{ * }}{{2\Delta }{g}_{i}}}\right) .
$$

成比例的概率选择${e}^{-\alpha {g}_{i}\left( q\right) }$。由于${\varepsilon }^{ * }/{2\Delta }{g}_{i} = \alpha$且算法以与${e}^{-\alpha {g}_{i}\left( q\right) }$成比例的概率选择$q$，我们看到这正是算法所做的！

我们通过将每次查询选择视为一个“小机制”来界定发布${S}_{i}$的隐私损失，在$T$轮提升过程中，该“小机制”被调用${kT}$次。根据引理6.4，每个小机制都是$\left( {{4\alpha T\rho }/\mu ,0}\right)$-差分隐私的。根据定理3.20，对于所有$\beta  > 0$，${kT}$个机制（每个机制都是$\left( {{\alpha 4T\rho }/\mu ,0}\right)$-差分隐私的）的组合是$\left( {{\varepsilon }_{\text{sample }},{\delta }_{\text{sample }}}\right)$-差分隐私的，其中

$$
{\varepsilon }_{\text{sample }}\overset{\text{ def }}{ = }\sqrt{{2kT}\log \left( {1/{\delta }_{\text{sample }}}\right) }\left( {{\alpha 4T\rho }/\mu }\right)  + {kT}{\left( \frac{\alpha 4T\rho }{\mu }\right) }^{2}. \tag{6.7}
$$

我们的总隐私损失来自对基础清理器的$T$次调用的组合以及来自${kT}$个样本的累积损失。我们得出结论，整个提升算法是：$\left( {{\varepsilon }_{\text{boost }},{\delta }_{\text{boost }}}\right)$-差分隐私的，其中

$$
{\varepsilon }_{\text{boost }} = T{\varepsilon }_{\text{base }} + {\varepsilon }_{\text{sample }}
$$

$$
{\delta }_{\text{boost }} = T{\delta }_{\text{base }} + {\delta }_{\text{sample }}
$$

为了得到定理陈述中所声称的参数，我们可以取：

$$
\mu  \in  O\left( {\left( {{T}^{3/2}\sqrt{k}\sqrt{\log \left( {1/\beta }\right) }{\alpha \rho }}\right) /{\varepsilon }_{\text{sample }}}\right) . \tag{6.8}
$$

### 6.2 基础概要生成器

算法SmallDB（第4节）基于这样一种见解：随机选择的一小部分数据库行能够为大量分数计数查询提供良好的答案。本节描述的基础概要生成器有一个类似的见解：一个能对一小部分查询的答案给出良好近似的小概要，也能对大多数查询给出良好近似。这两者都是泛化界的实例。在本节的其余部分，我们首先证明一个泛化界，然后用它来构造差分基础概要生成器。

#### 6.2.1 一个泛化界

我们有一个关于待近似的大量查询集合$\mathcal{Q}$的分布$\mathcal{D}$。下面的引理表明，一个足够小的概要（synopsis），若能对根据$\mathcal{Q}$上的分布$\mathcal{D}$随机选取的查询子集$S \subset  \mathcal{Q}$的答案给出足够好的近似，那么在$S$的选择上以高概率而言，它也能对$\mathcal{Q}$中大多数查询的答案（即，以$D$为权重的$\mathcal{Q}$的大部分质量）给出良好的近似。当然，为了使概要具有意义，它必须包含一种为$\mathcal{Q}$中所有查询提供答案的方法，而不仅仅是作为输入接收的子集$S \subseteq  \mathcal{Q}$。我们在6.2.2节和定理6.6中描述的特定生成器将生成合成数据库；要回答任何查询，只需将查询应用于合成数据库即可，但引理将以最一般的形式陈述。

令$R\left( {y,q}\right)$表示概要$y$（用作重建过程的输入时）对查询$q$给出的答案。如果$\mathop{\max }\limits_{{q \in  S}}\left| {R\left( {y,q}\right)  - q\left( x\right) }\right|  \leq  \lambda$，则概要${y\lambda }$相对于查询集合$S$与数据库$x$拟合。令$\left| y\right|$表示表示$y$所需的比特数。由于我们的概要将是合成数据库，因此对于某个适当选择的全域元素数量$N$，有$\left| y\right|  = N{\log }_{2}\left| \mathcal{X}\right|$。泛化界表明，如果${y\lambda }$相对于从分布$\mathcal{D}$中随机选择的足够大（大于$\left| y\right|$）的查询集合$S$与$x$拟合，那么以高概率而言，${y\lambda }$对于$\mathcal{D}$的大部分质量与$x$拟合。

引理6.5。令$\mathcal{D}$为查询集合$\mathcal{Q} =$ $\left\{  {q : {\mathcal{X}}^{ * } \rightarrow  \mathbb{R}}\right\}$上的任意分布。对于所有$m \in  \mathcal{N},\gamma  \in  \left( {0,1}\right) ,\eta  \in  \lbrack 0,1/2)$，令$a =$ $2\left( {\log \left( {1/\gamma }\right)  + m}\right) /\left( {m\left( {1 - {2\eta }}\right) }\right)$。那么，在$S \sim  {\mathcal{D}}^{a \cdot  m}$的选择上以至少$1 - \gamma$的概率，每个大小至多为$m$比特且相对于查询集合$S$与$x$ $\lambda$ -拟合的概要$y$，也相对于$\mathcal{D}$的至少$\left( {1/2 + \eta }\right)$ -部分与$x$ $\lambda$ -拟合。

在证明引理之前，我们注意到$a$是一个压缩因子：我们将${am}$个查询的答案压缩到一个$m$位的输出中，因此更大的$a$对应着更高的压缩率。通常，这意味着更好的泛化能力，实际上我们可以看到，如果$a$更大，在固定$m$和$\gamma$的情况下，我们可以得到更大的$\eta$。引理还表明，对于任何给定的输出大小$m$，为了在$\mathcal{D}$的大部分$\left( {1/2 + \eta \text{fraction}}\right)$上获得良好输出所需的输入查询数量仅为$O\left( {\log \left( {1/\gamma }\right)  + m}\right)$。这很有趣，因为基础生成器所需的查询数量$k$越少，通过对${kT}$个查询进行采样导致的隐私损失${\varepsilon }_{\text{sample }}$（以及它与松弛度$\mu$的反比关系，见公式6.7），会提高提升算法输出的准确性。

引理6.5的证明。固定一组根据${\mathcal{D}}^{a \cdot  m}$独立选择的查询$S \subset  \mathcal{Q}$。考察任意一个$m$位的概要$y$。注意，$y$由一个$m$位的字符串描述。如果对于$\mathcal{D}$中至少$\left( {\log \left( {1/\gamma }\right)  + m}\right) /\left( {a \cdot  m}\right)$的比例，有$\left| {R\left( {y,q}\right)  - q\left( x\right) }\right|  >$ $\lambda$，我们称$y$是“坏的”，这意味着$\mathop{\Pr }\limits_{{q \sim  \mathcal{D}}}\left\lbrack  {\left| {R\left( {y,q}\right)  - q\left( x\right) }\right|  > \lambda }\right\rbrack   \geq  \left( {\log \left( {1/\gamma }\right)  + m}\right) /\left( {a \cdot  m}\right) .$

换句话说，如果存在一个分数权重至少为 $\left( {\log \left( {1/\gamma }\right)  + m}\right) /\left( {a \cdot  m}\right)$ 的集合 ${Q}_{y} \subset  \mathcal{Q}$，使得对于 $q \in  {Q}_{y}$ 有 $\left| {R\left( {y,q}\right)  - q\left( x\right) }\right|  > \lambda$，那么 $y$ 就是不好的。对于这样的 $y$，$y$ 对每个 $q \in  S$ 都给出 $\lambda$ 精度答案的概率是多少呢？这恰好是 $S$ 中的查询都不在 ${Q}_{y}$ 中的概率，即

$$
{\left( 1 - \left( \log \left( 1/\gamma \right)  + m\right) /\left( a \cdot  m\right) \right) }^{a \cdot  m} \leq  {e}^{-\left( {\log \left( {1/\gamma }\right)  + m}\right) } \leq  \gamma  \cdot  {2}^{-m}
$$

对 $y$ 的所有 ${2}^{m}$ 种可能选择取并集界，存在一个 $m$ 位概要 $y$，它对 $S$ 中的所有查询都准确，但对一个分数权重为 $\left( {\log \left( {1/\beta }\right)  + m}\right) /\left( {a \cdot  m}\right)$ 的集合不准确的概率至多为 $\gamma$。令 $k = {am} = \left| S\right|$，我们可以看到，只需满足

$$
a > \frac{2\left( {\log \left( {1/\gamma }\right)  + m}\right) }{m \cdot  \left( {1 - {2\eta }}\right) }. \tag{6.9}
$$

这个简单的引理非常强大。它告诉我们，在第 $t$ 轮构建基础生成器时，我们只需要担心为从 ${\mathcal{D}}_{t}$ 中采样的一小部分随机查询提供良好的答案；对 ${\mathcal{D}}_{t}$ 中的大多数查询表现良好会自动实现！

#### 6.2.2 基础生成器

我们的第一个生成器采用暴力方法。根据分布 $\mathcal{D}$ 独立采样出一个包含 $k$ 个查询的集合 $S$ 后，基础生成器将通过拉普拉斯机制为 $S$ 中的所有查询生成带噪声的答案。然后，该算法不再使用实际数据库，而是搜索大小为 $n$ 的任何数据库，使得这些带噪声的答案足够接近，并输出这个数据库。由于拉普拉斯机制的 $k$ 次调用之后的所有操作都属于后处理，因此隐私性是直接的。因此，隐私损失的唯一来源是拉普拉斯机制的这 $k$ 次调用的累积损失，我们知道如何通过组合定理来分析这种损失。实用性将源于拉普拉斯机制的实用性——即我们不太可能在哪怕一个查询上出现“非常大”的误差——再加上真实数据库 $x$ 是一个符合这些带噪声响应的 $n$ 元素数据库这一事实。

---

<!-- Footnote -->

${}^{1}$ 这个论证假设数据库的大小 $n$ 是已知的。或者，我们可以包含一个形式为“数据库中有多少行？”的带噪声查询，并对大小接近该查询响应的所有数据库进行穷举搜索。

<!-- Footnote -->

---

定理 6.6（任意查询的基础概要生成器）。对于任何数据全域 $\mathcal{X}$、数据库大小 $n$ 以及敏感度至多为 $\rho$ 的查询类 $\mathcal{Q} : \left\{  {{\mathcal{X}}^{ * } \rightarrow  \mathbb{R}}\right\}$，对于任何 ${\varepsilon }_{\text{base }},{\delta }_{\text{base }} > 0$，存在一个 $\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ -差分隐私的 $\left( {k,\lambda ,\eta  = 1/3,\beta }\right)$ -基础概要生成器用于 $\mathcal{Q}$，其中 $k = {am} > 6\left( {m + \log \left( {2/\beta }\right) }\right)  = 6\left( {n\log \left| \mathcal{X}\right|  + \log \left( {2/\beta }\right) }\right)$ 且 $\lambda  > {2b}\left( {\log k + \log \left( {2/\beta }\right) }\right)$，其中 $b = \rho \sqrt{{am}\log \left( {1/{\delta }_{\text{base }}}\right) }/{\varepsilon }_{\text{base }}$。

生成器的运行时间为

$$
{\left| \mathcal{X}\right| }^{n} \cdot  \operatorname{poly}\left( {n,\log \left( {1/\beta }\right) ,\log \left( {1/{\varepsilon }_{\text{base }}}\right) ,\log \left( {1/{\delta }_{\text{base }}}\right) }\right) .
$$

证明。我们首先从高层次描述基础生成器，然后确定$k$和$\lambda$的值。基础生成器生成的概要$y$将是一个大小为$n$的合成数据库。因此$m = \left| y\right|  =$ $n \cdot  \log \left| \mathcal{X}\right|$。生成器首先根据$\mathcal{D}$独立采样选择一组包含$k$个查询的集合$S$。它使用拉普拉斯机制（Laplace mechanism）为每个查询$q \in  S$计算一个含噪答案，为每个真实答案添加一个从$\operatorname{Lap}\left( b\right)$中独立抽取的值，其中$b$的值将在后面确定。令$\{ \overset{⏜}{q\left( x\right) }{\} }_{q \in  \mathcal{Q}}$为含噪答案的集合。生成器枚举所有大小为$n$的数据库${\left| \mathcal{X}\right| }^{n}$，并输出字典序上第一个满足对于每个$q \in  S$都有$\left| {q\left( y\right)  - \overset{⏜}{q\left( x\right) }}\right|  \leq  \lambda /2$的数据库$y$。如果没有找到这样的数据库，则输出$\bot$，此时我们称生成器失败。注意，如果$\left| {\overset{⏜}{q\left( x\right) } - q\left( x\right) }\right|  < \lambda /2$且$\left| {q\left( y\right)  - \overset{⏜}{q\left( x\right) }}\right|  < \lambda /2$，那么$\left| {q\left( y\right)  - q\left( x\right) }\right|  < \lambda$。

我们的特定生成器有两个潜在的失败来源。一种可能性是$y$无法泛化，或者如引理6.5的证明中所定义的那样是“坏的”。第二种可能性是从拉普拉斯分布中抽取的某个样本的绝对值过大，这可能导致生成器失败。我们将选择参数，使得这两种事件各自发生的概率至多为$\beta /2$。

将$\eta  = 1/3$和$m = n\log \left| X\right|$代入方程6.9表明，取$a > 6\left( {1 + \log \left( {2/\beta }\right) /m}\right)$足以使由于$S$的选择而导致失败的概率被限制在$\beta /2$以内。因此，取$k = {am} > 6\left( {m + \log \left( {2/\beta }\right) }\right)  = 6\left( {n\log \left| \mathcal{X}\right|  + \log \left( {2/\beta }\right) }\right)$就足够了。

我们有$k$个敏感度至多为$\rho$的查询。使用参数为$b = 2\sqrt{{2k}\log \left( {1/{\delta }_{\text{base }}}\right) }\rho /{\varepsilon }_{\text{base }}$的拉普拉斯机制（Laplace mechanism）可确保每个查询产生的隐私损失至多为${\varepsilon }_{\text{base }}/\sqrt{{2k}\ln \left( {1/{\delta }_{\text{base }}}\right) }$，根据推论3.21，这可确保整个过程是$\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ - 差分隐私（differentially private）的。

我们将选择$\lambda$，使得从$\operatorname{Lap}\left( b\right)$中抽取的任何样本的绝对值超过$\lambda /2$的概率至多为$\beta /2$。在所有$k$次抽取的绝对值至多为$\lambda$这一事件的条件下，我们知道输入数据库本身将与我们的含噪答案$\lambda$ - 拟合，因此该过程不会失败。

回顾一下，拉普拉斯分布（Laplace distribution）的集中性质确保，从$\operatorname{Lap}\left( b\right)$中抽样时，至少以$1 - {e}^{t}$的概率，抽样值的绝对值会被${tb}$所界定。令$\lambda /2 = {tb}$，则给定抽样值的绝对值超过$\lambda /2$的概率被${e}^{-t} = {e}^{-\lambda /{2b}}$所界定。为确保$k$次抽样中没有一个抽样值的绝对值超过$\lambda /2$，根据联合界（union bound），只需满足

$$
k{e}^{-\lambda /{2b}} < \beta /2
$$

$$
 \Leftrightarrow  {e}^{\lambda /{2b}} > k\frac{2}{\beta }
$$

$$
 \Leftrightarrow  \lambda /2 > b\left( {\log k + \log \left( {2/\beta }\right) }\right) 
$$

$$
 \Leftrightarrow  \lambda  > {2b}\left( {\log k + \log \left( {2/\beta }\right) }\right) \text{.}
$$

线性查询的特殊情况。对于线性查询的特殊情况，可以避免对小型数据库进行暴力搜索。该技术所需的时间是关于$\left( {\left| \mathcal{Q}\right| ,\left| \mathcal{X}\right| ,n,\log \left( {1/\beta }\right) }\right)$的多项式时间。我们将专注于计数查询的情况并简述构造过程。

与任意查询的基础生成器的情况一样，线性查询的基础生成器首先根据$\mathcal{D}$选择一组包含$k = {am}$个查询的集合$S$，并使用拉普拉斯噪声（Laplace noise）计算带噪声的答案。然后，线性查询生成器对$S$运行一个合成器（syntheticizer），大致来说，该合成器会将任何能对任意查询集合$R$给出良好近似的概要转换为一个合成数据库，该数据库能在集合$R$上产生类似质量的近似结果。合成器的输入将是$S$中查询的带噪声值，即$R = S$。（回顾一下，当我们修改数据库大小时，我们总是从计数查询的分数形式来考虑：“数据库行中满足属性$P$的比例是多少？”）

生成的数据库可能非常大，这意味着它可能有很多行。然后，基础生成器仅对合成数据库的行进行${n}^{\prime } =$ $\left( {\log k\log \left( {1/\beta }\right) }\right) /{\alpha }^{2}$的子抽样，创建一个较小的合成数据库，该数据库至少以$1 - \beta$的概率相对于大型合成数据库给出的答案具有$\alpha$ - 精度。这产生了一个$m = \left( {\left( {\log k\log \left( {1/\beta }\right) }\right) /{\alpha }^{2}}\right) \log \left| \mathcal{X}\right|$位的概要，根据泛化引理（generalization lemma），在选择$k$个查询时，以$\left( {1 - \log \left( {1/\beta }\right) }\right)$的概率，该概要能在$\mathcal{Q}$的$\left( {1/2 + \eta }\right)$比例（由$\mathcal{D}$加权）上给出良好的答案。

与任意查询的基础生成器的情况一样，我们要求$k = {am} > 6\log \left( {1/\beta }\right)  + {6m}$。取${\alpha }^{2} = \left( {\log \mathcal{Q}}\right) /n$，我们得到

$$
k > 6\log \left( {1/\beta }\right)  + 6\frac{\log k\log \left( {1/\beta }\right) \log \left| \mathcal{X}\right| }{{\alpha }^{2}}
$$

$$
 = 6\log \left( {1/\beta }\right)  + {6n}\log k\log \left( {1/\beta }\right) \frac{\log \left| \mathcal{X}\right| }{\log \left| \mathcal{Q}\right| }.
$$

合成器并非平凡的。其性质由以下定理总结。

定理6.7。设$\mathcal{X}$为一个数据全域（data universe），$\mathcal{Q}$为一组分数计数查询，$A$为一个具有效用$\left( {\alpha ,\beta ,0}\right)$且输出任意的$\left( {\varepsilon ,\delta }\right)$ - 差分隐私（differentially private）概要生成器。那么存在一个$\left( {\varepsilon ,\delta }\right)$ - 差分隐私且具有效用$\left( {{3\alpha },\beta ,0}\right) .{A}^{\prime }$的合成器${A}^{\prime }$，它输出一个（可能很大的）合成数据库。其运行时间是$A$的运行时间和$\left( {\left| \mathcal{X}\right| ,\left| \mathcal{Q}\right| ,1/\alpha ,\log \left( {1/\beta }\right) }\right)$的多项式时间。

在我们的例子中，$A$ 是拉普拉斯机制（Laplace mechanism），而概要仅仅是一组含噪声的答案。组合定理表明，要使 $A$ 满足 $\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ -差分隐私（differentially private），拉普拉斯机制的参数应为 $\rho /\left( {{\varepsilon }_{\text{base }}/\sqrt{{2k}\log \left( {1/{\delta }_{\text{base }}}\right) }}\right)$。对于分数计数查询，敏感度为 $\rho  = 1/n$。

因此，当我们应用该定理时，我们将得到一个阶为 $\left( {\sqrt{k\log \left( {1/\beta }\right) }/{\varepsilon }_{\text{base }}}\right) \rho$ 的 $\alpha$。这里，$\rho$ 是敏感度。对于计数查询，其值为 1，但我们将转向分数计数查询，因此 $\rho  = 1/n$。

定理 6.7 的证明概要。运行 $A$ 以获得 $R$ 中所有查询的（差分隐私的）（分数）计数。然后，我们将使用线性规划来找到一个低权重的分数数据库，该数据库能近似这些分数计数，如下所述。最后，我们通过对分数计数进行舍入，将这个分数数据库转换为标准的合成数据库。

$A$ 的输出为每个查询 $q \in  \mathcal{Q}$ 产生一个分数计数。输入数据库 $x$ 不会再被访问，因此 ${A}^{\prime }$ 是 $\left( {\varepsilon ,\delta }\right)$ -差分隐私的。设 $v$ 为所得的计数向量，即 ${v}_{q}$ 是 $A$ 的输出在查询 $q$ 上给出的分数计数。以概率 $1 - \beta$，$v$ 中的所有条目都是 $\alpha$ -准确的。

一个近似这些计数的“分数”数据库 $z$ 可按如下方式获得。回顾数据库的直方图表示，其中对于全域 $\mathcal{X}$ 中的每个元素，直方图包含该元素在数据库中的实例数量。现在，对于每个 $i \in  \mathcal{X}$，我们引入一个变量 ${a}_{i} \geq  0$，它将“计数” $i$ 在分数数据库 $z$ 中的（分数）出现次数。我们将施加约束

$$
\mathop{\sum }\limits_{{i \in  \mathcal{X}}}{a}_{i} = 1
$$

我们将查询 $q$ 在 $z$ 中的计数表示为满足 $q$ 的项目 $i$ 的计数之和：

$$
\mathop{\sum }\limits_{{i \in  \mathcal{X}\text{ s.t. }q\left( i\right)  = 1}}{a}_{i}
$$

我们希望所有这些计数与 ${v}_{q}$ 中相应计数的加性 $\alpha$ 精度在一定范围内。将其写成线性不等式，我们得到：

$$
\left( {{v}_{q} - \alpha }\right) \mathop{\sum }\limits_{{i \in  \mathcal{X}}}{a}_{i} \leq  \mathop{\sum }\limits_{{i \in  \mathcal{X}\text{ s.t. }q\left( i\right)  = 1}}{a}_{i} \leq  \left( {{v}_{q} + \alpha }\right) \mathop{\sum }\limits_{{i \in  \mathcal{X}}}{a}_{i}.
$$

当这些计数相对于 ${v}_{c}$ 中的计数都是 $\alpha$ -准确时，同样（以概率 $1 - \beta$）它们相对于原始数据库 $x$ 上的真实计数都是 ${2\alpha }$ -准确的。

我们为每个查询编写一个具有两个此类约束（总共 $2\left| \mathcal{Q}\right|$ 个约束）的线性规划。${A}^{\prime }$ 试图找到这个线性规划的一个分数解。为了说明这样的解存在，观察到数据库 $x$ 本身与计数向量 $v$ 是 $\alpha$ -接近的，因此存在该线性规划的一个解（实际上甚至是一个整数解），因此 ${A}^{\prime }$ 将找到某个分数解。

我们得出结论，${A}^{\prime }$ 可以生成一个具有 $\left( {{2\alpha },\beta ,0}\right)$ -效用的分数数据库，但我们真正需要的是一个合成（整数）数据库。为了将分数数据库转换为整数数据库，对于 $i \in  \mathcal{X}$，我们将每个 ${a}_{i}$ 向下舍入到最接近的 $\alpha /\left| \mathcal{X}\right|$ 的倍数，这使得每个分数计数最多改变一个 $\alpha /\left| \mathcal{X}\right|$ 的加法因子，因此舍入后的计数具有 $\left( {{3\alpha },\beta ,0}\right)$ 效用。现在我们可以将舍入后的分数数据库（总权重为 1）视为一个大小至多为 $\left| \mathcal{X}\right| /\alpha$ 的（多项式）整数合成数据库。

回顾一下，在我们应用定理 6.7 时，我们将 $A$ 定义为添加参数为 $\rho /\left( {{\varepsilon }_{\text{base }}/\sqrt{{2k}\log \left( {1/{\delta }_{\text{base }}}\right) }}\right)$ 的拉普拉斯噪声的机制。我们有 $k$ 次抽样，因此通过取

$$
{\alpha }^{\prime } = \rho \sqrt{{2k}\log \left( {1/{\delta }_{\text{base }}}\right) }\left( {\log k + \log \left( {1/\beta }\right) }\right) 
$$

我们可知 $A$ 是 $\left( {{\alpha }^{\prime },\beta ,0}\right)$ -准确的。对于基础生成器，我们选择的误差为 ${\alpha }^{2} = \left( {\log \left| \mathcal{Q}\right| }\right) /n$。如果合成器的输出太大，我们进行子抽样

$$
{n}^{\prime } = \frac{\log \left| \mathcal{Q}\right| \log \left( {1/\beta }\right) }{{\alpha }^{2}} = \frac{\log k\log \left( {1/\beta }\right) }{{\alpha }^{2}}
$$

行。以概率 $1 - \beta$，得到的数据库能同时在所有概念上保持 $O\left( {\rho \sqrt{\left( {\log \left| \mathcal{Q}\right| }\right) /n} + \left( {\sqrt{{2k}\log \left( {1/{\delta }_{\text{base }}}\right) }/{\varepsilon }_{\text{base }}}\right) \left( {\log k + \log \left( {1/\beta }\right) }\right)  - }\right.$ 的准确性。

最后，如果查询 $S \in  {\mathcal{D}}^{k}$ 的选择不能实现良好的泛化，基础生成器可能会失败。根据我们选择的参数，这种情况发生的概率至多为 $\beta$，导致整个生成器的总失败概率为 ${3\beta }$。

定理 6.8（分数线性查询的基础生成器）。对于任何数据全域 $\mathcal{X}$、数据库大小 $n$ 以及分数线性查询类 $\mathcal{Q} : \left\{  {{\mathcal{X}}^{n} \rightarrow  \mathbb{R}}\right\}$（敏感度至多为 $1/n$），对于任何 ${\varepsilon }_{\text{base }},{\delta }_{\text{base }} > 0$，存在一个用于 $\mathcal{Q}$ 的 $\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ -差分隐私 $\left( {k,\lambda ,1/3,{3\beta }}\right)$ -基础概要生成器，其中

$$
k = O\left( \frac{n\log \left( \left| \mathcal{X}\right| \right) \log \left( {1/\beta }\right) }{\log \left| \mathcal{Q}\right| }\right) 
$$

$$
\lambda  = O\left( {\frac{\log \left( {1/\beta }\right) }{\sqrt{n}}\left( {\sqrt{\log \left| \mathcal{Q}\right| } + \sqrt{\frac{\log \left| \mathcal{X}\right| }{\log \left| \mathcal{Q}\right| }} \cdot  \frac{1}{{\varepsilon }_{\text{base }}}}\right) }\right) .
$$

基础生成器的运行时间为 $\operatorname{poly}\left( {\left| \mathcal{X}\right| ,n,\log \left( {1/\beta }\right) ,}\right.$ $\log \left( {1/{\varepsilon }_{\text{base }}}\right)$）。

这里使用的抽样界限与 SmallDB 机制构建中使用的界限相同，但参数不同。这里我们在一个复杂的提升算法中，针对一个非常小的查询集，将这些界限用于基础生成器；在那里我们将它们用于一次性生成一个具有巨大查询集的合成数据库。

#### 6.2.3 组合各要素

总误差来自 $\mu$ 的选择（见方程 6.2）以及基础生成器的准确性参数 $\lambda$。

让我们回顾一下定理 6.1：

定理6.9（定理6.1）。设$\mathcal{Q}$为一个敏感度至多为$\rho$的查询族。对于参数的适当设置，且经过$T = \log \left| \mathcal{Q}\right| /{\eta }^{2}$轮迭代，图6.1中的算法是一个准确且具有差分隐私性的查询增强算法：

1. 当使用一个$\left( {k,\lambda ,\eta ,\beta }\right)$ - 基础概要生成器实例化时，增强算法的输出以至少$1 - {T\beta }$的概率为$\mathcal{Q}$中的所有查询提供$\left( {\lambda  + \mu }\right)$ - 准确的答案，其中

$$
\mu  \in  O\left( {\left( {\left( {{\log }^{3/2}\left| Q\right| }\right) \sqrt{k}\sqrt{\log \left( {1/\beta }\right) }\rho }\right) /\left( {{\varepsilon }_{\text{sample }} \cdot  {\eta }^{3}}\right) }\right) . \tag{6.10}
$$

2. 如果基础概要生成器具有$\left( {{\varepsilon }_{\text{base }},{\delta }_{\text{base }}}\right)$ - 差分隐私性，那么增强算法具有$\left( {\left( {{\varepsilon }_{\text{sample }} + T \cdot  {\varepsilon }_{\text{base }}}\right) ,T(\beta  + }\right.$ ${\delta }_{\text{base }})$ - 差分隐私性。

根据等式6.7，

$$
{\varepsilon }_{\text{sample }}\overset{\text{ def }}{ = }\sqrt{{2kT}\log \left( {1/\beta }\right) }\left( {{\alpha 4T\rho }/\mu }\right)  + {kT}{\left( \frac{\alpha 4T\rho }{\mu }\right) }^{2},
$$

其中$\alpha  = \left( {1/2}\right) \left( {\ln \left( {1 + {2\eta }}\right) \left( {1 - {2\eta }}\right) }\right)  \in  O\left( 1\right)$。我们始终有$T =$ $\left( {\log \left| \mathcal{Q}\right| }\right) /{\eta }^{2}$，因此将此值代入上述等式，我们可以看到定理陈述中的边界是可以接受的。

$$
\mu  \in  O\left( {\left( {\left( {{\log }^{3/2}\left| Q\right| }\right) \sqrt{k}\sqrt{\log \left( {1/\beta }\right) }\rho }\right) /\left( {{\varepsilon }_{\text{sample }} \cdot  {\eta }^{3}}\right) }\right) 
$$

在定理陈述中该边界是可接受的。

对于任意查询的情况，当$\eta$为常数时，我们有

$$
\lambda  \in  O\left( {\frac{\rho }{{\varepsilon }_{\text{base }}}\left( {\sqrt{n\log \left| \mathcal{X}\right| \log \left( {1/{\delta }_{\text{base }}}\right) }\left( {\log \left( {n\log \left| \mathcal{X}\right| }\right)  + \log \left( {2/\beta }\right) }\right) }\right) }\right) .
$$

现在，${\varepsilon }_{\text{boost }} = T{\varepsilon }_{\text{base }} + {\varepsilon }_{\text{sample }}$。令这两项相等，即$T{\varepsilon }_{\text{base }} =$ ${\varepsilon }_{\text{boost }}/2 = {\varepsilon }_{\text{sample }}$，由此我们可以用${2T}/{\varepsilon }_{\text{boost }} = \left( {\log \left| \mathcal{Q}\right| /{\eta }^{2}}\right) /2{\varepsilon }_{\text{boost }}$替换$1/{\varepsilon }_{\text{base }}$项。现在我们关于$\lambda$和$\mu$的项具有相似的分母，因为$\eta$是常数。因此，我们可以得出总误差的边界为：

$$
\lambda  + \mu  \in  \widetilde{O}\left( \frac{\sqrt{n\log \left| \mathcal{X}\right| }\rho {\log }^{3/2}\left| \mathcal{Q}\right| {\left( \log \left( 1/\beta \right) \right) }^{3/2}}{{\varepsilon }_{\text{boost }}}\right) .
$$

通过类似的推理，对于分数计数查询的情况，我们得到

$$
\lambda  + \mu  \in  \widetilde{O}\left( \frac{\sqrt{\log \left| \mathcal{X}\right| }\log \left| \mathcal{Q}\right| \log {\left( 1/\beta \right) }^{3/2}}{{\varepsilon }_{\text{boost }}\sqrt{n}}\right) .
$$

为了转换为普通非分数计数查询的边界，我们乘以$n$得到

$$
\lambda  + \mu  \in  \widetilde{O}\left( \frac{\sqrt{n\log \left| \mathcal{X}\right| }\log \left| \mathcal{Q}\right| \log {\left( 1/\beta \right) }^{3/2}}{{\varepsilon }_{\text{boost }}}\right) .
$$

### 6.3 参考文献注释

增强算法（图6.1）是Schapire和Singer [78]提出的AdaBoost算法的一个变体。关于增强算法的精彩综述可参见Schapire [77]，关于增强算法的全面论述可参见Freund和Schapire [79]所著的教科书《增强算法》。本节涵盖的隐私增强算法归功于Dwork等人 [32]，其中还包含线性查询的基础生成器。而这个基础生成器又依赖于Dwork等人 [28]的合成器。特别地，定理6.7来自 [28]。Dwork、Rothblum和Vadhan也探讨了通常意义下的差分隐私增强算法。

## 7 当最坏情况敏感度不典型时

在本节中，我们简要描述两种通用技术，它们都能提供无条件的隐私保证，通常可以让数据分析师的工作更轻松，尤其是在处理具有任意或难以分析的最坏情况敏感度的函数时。当分析师由于某些外部原因有理由相信某些函数在实际应用中“通常”不敏感时，这些算法在计算这些函数时最为有用。

### 7.1 子采样与聚合

子采样与聚合（Subsample and Aggregate）技术产生了一种“强制”计算函数$f\left( x\right)$的方法，即使对于任意函数$f$，该计算也不敏感。证明隐私性将是微不足道的。准确性取决于函数$f$和特定数据集$x$的属性；特别是，如果在$f\left( S\right)$上能够以高概率准确估计$f\left( x\right)$，其中$S$是$x$中元素的一个随机子集，那么准确性应该是良好的。许多最大似然统计估计器在“典型”数据集上具有这种属性——这就是为什么这些估计器在实践中被采用的原因。

<!-- Media -->

<!-- figureText: $x$ ${x}_{\left( {k - 1}\right) \frac{n}{k} + 1},\ldots ,{x}_{n}$ ... ${z}_{k}$ ${SA}\left( x\right)$ ${x}_{\frac{n}{k} + 1},\ldots ,{x}_{\frac{2n}{k}}$ ${z}_{1}$ ${z}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195c1d9-6c32-7efb-8261-e859ff255e66_24.jpg?x=549&y=501&w=739&h=423&r=0"/>

图7.1：使用通用差分隐私聚合算法$\mathcal{M}$的子采样与聚合。

<!-- Media -->

在子采样与聚合中，数据库$x$的$n$行被随机划分为$m$个块${B}_{1},\ldots ,{B}_{m}$，每个块的大小为$n/m$。函数$f$在每个块上独立地精确计算，不添加噪声。然后，中间结果$f\left( {B}_{1}\right) ,\ldots ,f\left( {B}_{m}\right)$通过差分隐私聚合机制进行组合——典型的例子包括标准聚合，如$\alpha$ - 截尾均值（The $\alpha$ -trimmed mean）、温莎化均值（The Winsorized mean）和中位数，但没有限制——然后添加根据所讨论的聚合函数的敏感度进行缩放的拉普拉斯噪声；见图7.1。

子采样与聚合的关键观察结果是，任何单个元素最多只能影响一个块，因此最多只能影响单个$f\left( {B}_{i}\right)$的值。因此，更改任何个体的数据最多只能改变聚合函数的一个输入。即使$f$是任意的，分析人员可以选择聚合函数，因此可以自由选择一个不敏感的函数，前提是该选择与数据库无关！因此，隐私性是直接的：对于任何$\delta  \geq  0$和任何函数$f$，如果聚合机制$\mathcal{M}$是$\left( {\varepsilon ,\delta }\right)$ - 差分隐私的，那么当使用$f$和$\mathcal{M}3$实例化子采样与聚合技术时，该技术也是$\left( {\varepsilon ,\delta }\right)$ - 差分隐私的。

---

<!-- Footnote -->

${}^{1}$ $\alpha$ - 截尾均值是在丢弃输入的顶部和底部$\alpha$比例的数据之后的均值。

${}^{2}$ 温莎化均值与$\alpha$ - 截尾均值类似，不同之处在于，顶部和底部$\alpha$比例的数据不是被丢弃，而是被替换为剩余的最极端值。

<!-- Footnote -->

---

实用性则是另一回事，即使在数据丰富且大的随机子集很可能产生相似结果的情况下，也很难进行论证。例如，数据可能是高维空间中的带标签训练点，函数是逻辑回归，它产生一个向量$v$，并且当且仅当对于某个（例如，固定的）阈值$T$满足$p \cdot  v \geq  T$时，将点$p$标记为 +1。直观地说，如果样本足够丰富且具有代表性，那么所有块都应该产生相似的向量$v$。困难在于对聚合函数的最坏情况敏感度获得一个良好的界限——我们可能需要使用范围的大小作为备用。尽管如此，已知有一些不错的应用，特别是在统计估计器领域，例如，可以证明，在“一般正态性”假设下，可以在不损失统计效率（大致来说，随着样本数量的增加的准确性）的情况下实现隐私性。我们在这里不定义一般正态性，但请注意，符合这些假设的估计器包括高斯等“良好”参数分布族的最大似然估计器，以及线性回归和逻辑回归的最大似然估计器。

假设函数 $f$ 具有基数为 $m$ 的离散值域，例如 $\left\lbrack  m\right\rbrack$。在这种情况下，子采样与聚合（Subsample and Aggregate）方法需要对从 $\left\lbrack  m\right\rbrack$ 中抽取的一组 $b$ 个元素进行聚合，我们可以使用带噪声的最大参数报告（Report Noisy Arg - Max）方法来找出最受欢迎的结果。即使中间结果完全一致，这种聚合方法也需要 $b \geq  \log m$ 才能得到有意义的结果。下面我们将看到一种没有这种要求的替代方法。

示例 7.1（模型选择）。统计学和机器学习领域的许多工作都致力于解决模型选择问题：给定一个数据集和一组离散的“模型”（每个模型都是一族概率分布），目标是确定最“拟合”

---

<!-- Footnote -->

${}^{3}$ 聚合函数的选择甚至可能取决于数据库，但必须以差分隐私的方式进行选择。此时，隐私成本就是选择操作与聚合函数组合的成本。

<!-- Footnote -->

---

数据的模型。例如，给定一组带标签的 $d$ 维数据，模型集合可能是最多包含 $s \ll  d$ 个特征的所有子集，目标是找到最能预测标签的特征集。函数 $f$ 可能是通过任意学习算法从给定的 $m$ 个模型集合中选择最佳模型，这一过程称为模型拟合。通过带噪声的最大值报告（Report Noisy Max）方法可以进行聚合以找出最受欢迎的值，该方法还能给出其受欢迎程度的估计。

示例 7.2（显著特征）。这是模型拟合的一个特殊情况。数据是 ${\mathbb{R}}^{d}$ 中的一组点，函数是非常流行的套索回归（LASSO），其输出是一个最多包含 $s \ll  d$ 个显著特征的列表 $L \in  {\left\lbrack  d\right\rbrack  }^{s}$。我们可以通过两种方式对输出进行聚合：逐个特征进行聚合——相当于对子采样与聚合（Subsample and Aggregate）方法进行 $d$ 次执行，每个特征执行一次，每次的取值范围大小为 2；或者对整个集合进行聚合，在这种情况下，取值范围的基数为 $\left( \begin{array}{l} d \\  s \end{array}\right)$。

### 7.2 提议 - 测试 - 发布

此时，人们可能会问：如果各数据块之间没有实质性的一致性，聚合的意义是什么？更一般地说，对于现实生活中任何规模合理的统计分析，我们期望结果相当稳定，不受任何单个个体存在与否的影响。实际上，这正是统计量显著性背后的直觉，也是差分隐私实用性的基础。我们甚至可以更进一步说，如果一个统计量不稳定，我们就不应该对计算它感兴趣。通常，我们的数据库实际上是来自更大总体的一个样本，我们的真正目标不是计算数据库本身的统计量值，而是估计潜在总体的统计量值。因此，在计算统计量时，我们实际上已经隐含地假设该统计量在子采样下是稳定的！

到目前为止，我们所看到的一切方法即使在非常“特殊”的数据集上也能提供隐私保护，而对于这些数据集，“通常”稳定的算法可能会非常不稳定。在本节中，我们介绍一种方法，即提议 - 测试 - 发布（Propose - Test - Release）方法，其背后的理念是：如果稳定性不足，那么可以放弃分析，因为结果实际上是没有意义的。也就是说，该方法允许分析人员检查在给定数据集上，函数是否满足某种“鲁棒性”或“稳定性”标准，如果不满足，则停止分析。

我们首次应用提议 - 测试 - 发布（Propose - Test - Release）方法的目标是提出拉普拉斯机制（Laplace mechanism）的一种变体，该变体添加的噪声规模严格小于函数的敏感度。这引出了局部敏感度的概念，它是针对（函数，数据库）对定义的，例如 (f, x)。简单来说，$f$ 相对于 $x$ 的局部敏感度是指，对于任何与 $x$ 相邻的 $y$，$f\left( y\right)$ 与 $f\left( x\right)$ 之间可能的最大差值。

定义 7.1（局部敏感度）。函数 $f : {\mathcal{X}}^{n} \rightarrow  {\mathbb{R}}^{k}$ 相对于数据库 $x$ 的局部敏感度为：

$$
\mathop{\max }\limits_{{y\text{ adjacent to }x}}\parallel f\left( x\right)  - f\left( y\right) {\parallel }_{1}.
$$

提议 - 测试 - 发布（Propose - Test - Release）方法是首先对局部敏感度提出一个界限，例如$b$ ，通常数据分析师对这个界限应该是多少会有一些想法，然后运行一个差分隐私测试，以确保数据库与任何该界限不成立的数据库“距离较远”。如果测试通过，则假设敏感度以$b$ 为界，并使用一种差分隐私机制，例如参数为$b/\epsilon$ 的拉普拉斯机制（Laplace mechanism），来发布对查询的（略有）噪声的响应。

请注意，我们可以将这种方法视为一种两方算法，其中一方扮演诚实的数据分析师，另一方是拉普拉斯机制。诚实的分析师和机制之间存在一种相互作用，在这种相互作用中，算法会要求对敏感度进行估计，然后“指示”机制在响应后续查询时使用这个估计的敏感度。为什么需要如此复杂呢？为什么机制不能简单地根据局部敏感度添加噪声，而不进行这种隐私估计游戏呢？原因是局部敏感度本身可能是敏感的。这一事实，再加上关于数据库的一些辅助信息，可能会导致隐私问题：对手可能知道数据库是$x$ 之一，对于所讨论的计算，它的局部敏感度非常低，以及与之相邻的$y$ ，对于该函数，它的局部敏感度非常高。在这种情况下，对手可能能够相当准确地猜测出$x$ 和$y$ 中哪个是真正的数据库。例如，如果$f\left( x\right)  = f\left( y\right)  = s$ 且响应与$s$ 相差甚远，那么对手会猜测是$y$ 。

这可以通过差分隐私的数学原理来解释。中位数函数存在相邻的实例，它们具有相同的中位数，例如$m$ ，但局部敏感度存在任意大的差距。假设通过拉普拉斯机制根据局部敏感度对噪声进行缩放来计算中位数查询的响应$R$ 。当数据库为$x$ 时，概率质量接近$m$ ，因为敏感度较小；但当数据库为$y$ 时，概率质量分布很分散，因为敏感度较大。作为一个极端情况，假设在$x$ 上的局部敏感度恰好为零，例如，$\mathcal{X} = \left\{  {0,{10}^{6}}\right\}  ,n$ 是偶数，且大小为$n + 1$ 的$x$ 包含$1 + n/2$ 个零。那么$x$ 的中位数为零，当数据库为$x$ 时，中位数的局部敏感度为 0。相比之下，相邻的数据库$y$ 大小为$n$ ，包含$n/2$ 个零，中位数为零（我们定义中位数在出现平局时取较小的值），当数据库为$y$ 时，中位数的局部敏感度为${10}^{6}$ 。在$x$ 上，拉普拉斯机制（参数为$0/\varepsilon  = 0$ ）的所有概率质量都集中在单点 0 上；但在$y$ 上，概率分布的标准差为$\sqrt{2} \cdot  {10}^{6}$ 。这破坏了差分隐私的所有希望。

为了检验该数据库与局部敏感度大于所提议界限 $b$ 的数据库“差异较大”，我们可以提出这样的查询：“真实数据库与局部敏感度超过 $b$ 的最近数据库之间的距离是多少？”到一组固定数据库的距离是一个（全局）敏感度为 1 的查询，因此可以通过向真实答案添加噪声 $\operatorname{Lap}\left( {1/\varepsilon }\right)$ 以差分隐私的方式运行此测试。为了保障隐私，该算法可以将这个含噪距离与一个保守阈值进行比较——由于极大幅度拉普拉斯噪声的异常事件，该阈值被超过的可能性极小。例如，如果使用的阈值为 ${\ln }^{2}n$，根据拉普拉斯分布的性质，误报（即当局部敏感度实际上超过 $b$ 时通过测试）的概率至多为 $O\left( {n}^{-\varepsilon \ln n}\right)$。由于误报的概率极小，该技术无法为任何 $\varepsilon$ 实现 $\left( {\varepsilon ,0}\right)$ -差分隐私。

要将此方法应用于区块共识，就像我们在讨论子采样与聚合时那样，将中间结果 $f\left( {B}_{1}\right) ,\ldots ,f\left( {B}_{m}\right)$ 视为一个数据集，并考虑这些值的某种集中度度量。直观地说，如果这些值高度集中，那么我们就实现了区块之间的共识。当然，我们仍然需要找到合适的集中度概念，一个有意义且具有差分隐私实例化的概念。在后面的章节中，我们将定义并整合两个似乎与子采样和聚合相关的稳定性概念：不敏感性（对移除或添加几个数据点的不敏感）和子采样下的稳定性，这体现了子样本应产生与完整数据集相似结果的概念。

#### 7.2.1 示例：数据集的规模

给定一个数据集，一个自然的问题是：“该数据集的规模或离散程度是多少？”这与数据位置问题不同，数据位置可以用中位数或均值来表示。数据规模通常用方差或分位数间距来表示。我们将重点关注四分位距（IQR），它是一种众所周知的用于数据规模的稳健估计量。我们先有一些大致的直观认识。假设数据是从具有累积分布函数 $F$ 的分布中独立同分布抽取的样本。那么定义为 ${F}^{-1}\left( {3/4}\right)  - {F}^{-1}\left( {1/4}\right)$ 的 $\operatorname{IQR}\left( F\right)$ 是一个常数，仅取决于 $F$。它可能非常大，也可能非常小，但无论如何，如果 $F$ 在两个四分位数处的密度足够高，那么给定足够多来自 $F$ 的样本，经验（即样本）四分位距应该接近 $\operatorname{IQR}\left( F\right)$。

我们用于四分位距的提议 - 测试 - 发布算法首先测试需要改变多少个数据库点才能得到一个四分位距“足够不同”的数据集。只有当（含噪）回复是“足够大”时，该算法才会发布数据集四分位距的近似值。

“足够不同”的定义是乘法意义上的，因为对于规模差异使用加法概念没有意义——加法量的合适规模是什么呢？因此，该算法使用规模的对数，这会导致四分位距上的乘法噪声。为了说明这一点，假设在典型情况下，修改单个点不会使样本四分位距改变 2 倍。那么样本四分位距的（以 2 为底）对数的局部敏感度被限制为 1。这使我们可以通过向该值添加从 $\operatorname{Lap}\left( {1/\varepsilon }\right)$ 中随机抽取的值来私密地发布样本四分位距对数的近似值。

设 $\operatorname{IQR}\left( x\right)$ 表示当数据集为 $x$ 时的样本四分位距。该算法（隐式地）提议将从 $\operatorname{Lap}\left( {1/\varepsilon }\right)$ 中抽取的噪声添加到值 ${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)$ 上。为了测试这种量级的噪声是否足以实现差分隐私，我们将 $\mathbb{R}$ 离散化为不相交的区间 $\{ \lbrack k\ln 2,\left( {k + 1}\right) \ln 2){\} }_{k \in  \mathbf{Z}}$，并询问必须修改多少个数据点才能得到一个新的数据库，使其四分位距的对数（以 2 为底）所在的区间与 ${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)$ 的不同。如果答案至少为 2，那么（四分位距的对数的）局部敏感度受区间宽度的限制。现在我们给出更多细节。

为了理解区间大小的选择，我们写下

$$
{\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)  = \frac{\ln \operatorname{IQR}\left( x\right) }{\ln 2} = \frac{c\ln 2}{\ln 2},
$$

由此我们发现，在 $\ln 2$ 的尺度上观察 $\ln \left( {\operatorname{IQR}\left( x\right) }\right)$ 等同于在 1 的尺度上观察 ${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)$。因此，我们有按比例缩放的区间，这些区间的端点是一对相邻的整数：${B}_{k} = \lbrack k,k + 1),k \in  \mathbf{Z}$，并且我们令 ${k}_{1} = \left\lfloor  {{\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right) }\right\rfloor$，所以 ${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)  \in  \left\lbrack  {{k}_{1},{k}_{1} + 1}\right)$，我们非正式地说四分位距的对数位于区间 ${k}_{1}$ 中。考虑以下测试查询：

## ${\mathbf{Q}}_{\mathbf{0}}$：为了得到一个新的数据库 $z$ 使得 ${\log }_{2}\left( {\operatorname{IQR}\left( z\right) }\right)  \notin  {B}_{{k}_{1}}$，需要改变多少个数据点？

设${A}_{0}\left( x\right)$为当数据库为$x$时${\mathbf{Q}}_{\mathbf{0}}$的真实答案。如果${A}_{0}\left( x\right)  \geq  2$，那么$x$的邻域$y$满足$\mid  {\log }_{2}\left( {\operatorname{IQR}\left( y\right) }\right)  -$ ${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)  \mid   \leq  1$。也就是说，它们彼此接近。这并不等同于在离散化中处于同一区间：${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)$可能靠近区间$\left\lbrack  {{k}_{1},{k}_{1} + 1}\right)$的一个端点，而${\log }_{2}\left( {\operatorname{IQR}\left( y\right) }\right)$可能恰好位于该端点的另一侧。设${R}_{0} = {A}_{0}\left( x\right)  + \operatorname{Lap}\left( {1/\varepsilon }\right)$，一个小的${R}_{0}$，即使从拉普拉斯分布中抽取的值的幅度很小，实际上也可能并不表明四分位距具有高敏感性。为了处理局部敏感性非常小，但${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)$非常接近边界的情况，我们考虑第二种离散化${\left\{  {B}_{k}^{\left( 2\right) } = \lbrack k - {0.5},k + {0.5})\right\}  }_{k \in  \mathbf{Z}}$。我们分别用${B}^{\left( 1\right) }$和${B}^{\left( 2\right) }$表示这两种离散化。值${\log }_{2}\left( {\operatorname{IQR}\left( x\right) }\right)  -$，实际上，任何值都不可能在两种离散化中都接近边界。如果${R}_{0}$在至少一种离散化中较大，则测试通过。

下面用于计算数据库规模的Scale算法（算法12）假设已知数据库的大小$n$，并且距离查询（“到四分位距敏感性超过$b$的数据库有多远？”）是在询问必须移动多少个点才能到达一个四分位距具有高敏感性的数据库。我们可以通过让算法首先询问（敏感性为1）查询：“$x$中有多少个数据点？”来避免这个假设。我们注意到，出于技术原因，为了处理${IQR}\left( x\right)  = 0$的情况，我们定义$\log 0 =  - \infty$、$\lfloor  - \infty \rfloor  =  - \infty$，并令$\lbrack  - \infty , - \infty ) = \{  - \infty \}$。

<!-- Media -->

算法12 Scale算法（发布四分位距）

---

要求：数据集：$x \in  {\mathcal{X}}^{ * }$，隐私参数：$\epsilon ,\delta  > 0$

对于第$j$种离散化$\left( {j = 1,2}\right)$执行

计算${R}_{0}\left( x\right)  = {A}_{0}\left( x\right)  + {z}_{0}$，其中${z}_{0}{ \in  }_{R}\operatorname{Lap}\left( {1/\varepsilon }\right)$。

如果${R}_{0} \leq  1 + \ln \left( {1/\delta }\right)$则

令${s}^{\left( j\right) } =  \bot$。

否则

设${s}^{\left( j\right) } = {IQR}\left( x\right)  \times  {2}^{{z}_{s}^{\left( j\right) }}$，其中${z}_{s}^{\left( j\right) } \sim  \operatorname{Lap}\left( {1/\varepsilon }\right)$。

结束条件判断

结束循环

若 ${s}^{\left( 1\right) } \neq   \bot$ 成立，则

返回 ${s}^{\left( 1\right) }$。

否则

返回 ${s}^{\left( 2\right) }$。

结束条件判断

---

<!-- Media -->

注意，该算法效率很高：设 ${x}_{\left( 1\right) },{x}_{\left( 2\right) },\ldots ,{x}_{\left( n\right) }$ 表示排序后的 $n$ 个数据库点，设 $x\left( m\right)$ 表示中位数，因此 $m = \lfloor \left( {n + 1}\right) /2\rfloor$。那么中位数的局部敏感度为 $\max \{ x\left( m\right)  -$ $x\left( {m - 1}\right) ,x\left( {m + 1}\right)  - x\left( m\right) \}$，更重要的是，可以通过考虑 $O\left( n\right)$ 个宽度为 ${2}^{{k}_{1}}$ 和 ${2}^{{k}_{1} + 1}$ 的滑动区间来计算 ${A}_{0}\left( x\right)$，每个区间的一个端点都在 $x$ 中。每个区间的计算成本是恒定的。

我们不会证明该算法的收敛边界，因为为了简单起见，我们使用的对数底数远非最优（更好的底数是 $1 + 1/\ln n$）。我们简要概述一下隐私性证明的步骤。

定理 7.1。缩放算法（算法 12）具有 $\left( {{4\varepsilon },\delta }\right)$ -差分隐私性。

证明。（概要）设 $s$ 是单次离散化得到的结果的简写，并定义 ${\mathcal{D}}_{0} = \left\{  {x : {A}_{0}\left( x\right)  \geq  2}\right\}$，证明过程表明：

1. 查询 ${\mathbf{Q}}_{\mathbf{0}}$ 的最坏情况敏感度至多为 1。

2. 相邻数据库产生 $\bot$ 的可能性几乎相等：对于所有相邻数据库 $x,y$：

$$
\Pr \left\lbrack  {s =  \bot   \mid  x}\right\rbrack   \leq  {e}^{\varepsilon }\Pr \left\lbrack  {s =  \bot   \mid  y}\right\rbrack  .
$$

3. 不在 ${\mathcal{D}}_{0}$ 中的数据库不太可能通过测试：

$$
\forall x \notin  {\mathcal{D}}_{0} : \Pr \left\lbrack  {s \neq   \bot   \mid  x}\right\rbrack   \leq  \frac{\delta }{2}.
$$

4. $\forall C \in  {\mathbb{R}}^{ + },x \in  {\mathcal{D}}_{0}$ 以及 $x$ 的所有邻居 $y$：

$$
\Pr \left\lbrack  {s \in  C \mid  x}\right\rbrack   \leq  {e}^{2\varepsilon }\Pr \left\lbrack  {s \in  C \mid  y}\right\rbrack  .
$$

因此，对于每次离散化，我们都能得到 $\left( {{2\varepsilon },\delta /2}\right)$ -差分隐私性。应用定理 3.16（附录 B），该定理指出“隐私预算 ε 和松弛参数 δ 是可加的”，可得到 $\left( {{4\varepsilon },\delta }\right)$ -差分隐私性。

### 7.3 稳定性与隐私性

#### 7.3.1 两种稳定性概念

我们首先区分本节中相互交织的两种稳定性概念：子采样稳定性（subsampling stability），即在数据的随机子样本下能得到相似的结果；以及针对给定数据集的扰动稳定性（perturbation stability），或低局部敏感性。在本节中，我们将定义并使用这两种稳定性的极端版本。

- 子采样稳定性：如果当 $\widehat{x}$ 是从 $x$ 中独立地以概率 $q$ 包含每个条目的随机子样本时，$f\left( \widehat{x}\right)  = f\left( x\right)$ 至少以概率 $3/4$ 成立，我们就称 $f$ 在 $x$ 上是 $q$ -子采样稳定的。我们将在算法 ${\mathcal{A}}_{\text{samp }}$（样本与聚合的一种变体）中使用这个概念。

- 扰动稳定性：如果 $f$ 在 $x$ 的所有邻域上都取 $f\left( x\right)$ 值（否则为不稳定），我们就称 $f$ 在 $x$ 上是稳定的。换句话说，如果 $f$ 在 $x$ 上的局部敏感性为零，那么 $f$ 在 $x$ 上是稳定的。我们将使用这个概念（在下面的算法 ${\mathcal{A}}_{\text{dist }}$ 中实现）用于 ${\mathcal{A}}_{\text{samp }}$ 的聚合步骤。

算法 ${\mathcal{A}}_{\text{samp }}$ 的核心是扰动稳定性的一种宽松版本，这里我们不要求在相邻数据库上值保持不变（这个概念对于任意范围，包括任意离散范围都有意义），而只要求在相邻数据库上的值“接近”（这个概念要求在值域上有一个度量）。

具有任意值域的函数 $f$，特别是子采样与聚合中的输出聚合问题，催生了下一个算法 ${\mathcal{A}}_{\text{dist }}$。如果 $x$ 与最近的不稳定数据集的距离至少为 $\frac{2\log \left( {1/\delta }\right) }{\varepsilon }$，那么算法在输入 $f,x,{\mathcal{A}}_{\text{dist }}$ 时以高概率输出 $f\left( x\right)$。该算法在概念上很简单：计算到最近的不稳定数据集的距离，添加拉普拉斯噪声 $\operatorname{Lap}\left( {1/\varepsilon }\right)$，并检查这个带噪声的距离是否至少为 $\frac{2\log \left( {1/\delta }\right) }{\varepsilon }$。如果是，则输出 $f\left( x\right)$，否则输出 $\bot$。现在我们将其形式化一些。

我们首先定义扰动稳定性的一种定量度量。

定义 7.2。如果从 $x$ 中添加或移除任意 $k$ 个元素都不会改变 $f$ 的值，即对于所有满足 $\left| {x\Delta y}\right|  \leq  k$ 的 $y$ 都有 $f\left( x\right)  = f\left( y\right)$，则称函数 $f : {\mathcal{X}}^{ * } \rightarrow  \mathcal{R}$ 在输入 $x$ 上是 $k$ -稳定的。如果 $f$ 在 $x$ 上（至少）是 1 -稳定的，我们就称它在 $x$ 上是稳定的，否则为不稳定。

定义 7.3。数据集 $x \in  {\mathcal{X}}^{ * }$ 相对于函数 $f$ 的不稳定距离是指为了达到一个在 $f$ 下不稳定的数据集，必须从 $y$ 中添加或移除的元素数量。

注意，当且仅当 $x$ 到不稳定状态的距离至少为 $k$ 时，$f$ 在 $x$ 上是 $k$ -稳定的。

算法 ${\mathcal{A}}_{\text{dist }}$ 是针对离散值函数 $g$ 的提议 - 测试 - 发布（Propose - Test - Release）的一个实例，如图 13 所示。

<!-- Media -->

算法 13 ${\mathcal{A}}_{\text{dist }}$（基于到不稳定状态的距离发布 $g\left( x\right)$）

---

要求：数据集：$x \in  {\mathcal{X}}^{ * }$ ，隐私参数：$\epsilon ,\delta  > 0$ ，函数 $g$ ：

		${\mathcal{X}}^{ * } \rightarrow  \mathbb{R}$

$d \leftarrow$ 从 $x$ 到最近的不稳定实例的距离

		$\widehat{d} \leftarrow  d + \operatorname{Lap}\left( {1/\varepsilon }\right)$

如果 $\widehat{d} > \frac{\log \left( {1/\delta }\right) }{\varepsilon }$ ，则

输出 $g\left( x\right)$

否则

输出 1

结束条件判断

---

<!-- Media -->

根据拉普拉斯分布（Laplace distribution）的性质，以下命题的证明是显而易见的。

命题 7.2。对于每个函数 $g$ ：

1. ${\mathcal{A}}_{\text{dist }}$ 是 $\left( {\varepsilon ,\delta }\right)$ -差分隐私（differentially private）的。

2. 对于所有 $\beta  > 0$ ：如果 $g$ 在 $x$ 上是 $\frac{\ln \left( {1/\delta }\right)  + \ln \left( {1/\beta }\right) }{\varepsilon }$ -稳定的，那么 ${\mathcal{A}}_{\text{dist }}\left( x\right)  =$ $g\left( x\right)$ 的概率至少为 $1 - \beta$ ，其中概率空间是 ${\mathcal{A}}_{\text{dist }}$ 的随机翻转。

从以下意义上来说，这个基于距离的结果是最优的：如果有两个数据集 $x$ 和 $y$ ，${\mathcal{A}}_{\text{dist }}$ 分别以至少恒定的概率输出不同的值 $g\left( x\right)$ 和 $g\left( y\right)$ ，那么从 $x$ 到 $y$ 的距离必须为 $\Omega \left( {\log \left( {1/\delta }\right) /\varepsilon }\right)$ 。

到不稳定性的距离可能难以计算，甚至难以确定其下界，因此这通常不是一个实际可行的解决方案。有两个例子表明到不稳定性的距离很容易界定，即中位数（median）和众数（mode，出现频率最高的值）。

如果函数（例如 $f$ ）在感兴趣的特定数据集上不稳定，${\mathcal{A}}_{\text{dist }}$ 可能也不令人满意。例如，假设由于 $x$ 中存在一些离群值，$f$ 不稳定。平均值函数就会出现这种情况，不过对于这个函数，有一些众所周知的鲁棒替代方法，如温莎化均值（winsorized mean）、截尾均值（trimmed mean）和中位数。那么对于一般的函数 $f$ 呢？是否有一种方法可以“强制”任意的 $f$ 在数据库 $x$ 上保持稳定？

这将是 ${\mathcal{A}}_{\text{samp }}$ 的目标，${\mathcal{A}}_{\text{samp }}$ 是子采样与聚合（Subsample and Aggregate）的一种变体，只要 $f$ 在 $x$ 上是子采样稳定的，它就会以高概率（基于其自身的随机选择）输出 $f\left( x\right)$ 。

#### 7.3.2 算法 ${\mathcal{A}}_{\text{samp }}$

在${\mathcal{A}}_{\text{samp }}$中，块${B}_{1},\ldots ,{B}_{m}$是有放回选取的，因此每个块与输入具有相同的分布（尽管现在$x$的一个元素可能出现在多个块中）。我们将这些子采样数据集称为${\widehat{x}}_{1},\ldots ,{\widehat{x}}_{m}$。然后，中间输出$z = \left\{  {f\left( {\widehat{x}}_{1}\right) ,\ldots ,f\left( {\widehat{x}}_{m}\right) }\right\}$通过${\mathcal{A}}_{\text{dist }}$以函数$g =$众数（mode）的方式进行聚合。用于估计$z$上众数稳定性的距离度量是众数的流行度与第二频繁值的流行度之差的缩放版本。算法${\mathcal{A}}_{\text{samp }}$如图14所示，其运行时间主要由运行约$1/{q}^{2}$次的$f$决定；因此，只要$f$高效，该算法就是高效的。

算法${\mathcal{A}}_{\text{samp }}$的关键特性是，对于输入$f,x$，只要$f$在$x$上对于$q = \frac{\varepsilon }{{64}\log \left( {1/\delta }\right) }$是$q$ - 子采样稳定的，那么在其自身的随机选择下，它以高概率输出$f\left( x\right)$。这一结果具有重要的统计学解释。回顾示例7.1中关于模型选择的讨论。给定一组模型，模型选择的样本复杂度是从其中一个模型的分布中抽取的、以至少$2/3$的概率选择正确模型所需的样本数量。该结果表明，差分隐私模型选择将（非隐私）模型选择的样本复杂度提高了一个与问题无关（且与范围无关）的因子$O\left( {\log \left( {1/\delta }\right) /\varepsilon }\right)$。

<!-- Media -->

算法14 ${\mathcal{A}}_{\text{samp }}$：子采样稳定$f$的自助法

---

要求：数据集：$x$，函数$f : {\mathcal{X}}^{ * } \rightarrow  \mathbb{R}$，隐私参数$\epsilon ,\delta  >$

			0.

			$q \leftarrow  \frac{\epsilon }{{64}\ln \left( {1/\delta }\right) },m \leftarrow  \frac{\log \left( {n/\delta }\right) }{{q}^{2}}.$

2: 从$x$中对$m$个数据集${\widehat{x}}_{1},\ldots ,{\widehat{x}}_{m}$进行子采样，其中${\widehat{x}}_{i}$以概率${\widehat{x}}_{1},\ldots ,{\widehat{x}}_{m}$独立包含$m$的每个

位置。

3: 如果$x$的某个元素出现在超过${2mq}$个集合${\widehat{x}}_{i}$中，则

停止并输出$\bot$。

否则

				$z \leftarrow  \left\{  {f\left( {\widehat{x}}_{1}\right) ,\cdots ,f\left( {\widehat{x}}_{m}\right) }\right\}  .$

对于每个$r \in  \mathbb{R}$，令$\operatorname{count}\left( r\right)  = \# \left\{  {i : f\left( {\widehat{x}}_{i}\right)  = r}\right\}$。

令${\text{count}}_{\left( i\right) }$表示第$i$大的计数，$i = 1,2$。

				$d \leftarrow  \left( {{\text{count}}_{\left( 1\right) } - {\text{count}}_{\left( 2\right) }}\right) /\left( {4mq}\right)  - 1$

注释：现在使用$d$运行${\mathcal{A}}_{\text{dist }}\left( {g,z}\right)$以估计到

不稳定性的距离：

				$\widehat{d} \leftarrow  d + \operatorname{Lap}\left( \frac{1}{\epsilon }\right)$ .

if $\widehat{d} > \ln \left( {1/\delta }\right) /\varepsilon$ then

输出$g\left( z\right)  =$众数(z)。

else

输出 $\bot$。

结束条件判断

结束条件判断

---

<!-- Media -->

## 定理 7.3。

1. 算法 ${\mathcal{A}}_{\text{samp }}$ 具有 $\left( {\varepsilon ,\delta }\right)$ -差分隐私性。

2. 如果 $f$ 在输入 $x$ 上是 $q$ -子采样稳定的，其中 $q = \frac{\varepsilon }{{64}\ln \left( {1/\delta }\right) }$，那么算法 ${\mathcal{A}}_{\text{samp }}\left( x\right)$ 输出 $f\left( x\right)$ 的概率至少为 $1 - {3\delta }$。

3. 如果 $f$ 可以在长度为 $n$ 的输入上在时间 $T\left( n\right)$ 内计算得出，那么 ${\mathcal{A}}_{\text{samp }}$ 的期望运行时间为 $O\left( \frac{\log n}{{q}^{2}}\right) \left( {T\left( {qn}\right)  + n}\right)$。

请注意，这里的效用声明是逐输入的保证；$f$ 不必在所有输入上都是 $q$ -子采样稳定的。重要的是，它不依赖于范围 $\mathcal{R}$ 的大小。在模型选择的背景下，这意味着只要有一个特定的模型以合理的概率被选中，就可以在样本复杂度适度增加（约 $\log \left( {1/\delta }\right) /\varepsilon$）的情况下有效地满足差分隐私。

隐私性的证明源于 $d$ 计算的不敏感性、提议 - 测试 - 发布技术的隐私性以及子采样和聚合的隐私性，略有修改以考虑到该算法进行有放回采样，因此聚合器具有更高的敏感性，因为任何个体可能会影响多达 ${2mq}$ 个块。分析这种方法效用的主要观察结果是，众数的稳定性是众数频率与次最流行元素频率之差的函数。下一个引理表明，如果 $f$ 在 $x$ 上是子采样稳定的，那么 $x$ 相对于众数 $g\left( z\right)  = g\left( {f\left( {\widehat{x}}_{1}\right) ,\ldots ,f\left( {\widehat{x}}_{m}\right) }\right)$ 远非不稳定（但不一定相对于 $f$），而且可以高效且私密地估计 $x$ 到不稳定状态的距离。

引理 7.4。固定 $q \in  \left( {0,1}\right)$。给定 $f : {\mathcal{X}}^{ * } \rightarrow  \mathcal{R}$，设 $\widehat{f} : {\mathcal{X}}^{ * } \rightarrow  \mathcal{R}$ 为函数 $\widehat{f} = \operatorname{mode}\left( {f\left( {\widehat{x}}_{1}\right) ,\ldots ,f\left( {\widehat{x}}_{m}\right) }\right)$，其中每个 ${\widehat{x}}_{i}$ 以概率 $q$ 独立包含 $x$ 中的每个元素，且 $m = \ln \left( {n/\delta }\right) /{q}^{2}$。设 $d\left( z\right)  = \left( {{\operatorname{count}}_{\left( 1\right) } - {\operatorname{count}}_{\left( 2\right) }}\right) /\left( {4mq}\right)  - 1$；也就是说，给定一个值的“数据库” $z$，$d\left( z\right)  + 1$ 是两个最流行值出现次数的缩放差值。固定一个数据集 $x$。设 $E$ 为 $x$ 中没有位置被包含在超过 ${2mq}$ 个子集 ${\widehat{x}}_{i}$ 中的事件。那么，当 $q \leq  \varepsilon /{64}\ln \left( {1/\delta }\right)$ 时，我们有：

1. $E$ 发生的概率至少为 $1 - \delta$。

2. 在$E,d$的条件下，对$\widehat{f}$在$x$上的稳定性进行了下界约束，并且$d$的全局敏感度为1。

3. 如果$f$在$x$上具有$q$ - 子采样稳定性，那么在子样本的选择上，至少以$1 - \delta$的概率，我们有$\widehat{f}\left( x\right)  = f\left( x\right)$，并且在这个事件发生的条件下，最终测试至少以$1 - \delta$的概率通过，其中该概率是基于从$\operatorname{Lap}\left( {1/\varepsilon }\right)$中抽样得到的。

第2部分和第3部分中的事件同时发生的概率至少为$1 - {2\delta }$。

证明。第1部分可由切尔诺夫界（Chernoff bound）得出。为了证明第2部分，注意到，在事件$E$发生的条件下，在原始数据集中添加或移除一个条目，任何计数${\text{count}}_{\left( r\right) }$的变化最多为${2mq}$。因此，${\text{count}}_{\left( 1\right) } - {\text{count}}_{\left( 2\right) }$的变化最多为${4mq}$。这反过来意味着，对于任何$x$，$d\left( {f\left( {\widehat{x}}_{1}\right) ,\ldots ,f\left( {\widehat{x}}_{m}\right) }\right)$的变化最多为1，因此其全局敏感度为1。这也意味着$d$对$\widehat{f}$在$x$上的稳定性进行了下界约束。

我们现在转向第3部分。我们要论证两个事实：

1. 如果$f$在$x$上具有$q$ - 子采样稳定性，那么两个最受欢迎的区间的计数之间很可能存在较大差距。具体来说，我们要证明，以高概率有${\operatorname{count}}_{\left( 1\right) } - {\operatorname{count}}_{\left( 2\right) } \geq$ $m/4$。注意，如果最受欢迎的区间的计数至少为${5m}/8$，那么第二受欢迎的区间的计数最多为${3m}/8$，差值为$m/4$。根据子采样稳定性的定义，最受欢迎的区间的期望计数至少为${3m}/4$，因此，由切尔诺夫界，取$\alpha  = 1/8$时，其计数小于${5m}/8$的概率至多为${e}^{-{2m}{\alpha }^{2}} = {e}^{-m/{32}}$。（所有概率均基于子采样。）

2. 当两个最受欢迎的区间的计数之间的差距较大时，算法不太可能失败；也就是说，测试很可能成功。需要担心的是，从$\operatorname{Lap}\left( \frac{1}{\varepsilon }\right)$中抽样得到的值为负且绝对值较大，以至于即使$d$很大，$\widehat{d}$也会低于阈值$\left( {\ln \left( {1/\delta }\right) /\varepsilon }\right)$。为了确保这种情况发生的概率至多为$\delta$，只需满足$d > 2\ln \left( {1/\delta }\right) /\varepsilon$即可。

根据定义，$d = \left( {{\text{count}}_{\left( 1\right) } - {\text{count}}_{\left( 2\right) }}\right) /\left( {4mq}\right)  - 1$，并且假设我们处于刚刚描述的高概率情况，这意味着

$$
d \geq  \frac{m/4}{4mq} - 1 = \frac{1}{16q} - 1
$$

因此，只需满足

$$
\frac{1}{16q} > 2\ln \left( {1/\delta }\right) /\varepsilon 
$$

取$q \leq  \varepsilon /{64}\ln \left( {1/\delta }\right)$就足够了。

最后，注意到对于$q$和$m$的这些值，我们有${e}^{-m/{32}} < \delta$。

示例7.3. [原始数据问题] 假设我们有一位分析师，我们可以信任他会遵循指令，并且只发布根据这些指令获得的信息。更理想的情况是，假设我们有 $b$ 位这样的分析师，并且我们可以信任他们彼此之间不会交流。这些分析师不需要完全相同，但他们需要考虑一组共同的选项。例如，这些选项可能是一组固定的可能统计量 $S$ 中的不同统计量，在第一步中，分析师的目标是从 $S$ 中选择最显著的统计量以便最终发布。之后，所选的统计量将以差分隐私的方式重新计算，并且结果可以发布。

如前所述，该过程完全不具备隐私性：第一步中对统计量的选择可能依赖于单个个体的数据！尽管如此，我们可以使用子采样与聚合框架来执行第一步，让第 $i$ 位分析师接收数据点的一个子样本，并将函数 ${f}_{i}$ 应用于这个较小的数据库以获得一个选项。然后，这些选项将按照算法 ${\mathcal{A}}_{\text{samp }}$ 进行聚合；如果有明显的胜出者，那么这个胜出者极有可能就是所选的统计量。这个统计量是以差分隐私的方式选择的，并且在第二步中，它将以差分隐私的方式进行计算。

## 参考文献注释

子采样与聚合方法由尼西姆（Nissim）、拉斯霍德尼科娃（Raskhodnikova）和史密斯（Smith）[68]发明，他们是首次定义并利用低局部敏感度的人。提议 - 测试 - 发布方法归功于德沃尔（Dwork）和雷（Lei）[22]，发布四分位距的算法也是如此。关于稳定性和隐私性的讨论，以及融合这两种技术的算法 ${\mathcal{A}}_{\text{samp }}$，归功于史密斯和塔库尔塔（Thakurta）[80]。这篇论文通过分析著名的套索（LASSO）算法的子采样稳定性条件，并表明在套索算法已知具有良好解释能力的（固定数据以及分布）条件下，通过（${\mathcal{A}}_{\text{samp }}$ 的一种推广）可以“免费”获得差分隐私，从而展示了 ${\mathcal{A}}_{\text{samp }}$ 的强大之处。

## 8 下界与分离结果

在本节中，我们将研究各种下界和权衡问题：

1. 为了不完全破坏任何合理的隐私概念，响应必须达到多大的不准确性？

2. 前一个问题的答案如何依赖于查询的数量？

3. 我们能否在每种差分隐私所允许的准确性方面，将 $\left( {\varepsilon ,0}\right)$ -差分隐私与 $\left( {\varepsilon ,\delta }\right)$ -差分隐私区分开来？

4. 在保持 $\left( {\varepsilon ,0}\right)$ -差分隐私的同时，线性查询和任意低敏感度查询所能达到的效果之间是否存在本质差异？

另一种不同类型的分离结果区分了生成处理给定类中所有查询的数据结构的计算复杂度，与生成实现相同目标的合成数据库的计算复杂度。我们将对这一结果的讨论推迟到第9节。

### 8.1 重构攻击

我们在第1节中论证了，任何非平凡的机制都必须是随机化的。由此可知，至少对于某些数据库、查询和随机比特的选择，该机制产生的响应并非完全准确。为了保护隐私，答案必须达到多大的不准确性这一问题在所有计算模型中都有意义：交互式、非交互式以及第12节中讨论的模型。

对于失真的下界，为简单起见，我们假设数据库中每人只有一个——但非常敏感的——比特，因此我们可以将数据库视为一个 $n$ 比特的布尔向量 $d = \left( {{d}_{1},\ldots ,{d}_{n}}\right)$。这是对一种情况的抽象，在这种情况下，数据库行相当复杂，例如，它们可能是医疗记录，但攻击者只对一个特定字段感兴趣，比如是否存在镰状细胞特征。抽象后的攻击包括发出一系列查询，每个查询由数据库行的一个子集 $S$ 描述。该查询询问所选行中有多少个 1。将查询表示为集合 $S$ 的 $n$ 比特特征向量 $\mathbf{S}$，在对应于 $S$ 中行的所有位置上为 1，其他位置为 0，那么该查询的真实答案就是内积 $A\left( S\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}{d}_{i}{\mathbf{S}}_{i}.$

固定一个任意的隐私机制。我们用 $r\left( S\right)$ 表示对查询 $S$ 的响应。这可以通过显式方式获得，例如，如果该机制是交互式的且发出了查询 $S$，或者如果该机制预先获得了所有查询并生成了一个答案列表；也可以通过隐式方式获得，即如果该机制生成了一个概要，分析人员从中提取出 $r\left( S\right)$。请注意，$r\left( S\right)$ 可能取决于该机制所做的随机选择以及查询历史。用 $E\left( {S,r\left( S\right) }\right)$ 表示响应 $r\left( S\right)$ 的误差，也称为噪声或失真，因此 $E\left( {S,r\left( S\right) }\right)  = \left| {A\left( S\right)  - r\left( S\right) }\right|$。

我们想问的问题是：“为了保护隐私，需要多少噪声？”差分隐私是一种特定的隐私保证，但人们也可以考虑较弱的概念，因此在下界论证中，较为适度的目标不是保证隐私，而仅仅是防止出现隐私灾难。

定义 8.1。如果一个对手可以构建一个候选数据库 $c$，该数据库除了 $o\left( n\right)$ 个条目外，与真实数据库 $d$ 完全一致，即 $\parallel c - d{\parallel }_{0} \in  o\left( n\right)$，则称该机制是明显非隐私的。

换句话说，如果一个机制允许进行重构攻击，使得对手能够正确猜出数据库中除 $o\left( n\right)$ 个成员之外所有成员的秘密位，则该机制是明显非隐私的。（不要求对手知道哪些答案是正确的。）

定理 8.1。设 $\mathcal{M}$ 是一个失真幅度受 $E$ 限制的机制。那么存在一个对手，他可以将数据库重构到误差不超过 ${4E}$ 个位置。

该定理的一个简单推论是，一个总是添加幅度受例如 $n/{401}$ 限制的噪声的隐私机制，会使对手能够正确重构 ${99}\%$ 个条目。

证明。设 $d$ 为真实数据库。对手分两个阶段进行攻击：

1. 估计所有可能集合中 1 的数量：对所有子集 $S \subseteq  \left\lbrack  n\right\rbrack$ 查询 $\mathcal{M}$。

2. 排除“遥远”的数据库：对于每个候选数据库 $c \in  \{ 0,1{\} }^{n}$，如果存在 $\exists S \subseteq  \left\lbrack  n\right\rbrack$ 使得 $\left| {\mathop{\sum }\limits_{{i \in  S}}{c}_{i} - \mathcal{M}\left( S\right) }\right|  > E$，则排除 $c$。如果 $c$ 未被排除，则输出 $c$ 并停止。

由于 $\mathcal{M}\left( S\right)$ 的误差从不超过 $E$，真实数据库不会被排除，因此这个简单（但效率不高！）的算法将输出某个候选数据库 $c$。我们将证明 $c$ 和 $d$ 不同的位置数量最多为 $4 \cdot  E$。

设 ${I}_{0}$ 为满足 ${d}_{i} = 0$ 的索引，即 ${I}_{0} = \left\{  {i \mid  {d}_{i} = 0}\right\}$ 。类似地，定义 ${I}_{1} = \left\{  {i \mid  {d}_{i} = 1}\right\}$ 。由于 $c$ 未被排除，$\mid  \mathcal{M}\left( {I}_{0}\right)  -$ $\mathop{\sum }\limits_{{i \in  {I}_{0}}}{c}_{i}\left| { \leq  E\text{. However,by assumption}}\right| \mathcal{M}\left( {I}_{0}\right)  - \mathop{\sum }\limits_{{i \in  {I}_{0}}}{d}_{i} \mid   \leq  E$ 。根据三角不等式可知，$c$ 和 $d$ 在 ${I}_{0}$ 中至多有 ${2E}$ 个位置不同；同理，它们在 ${I}_{1}$ 中也至多有 ${2E}$ 个位置不同。因此，$c$ 和 $d$ 至多在 ${4E}$ 个位置上不一致。

如果我们考虑对查询次数设置更现实的界限会怎样呢？我们认为 $\sqrt{n}$ 是噪声的一个有趣阈值，原因如下：如果数据库包含从规模为 $N \gg  n$ 的总体中均匀随机抽取的 $n$ 个人，且满足给定条件的总体比例为 $p$ ，那么根据二项分布的性质，我们预计数据库中满足该属性的行数大约为 ${np} \pm  \Theta \left( \sqrt{n}\right)$ 。也就是说，抽样误差约为 $\sqrt{n}$ 。我们希望为保护隐私引入的噪声小于抽样误差，理想情况下为 $o\left( \sqrt{n}\right)$ 。接下来的结果研究了当查询次数与 $n$ 呈线性关系时，实现如此小误差的可行性。结果是否定的。

忽略计算复杂度，为了理解为什么可能存在查询高效的攻击，我们稍微修改一下问题，考虑数据库 $d \in  \{  - 1,1{\} }^{n}$ 和查询向量 $v \in  \{  - 1,1{\} }^{n}$ 。真实答案再次定义为 $d \cdot  v$ ，而响应是真实答案的含噪版本。现在，考虑一个与 $d$ 差异较大的候选数据库 $c$ ，例如 $\parallel c - d{\parallel }_{0} \in  \Omega \left( n\right)$ 。对于一个随机的 $v{ \in  }_{R}\{  - 1,1{\} }^{n}$ ，以恒定概率有 $\left( {c - d}\right)  \cdot  v \in  \Omega \left( \sqrt{n}\right)$ 。为了说明这一点，固定 $x \in  \{  - 1,1{\} }^{n}$ 并选择 $v{ \in  }_{R}\{  - 1,1{\} }^{n}$ 。那么 $x \cdot  v$ 是独立随机变量 ${x}_{i}{v}_{i}{ \in  }_{R}\{  - 1,1\}$ 的和，其期望为 0，方差为 $n$ ，并且服从缩放和平移后的二项分布。出于同样的原因，如果 $c$ 和 $d$ 至少在 ${\alpha n}$ 行上不同，并且随机选择 $v$ ，那么 $\left( {c - d}\right)  \cdot  v$ 服从均值为 0 且方差至少为 ${\alpha n}$ 的二项分布。因此，根据二项分布的性质，我们预计 $c \cdot  v$ 和 $d \cdot  v$ 以恒定概率至少相差 $\alpha \sqrt{n}$ 。注意，我们使用的是分布的反集中性质，而不是通常的集中性质。

当噪声被限制为$o\left( \sqrt{n}\right)$时，这为排除$c$提供了一种攻击方法：计算$c \cdot  v$与含噪响应$r\left( v\right)$之间的差值。如果该差值的幅度超过$\sqrt{n}$（在$v -$的选择上，这种情况会以恒定概率发生），则排除$c$。下一个定理将这一论点形式化，并进一步表明，即使面对很大一部分完全任意的响应，这种攻击仍然有效：如果管理员被限制在绝对误差$o\left( \sqrt{n}\right)$内回答至少$\frac{1}{2} + \eta$个问题，攻击者使用线性数量的$\pm  1$个问题，几乎可以重构整个数据库。

定理8.2。对于任意的$\eta  > 0$和任意函数$\alpha  = \alpha \left( n\right)$，存在常数$b$，并且有一种使用${bn} \pm  1$个问题的攻击方法，如果管理员在绝对误差$\alpha$内回答至少$\frac{1}{2} + \eta$个问题，该方法可以重构一个与真实数据库最多只有${\left( \frac{2\alpha }{\eta }\right) }^{2}$个条目不同的数据库。

证明。我们从一个简单的引理开始。

引理8.3。设$Y = \mathop{\sum }\limits_{{i = 1}}^{k}{X}_{i}$，其中每个${X}_{i}$是一个均值为零的$\pm  2$独立伯努利随机变量。那么对于任意的$y$和任意的$\ell  \in  \mathbb{N},\Pr \left\lbrack  {Y \in  \left\lbrack  {{2y},2\left( {y + \ell }\right) }\right\rbrack  }\right\rbrack   \leq  \frac{\ell  + 1}{\sqrt{k}}.$

证明。注意到$Y$总是偶数，并且$\Pr \left\lbrack  {Y = {2y}}\right\rbrack   = \left( \begin{matrix} k \\  \left( {k + y}\right) /2 \end{matrix}\right) {\left( \frac{1}{2}\right) }^{k}$。这个表达式至多为$\left( \begin{matrix} k \\  \lceil k/2\rceil  \end{matrix}\right) {\left( \frac{1}{2}\right) }^{k}$。使用斯特林近似（Stirling’s approximation），即$n$!可以近似为$\sqrt{2n\pi }{\left( n/e\right) }^{n}$，它的上界为$\sqrt{\frac{2}{\pi k}}$。通过对$\left\lbrack  {{2y},2\left( {y + \ell }\right) }\right\rbrack$中$Y$的$\ell  + 1$个可能值进行联合界（union bound），即可得到该结论。

攻击者的攻击方法是选择${bn}$个随机向量$v \in  \{  - 1,1{\} }^{n}$，获取响应$\left( {{y}_{1},\ldots ,{y}_{bn}}\right)$，然后输出任何满足以下条件的数据库$c$：对于至少$\frac{1}{2} + \eta$个索引$i$，有$\left| {{y}_{i} - {\left( Ac\right) }_{i}}\right|  \leq  \alpha$，其中$A$是以随机查询向量$v$为行的${bn} \times  n$矩阵。

设真实数据库为$d$，并设$c$为重构后的数据库。根据该机制行为的假设，对于$i \in  \left\lbrack  {bn}\right\rbrack$的$1/2 + \eta$比例，有$\left| {{\left( Ad\right) }_{i} - {y}_{i}}\right|  \leq  \alpha$。由于$c$未被排除，我们还可知对于$i \in  \left\lbrack  {bn}\right\rbrack$的$1/2 + \eta$比例，有$\left| {{\left( Ac\right) }_{i} - {y}_{i}}\right|  \leq  \alpha$。由于任意两组这样的索引在$i \in  \left\lbrack  {bn}\right\rbrack$的至少${2\eta }$比例上是一致的，根据三角不等式，对于$i,\left| {\left\lbrack  \left( c - d\right) A\right\rbrack  }_{i}\right|  \leq  {2\alpha }$的至少${2\eta bn}$个值成立。

我们希望证明，除了${\left( \frac{2\alpha }{\eta }\right) }^{2}$个条目外，$c$与$d$是一致的。我们将证明，如果重构后的$c$与$d$相差甚远，至少在${\left( 2\alpha /\eta \right) }^{2}$个条目上不一致，那么随机选择的$A$对于$i$的至少${2\eta bn}$个值满足$\left| {\left\lbrack  A\left( c - d\right) \right\rbrack  }_{i}\right|  \leq  {2\alpha }$的概率将极小——小到对于一个随机的$A$，甚至不太可能存在一个与$d$相差甚远且未被$A$中的查询排除的$c$。

假设向量$z = \left( {c - d}\right)  \in  \{  - 2,0,2{\} }^{n}$的汉明重量（Hamming weight）至少为${\left( \frac{2\alpha }{\eta }\right) }^{2}$，因此$c$与$d$相差甚远。我们已经论证过，由于$c$是由攻击者生成的，对于$i$的至少${2\eta bn}$个值，有$\left| {\left( Az\right) }_{i}\right|  \leq  {2\alpha }$。我们将这样的$z$称为相对于$A$的“坏”值。我们将证明，在$A$的选择上，大概率不会有相对于$A$的“坏”$z$。

对于任意的$i,{v}_{i}z$，它是至少${\left( \frac{2\alpha }{\eta }\right) }^{2} \pm  2$个随机值的和。设$k = {\left( 2\alpha /\eta \right) }^{2}$和$\ell  = {2\alpha }$，根据引理8.3，${v}_{i}z$落在大小为${4\alpha }$的区间内的概率至多为$\eta$，因此$\left| {{v}_{i}z}\right|  \leq  {2\alpha }$成立的查询的期望数量至多为${\eta bn}$。切诺夫界（Chernoff bounds）现在表明，这个数量超过${2\eta bn}$的概率至多为$\exp \left( {-\frac{\eta bn}{4}}\right)$。因此，特定的$z = c - d$相对于$A$为“坏”的概率至多为$\exp \left( {-\frac{\eta bn}{4}}\right)$。

对至多 ${3}^{n}$ 种可能的 $z\mathrm{\;s}$ 取并集界，我们得到，至少以 $1 - \exp \left( {-n\left( {\frac{\eta b}{4} - \ln 3}\right) }\right)$ 的概率，不存在不良的 $z$。取 $b > 4\ln 3/\eta$ 时，存在这种不良 $z$ 的概率在 $n$ 上呈指数级小。

防止明显的非隐私性对于隐私机制来说是一个非常低的标准，因此，如果差分隐私有意义，那么防止明显非隐私性的下界也将适用于任何确保差分隐私的机制。尽管在本专著中我们大部分情况下忽略计算问题，但还存在攻击效率的问题。假设我们能够证明（也许在某些计算假设下）存在难以破解的低失真机制；例如，对于那些生成接近原始数据库的候选数据库 $c$ 很困难的机制呢？那么，尽管低失真机制在理论上可能无法实现差分隐私，但可以想象它能为有界对手提供隐私保护。不幸的是，情况并非如此。特别是，当噪声始终在 $o\left( \sqrt{n}\right)$ 中时，存在一种使用恰好 $n$ 个固定查询的高效攻击；此外，甚至存在一种计算高效的攻击，需要线性数量的查询，其中 0.239 的比例可能会被带有极大噪声的回答。

对于“互联网规模”的数据集，获取 $n$ 个查询的响应是不可行的，因为 $n$ 非常大，例如 $n \geq  {10}^{8}$。如果数据管理者只允许进行亚线性数量的查询会怎样呢？这一探究引出了（后来发展为）$\left( {\varepsilon ,\delta }\right)$ -差分隐私的首个算法结果，其中展示了如何通过向每个真实答案添加阶为 $o\left( \sqrt{n}\right)$ 的二项式噪声（小于采样误差！）来针对亚线性数量的计数查询保持隐私。利用差分隐私的工具，我们可以通过以下两种方式实现：（1）高斯机制或（2）拉普拉斯机制和高级组合。

### 8.2 差分隐私的下界

上一节的结果给出了确保任何合理隐私概念所需失真的下界。相比之下，本节的结果是针对差分隐私的。尽管证明中的一些细节相当技术性，但主要思想很精妙：假设（不知何故）对手已将可能的数据库集合缩小到一个相对较小的由 ${2}^{s}$ 个向量组成的集合 $S$，其中每对向量之间的 ${L}_{1}$ 距离是某个较大的数 $\Delta$。进一步假设我们可以找到一个 $k$ 维查询 $F$，其每个输出坐标都是 1 - 利普希茨的，并且具有这样的性质：在我们集合中的不同向量上，该查询的真实答案看起来非常不同（在 ${L}_{\infty }$ 范数下）；例如，集合中任意两个元素上的距离可能是 $\Omega \left( k\right)$。从几何角度思考“答案空间” ${\mathbb{R}}^{k}$ 会很有帮助。集合 $S$ 中的每个元素 $x$ 在答案空间中产生一个向量 $F\left( x\right)$。实际响应将是答案空间中该点的一个扰动。然后，基于体积的鸽巢原理（在答案空间中）表明，如果（带噪声的）响应以中等概率“合理地”接近真实答案，那么 $\epsilon$ 不可能非常小。

这源于以下事实：对于$\left( {\varepsilon ,0}\right)$ - 差分隐私机制$\mathcal{M}$，对于任意不同的数据库$x,y$，$\mathcal{M}\left( x\right)$的支撑集中的任何响应也在$\mathcal{M}\left( y\right)$的支撑集中。结合向量的适当集合构造和一个（人为设计的、非计数的）查询，该结果得出了一个线性的失真下界$k/\varepsilon$。该论证引用了讨论群组隐私的定理2.2。在我们的例子中，所讨论的群组对应于$S$中一对向量之间$\left( {L}_{1}\right)$距离的贡献指标。

#### 8.2.1 通过填充论证得出的下界

我们从一个直观的观察开始，即如果当查询为$F$时，“可能的”响应区域是不相交的，那么我们可以从下方界定$\epsilon$，表明隐私性不能太好。当$\parallel F\left( {x}_{i}\right)  -$ ${\left. F\left( {x}_{j}\right) \right.\parallel  }_{\infty }$很大时，这意味着为了获得非常好的隐私性，即使限制在许多地方不同的数据库上，我们也必须在$F$的某些坐标上得到非常错误的响应。

该论证使用了数据库的直方图表示。在后续内容中，$d = \left| \mathcal{X}\right|$表示抽取数据库元素的全集的大小。

引理8.4。假设存在一个集合$S = \left\{  {{x}_{1},\ldots ,{x}_{{2}^{s}}}\right\}$，其中每个${x}_{i} \in  {\mathbb{N}}^{d}$，使得对于$i \neq  j,{\begin{Vmatrix}{x}_{i} - {x}_{j}\end{Vmatrix}}_{1} \leq  \Delta$。此外，设$F$ : ${\mathbb{N}}^{d} \rightarrow  {\mathbb{R}}^{k}$是一个$k$维查询。对于$1 \leq  i \leq  {2}^{s}$，设${B}_{i}$表示${\mathbb{R}}^{k}$（答案空间）中的一个区域，并假设${B}_{i}$是相互不相交的。如果$\mathcal{M}$是$F$的一个$\left( {\varepsilon ,0}\right)$ - 差分隐私机制，使得$\forall 1 \leq  i \leq  {2}^{s},\Pr \left\lbrack  {\mathcal{M}\left( {x}_{i}\right)  \in  {B}_{i}}\right\rbrack   \geq  1/2$，那么$\varepsilon  \geq  \frac{\ln \left( 2\right) \left( {s - 1}\right) }{\Delta }$。

证明。根据假设$\Pr \left\lbrack  {\mathcal{M}\left( {x}_{j}\right)  \in  {B}_{j}}\right\rbrack   \geq  {2}^{-1}$。由于区域${B}_{1},\ldots ,{B}_{{2}^{s}}$是不相交的，$\exists j \neq  i \in  \left\lbrack  {2}^{s}\right\rbrack$使得$\Pr \left\lbrack  {\mathcal{M}\left( {x}_{i}\right)  \in  {B}_{j}}\right\rbrack   \leq  {2}^{-s}$。也就是说，对于至少一个${2}^{s} - 1$区域${B}_{j}$，$\mathcal{M}\left( {x}_{i}\right)$被映射到这个${B}_{j}$的概率至多为${2}^{-s}$。将此与差分隐私相结合，我们有

$$
\frac{{2}^{-1}}{{2}^{-s}} \leq  \frac{\mathop{\Pr }\limits_{\mathcal{M}}\left\lbrack  {{B}_{j} \mid  {x}_{j}}\right\rbrack  }{\mathop{\Pr }\limits_{\mathcal{M}}\left\lbrack  {{B}_{j} \mid  {x}_{i}}\right\rbrack  } \leq  \exp \left( {\varepsilon \Delta }\right) .
$$

推论8.5。设$S = \left\{  {{x}_{1},\ldots ,{x}_{{2}^{s}}}\right\}$如引理8.4中所定义，并假设对于任意$i \neq  j,{\begin{Vmatrix}F\left( {x}_{i}\right)  - F\left( {x}_{j}\right) \end{Vmatrix}}_{\infty } \geq  \eta$ 。设${B}_{i}$表示${\mathbb{R}}^{k}$中以${x}_{i}$为中心、半径为$\eta /2$的${L}_{\infty }$球。设$\mathcal{M}$是满足以下条件的$F$的任意$\varepsilon$ -差分隐私机制

$$
\forall 1 \leq  i \leq  {2}^{s} : \Pr \left\lbrack  {\mathcal{M}\left( {x}_{i}\right)  \in  {B}_{i}}\right\rbrack   \geq  1/2.
$$

那么$\varepsilon  \geq  \frac{\left( {\ln 2}\right) \left( {s - 1}\right) }{\Delta }$ 。

证明。区域${B}_{1},\ldots ,{B}_{{2}^{s}}$是不相交的，因此引理8.4的条件得到满足。通过应用该引理并取对数可得出此推论。

在下面的定理8.8中，我们将考虑查询$F$，它们只是$k$个独立随机生成的（非线性！）查询。对于合适的$S$和$F$（我们将努力找到这些值），该推论表明，如果至少以概率$1/2$所有响应同时具有小误差，那么隐私性就不会太好。换句话说，

断言8.6（推论8.5的非正式重述）。为了获得$\varepsilon  \leq  \frac{\ln \left( 2\right) \left( {s - 1}\right) }{\Delta }$的$\left( {\varepsilon ,0}\right)$ -差分隐私，该机制必须以超过$1/2$的概率添加${L}_{\infty }$范数大于$\eta /2$的噪声。

作为一个热身练习，我们证明一个需要大数据域的较简单定理。

定理8.7。设$\mathcal{X} = \{ 0,1{\} }^{k}$ 。设$\mathcal{M} : {\mathcal{X}}^{n} \rightarrow  {\mathbb{R}}^{k}$是一个$\left( {\varepsilon ,0}\right)$ -差分隐私机制，使得对于每个数据库$x \in  {\mathcal{X}}^{n}$，至少以概率$1/2\mathcal{M}\left( x\right)$输出$x$的所有一维边际分布，且误差小于$n/2$ 。也就是说，对于每个$j \in  \left\lbrack  k\right\rbrack$，$\mathcal{M}\left( x\right)$的第$j$个分量应近似等于$x$中第$j$位为1的行数，误差小于$n/2$ 。那么$n \in  \Omega \left( {k/\varepsilon }\right)$ 。

注意，根据简单组合定理，这个界在常数因子范围内是紧的，并且对于$\delta  \in  {2}^{-o\left( n\right) }$，它将$\left( {\varepsilon ,0}\right)$ -差分隐私与$\left( {\varepsilon ,\delta }\right)$ -差分隐私区分开来，因为根据高级组合定理（定理3.20），参数为$b = \sqrt{k\ln \left( {1/\delta }\right) }/\varepsilon$的拉普拉斯噪声足以满足前者，而后者需要$\Omega \left( {k/\varepsilon }\right)$ 。取$k \in  \Theta \left( n\right)$，例如$\delta  = {2}^{-{\log }^{2}n}$，就可以得到这种区分。

证明。对于每个字符串 $w \in  \{ 0,1{\} }^{k}$，考虑由 $n$ 个相同行组成的数据库 ${x}_{w}$，所有这些行都等于 $w$。设 ${B}_{w} \in  {\mathbb{R}}^{k}$ 由所有能为 $x$ 上的单向边际提供误差小于 $n/2$ 的答案的数字元组组成。即，

$$
{B}_{w} = \left\{  \left( {{a}_{1},\ldots ,{a}_{k}}\right) \right\}   \in  {\mathbb{R}}^{k} : \forall i \in  \left\lbrack  k\right\rbrack  \left| {{a}_{i} - n{w}_{i}}\right|  < n/2\} .
$$

换句话说，${B}_{w}$ 是以 ${nw} \in  \{ 0,n{\} }^{k}$ 为中心、半径为 $n/2$ 的开 ${\ell }_{\infty }$。注意，集合 ${B}_{w}$ 是互不相交的。

如果 $\mathrm{M}$ 是一个用于回答单向边际的准确机制，那么对于每个 $w$，当数据库为 ${x}_{w}$ 时落入 ${B}_{w}$ 的概率至少应为 $1/2 : \Pr \left\lbrack  {\mathcal{M}\left( {x}_{w}\right)  \in  {B}_{w}}\right\rbrack   \geq  1/2$。因此，在推论 8.5 中令 $\Delta  = n$ 和 $s = k$，我们有 $\varepsilon  \geq  \frac{\left( {\ln 2}\right) \left( {s - 1}\right) }{\Delta }$。

定理 8.8。对于任意的 $k,d,n \in  \mathbb{N}$ 和 $\varepsilon  \in  (0,1/{40}\rbrack$，其中 $n \geq$ $\min \{ k/\varepsilon ,d/\varepsilon \}$，存在一个每个坐标敏感度至多为 1 的查询 $F : {\mathbb{N}}^{d} \rightarrow  {\mathbb{R}}^{k}$，使得任何 $\left( {\varepsilon ,0}\right)$ -差分隐私机制在某些权重至多为 $n$ 的数据库上以至少 $1/2$ 的概率添加 ${L}_{\infty }$ 范数为 $\Omega \left( {\min \{ k/\varepsilon ,d/\varepsilon \} }\right)$ 的噪声。

注意，与定理 8.7 中的要求不同，这里的 $d = \left| \mathcal{X}\right|$ 不必很大。

证明。设 $\ell  = \min \{ k,d\}$。利用纠错码，我们可以构造一个集合 $S = \left\{  {{x}_{1},\ldots ,{x}_{{2}^{s}}}\right\}$，其中 $s = \ell /{400}$，使得每个 ${x}_{i} \in  {\mathbb{N}}^{d}$，此外

1. $\forall i : {\begin{Vmatrix}{x}_{i}\end{Vmatrix}}_{1} \leq  w = \ell /\left( {1280\varepsilon }\right)$

2. $\forall i \neq  j,{\begin{Vmatrix}{x}_{i} - {x}_{j}\end{Vmatrix}}_{1} \geq  w/{10}$

我们在此不给出详细信息，但我们注意到$S$中的数据库大小至多为$w < n$，因此${\begin{Vmatrix}{x}_{i} - {x}_{j}\end{Vmatrix}}_{1} \leq  {2w}$。取$\Delta  = {2w}$，集合$S$满足推论8.5的条件。我们接下来的工作是得到查询$F$，以便应用推论8.5。给定$S = \left\{  {{x}_{1},\ldots ,{x}_{{2}^{s}}}\right\}$，其中每个${x}_{i} \in  {\mathbb{N}}^{d}$，第一步是定义一个从直方图空间到${\mathbb{R}}^{{2}^{s}},{\mathcal{L}}_{S} : {\mathbb{N}}^{d} \rightarrow  {\mathbb{R}}^{{2}^{s}}$中向量的映射。直观地（且不精确地！），给定一个直方图$x$，该映射会列出对于每个${x}_{i} \in  S$，从$x$到${x}_{i}$的${L}_{1}$距离。更精确地说，设$w$是我们集合中任何${x}_{i}$的权重上限，我们按如下方式定义该映射。

- 对于每个${x}_{i} \in  S$，在映射中有一个坐标$i$。

- ${\mathcal{L}}_{S}\left( x\right)$的第$i$个坐标是$\max \left\{  {w/{30} - {\begin{Vmatrix}{x}_{i} - z\end{Vmatrix}}_{1},0}\right\}$。

命题8.9。如果${x}_{1},\ldots ,{x}_{{2}^{s}}$满足条件

1. $\forall i{\begin{Vmatrix}{x}_{i}\end{Vmatrix}}_{1} \leq  w$；并且

2. $\forall i \neq  j{\begin{Vmatrix}{x}_{i} - {x}_{j}\end{Vmatrix}}_{1} \geq  w/{10}$

那么映射${\mathcal{L}}_{S}$是1 - 利普希茨的；特别地，如果${\begin{Vmatrix}{z}_{1} - {z}_{2}\end{Vmatrix}}_{1} = 1$，那么${\begin{Vmatrix}{\mathcal{L}}_{S}\left( {z}_{1}\right)  - {\mathcal{L}}_{S}\left( {z}_{2}\right) \end{Vmatrix}}_{1} \leq  1$，假设$w \geq  {31}$。证明。由于我们假设$w \geq  {31}$，我们有如果$z \in  {\mathbb{N}}^{d}$接近某个${x}_{i} \in  S$，即$w/{30} > {\begin{Vmatrix}{x}_{i} - z\end{Vmatrix}}_{1}$，那么$z$不可能接近任何其他${x}_{j} \in  S$，并且对于所有${\begin{Vmatrix}{z}^{\prime } - z\end{Vmatrix}}_{1} \leq  1$都成立。因此，对于任何满足$\begin{Vmatrix}{{z}_{1} - {z}_{2}}\end{Vmatrix} \leq  1$的${z}_{1},{z}_{2}$，如果$A$表示${\mathcal{L}}_{S}\left( {z}_{1}\right)$或${\mathcal{L}}_{S}\left( {z}_{2}\right)$中至少有一个非零的坐标集合，那么$A$要么为空集，要么为单元素集。鉴于此，该命题中的陈述可直接从对应于任何特定坐标的映射显然是1 - 利普希茨的这一事实得出。

我们最终可以描述查询 $F$。对应于任意 $r \in$ $\{  - 1,1{\} }^{{2}^{s}}$，我们将 ${f}_{r} : {\mathbb{N}}^{d} \rightarrow  \mathbb{R}$ 定义为

$$
{f}_{r}\left( x\right)  = \mathop{\sum }\limits_{{i = 1}}^{d}{\mathcal{L}}_{S}{\left( x\right) }_{i} \cdot  {r}_{i}
$$

这仅仅是内积 ${\mathcal{L}}_{S} \cdot  r.F$ 将是一个随机映射 $F : {\mathbb{N}}^{d} \rightarrow  {\mathbb{R}}^{k}$：独立且均匀地随机选取 ${r}_{1},\ldots ,{r}_{k} \in  \{  - 1,1{\} }^{{2}^{s}}$ 并定义

$$
F\left( x\right)  = \left( {{f}_{{r}_{1}}\left( x\right) ,\ldots ,{f}_{{r}_{k}}\left( x\right) }\right) .
$$

也就是说，$F\left( x\right)$ 仅仅是 ${\mathcal{L}}_{S}\left( x\right)$ 与随机选取的 $k$ 个 $\pm  1$ 向量的内积结果。

注意，对于任意 $x \in  S{\mathcal{L}}_{S}\left( x\right)$ 有一个坐标的值为 $w/{30}$（其他坐标均为零），因此对于 $\forall {r}_{i} \in  \{  - 1,1{\} }^{{2}^{s}}$ 和 $x \in  S$，我们有 $\left| {{f}_{{r}_{i}}\left( x\right) }\right|  = w/{30}$。现在考虑任意 ${x}_{h},{x}_{j} \in  S$，其中 $h \neq  j$。由此可知，对于任意 ${r}_{i} \in  \{  - 1,1{\} }^{{2}^{s}}$，

$$
\mathop{\Pr }\limits_{{r}_{i}}\left\lbrack  {\left| {{f}_{{r}_{i}}\left( {x}_{h}\right)  - {f}_{{r}_{i}}\left( {x}_{j}\right) }\right|  \geq  w/{15}}\right\rbrack   \geq  1/2
$$

（当 ${\left( {r}_{i}\right) }_{h} =  - {\left( {r}_{i}\right) }_{j}$ 时，此事件发生）。切尔诺夫界（Chernoff bound）的一个基本应用表明

$$
\mathop{\Pr }\limits_{{{r}_{1},\ldots ,{r}_{k}}}\text{[For at least}1/{10}\text{of the}{r}_{i}\mathrm{\;s}\text{,}
$$

$$
\left| {{f}_{{r}_{i}}\left( {x}_{h}\right)  - {f}_{{r}_{i}}\left( {x}_{j}\right) }\right|  \geq  w/{15}\rbrack  \geq  1 - {2}^{-k/{30}}.
$$

现在，满足 ${x}_{i},{x}_{j} \in  S$ 的数据库对 $\left( {{x}_{i},{x}_{j}}\right)$ 的总数至多为 ${2}^{2s} \leq  {2}^{k/{200}}$。取并集界，这意味着

$$
\mathop{\Pr }\limits_{{{r}_{1},\ldots ,{r}_{k}}}\lbrack \forall h \neq  j,\;\text{ For at least }1/{10}\text{ of the }{r}_{i}\mathrm{\;s},
$$

$$
\left. {\left| {{f}_{{r}_{i}}\left( {x}_{h}\right)  - {f}_{{r}_{i}}\left( {x}_{j}\right) }\right|  \geq  w/{15}}\right\rbrack   \geq  1 - {2}^{-k/{40}}
$$

这意味着我们可以固定 ${r}_{1},\ldots ,{r}_{k}$，使得以下情况成立。

$\forall h \neq  j,\;$ 对于 ${r}_{i}\mathrm{\;s},\;\left| {{f}_{{r}_{i}}\left( {x}_{h}\right)  - {f}_{{r}_{i}}\left( {x}_{j}\right) }\right|  \geq  w/{15}$ 中的至少 $1/{10}$ 个。因此，对于任意 ${x}_{h} \neq  {x}_{j} \in  S,{\begin{Vmatrix}F\left( {x}_{h}\right)  - F\left( {x}_{j}\right) \end{Vmatrix}}_{\infty } \geq  w/{15}$。

设置 $\Delta  = {2w}$ 和 $s = \ell /{400} > {3\varepsilon w}$（如我们上面所做的），以及 $\eta  = w/{15}$，我们满足推论 8.5 的条件，并得出 $\Delta  \leq  \left( {s - 1}\right) /\varepsilon$，从而证明了该定理（通过断言 8.6）。

该定理几乎是紧的：如果 $k \leq  d$，那么我们可以对 $F$ 中每个敏感度为 1 的分量查询应用参数为 $k/\varepsilon$ 的拉普拉斯机制（Laplace mechanism），并且我们预计最大失真为 $\Theta \left( {k\ln k/\varepsilon }\right)$。另一方面，如果 $d \leq  k$，那么我们可以对表示数据库的 $d$ 维直方图应用拉普拉斯机制，并且我们预计最大失真为 $\Theta \left( {d\ln d/\varepsilon }\right)$。

该定理实际上表明，给定集合$S$的知识以及实际数据库是元素$x \in  S$的知识，如果失真的${L}_{\infty }$范数过小，攻击者可以完全确定$x$。在现实生活中，攻击者如何获得攻击中使用的那种集合$S$呢？当一个非隐私数据库系统在一个数据集（例如$x$）上运行时，就可能出现这种情况。例如，$x$可以是$\{ 0,1{\} }^{n}$中的一个向量，攻击者可能通过一系列线性查询得知$x \in  \mathcal{C}$，即一个距离为（例如${n}^{2/3}$）的线性码。当然，如果数据库系统不承诺隐私，那就没有问题。问题在于，如果管理员在多次查询收到无噪声响应后，决定用差分隐私机制取代现有的系统。特别是，如果管理员选择对后续的$k$查询使用$\left( {\varepsilon ,\delta }\right)$ - 差分隐私，那么失真可能会低于$\Omega \left( {k/\varepsilon }\right)$下界，从而允许进行定理8.8证明中描述的攻击。

该定理还强调，关于数据库成员（集合）的辅助信息与关于整个数据库的信息之间存在根本差异。当然，我们早已知道这一点：被告知秘密比特的总数恰好为5000，会完全破坏差分隐私，而且一个已经知道除了一个个体之外数据库中每个成员的秘密比特的攻击者，就可以推断出剩余个体的秘密比特。

额外的结论。假设$k \leq  d$，那么在定理8.8中$\ell  = k$。上一节中概述的针对$k$查询的关于$k/\varepsilon$的线性噪声下界，立即揭示了计数查询和任意1 - 敏感度查询之间的区别，因为SmallDB构造在保持差分隐私的同时，以大约${n}^{2/3}$的噪声回答（超过）$n$个查询。实际上，这个结果还使我们能够得出结论，对于任意低敏感度查询的大集合，不存在小的$\alpha$ - 网，对于$\alpha  \in  o\left( n\right)$（因为否则网机制将产生一个具有所需精度的$\left( {\varepsilon ,0}\right)$算法）。

### 8.3 参考文献注释

包括定理8.1在内的第一批重构攻击归功于迪努尔（Dinur）和尼斯姆（Nissim）[18]，他们还给出了一种攻击方法，只要噪声始终为$o\left( \sqrt{n}\right)$，该攻击只需要多项式时间计算和$O\left( {n{\log }^{2}n}\right)$次查询。迪努尔、德沃克（Dwork）和尼斯姆意识到，当$n$达到“互联网规模”时，需要$n$次随机线性查询的攻击是不可行的，于是他们给出了第一个正面结果，表明对于次线性数量的子集和查询，可以通过添加缩放为$o\left( \sqrt{n}\right)$的噪声来实现一种隐私形式（现在已知这意味着$\left( {\varepsilon ,\delta }\right)$ - 差分隐私）[18]。这很令人兴奋，因为它表明，如果我们认为数据库是从一个基础总体中抽取的，那么，即使对于相对大量的计数查询，也可以用小于采样误差的失真来实现隐私。这最终通过更一般的查询[31, 6]引出了差分隐私。将这些查询视为一种隐私保护编程原语[6]的观点，启发了麦克谢里（McSherry）的隐私集成查询编程平台[59]。

定理8.2的重构攻击出现在[24]中，德沃克、麦克谢里和塔尔瓦尔（Talwar）在该文献中表明，即使有0.239比例的响应具有极大的任意噪声，只要其他响应的噪声为$o\left( \sqrt{n}\right)$，就可以进行多项式时间重构。

几何方法，尤其是引理8.4，归功于哈特（Hardt）和塔尔瓦尔（Talwar）[45]，他们还给出了一种基于几何的算法，在一个普遍被认可的猜想下，证明了对于少量$k \leq  n$查询，这些边界是紧的。后来，巴斯卡拉（Bhaskara）等人[5]消除了对该猜想的依赖。尼科洛夫（Nikolov）等人[66]将几何方法扩展到任意数量的查询，他们给出了一种具有实例最优均方误差的算法。对于少量查询的情况，通过一个增强论证，这会导致较低的预期最坏情况误差。定理8.8归功于德（De）[17]。

## 9 差分隐私与计算复杂度

到目前为止，我们对差分隐私的讨论忽略了计算复杂度问题，允许数据管理者和攻击者的计算能力不受限制。实际上，数据管理者和攻击者的计算能力可能都是受限的。

将我们自己限制在计算能力受限的数据管理者范围内，会限制数据管理者的操作，使得实现差分隐私变得更加困难。实际上，我们将展示一类计数查询的示例，在标准的复杂度理论假设下，即使已知低效算法，如SmallDB和私有乘法权重算法，也无法高效生成合成数据库。大致来说，数据库行是数字签名，使用数据管理者无法访问的密钥进行签名。直观地说，合成数据库中的任何一行要么是从原始数据库复制而来（违反隐私），要么必须是对新消息的签名，即伪造签名（违反数字签名方案的不可伪造性）。不幸的是，这种情况并不局限于基于数字签名的（可能是人为构造的）示例：即使创建一个能保持相对准确的双向边际的合成数据库也很困难。从积极的方面来看，给定一组$\mathcal{Q}$查询和一个从全域$\mathcal{X}$中抽取行的$n$行数据库，可以在关于$n,\left| \mathcal{X}\right|$和$\left| \mathcal{Q}\right|$的多项式时间内生成一个合成数据库。

如果我们放弃合成数据库的目标，满足于一种数据结构，从中我们可以获得每个查询答案的相对准确近似值，那么情况会有趣得多。事实证明，这个问题与追踪叛徒问题密切相关，在追踪叛徒问题中，目标是在向付费客户分发数字内容的同时阻止盗版。

如果对手被限制在多项式时间内，那么实现差分隐私就会变得更容易。事实上，安全函数评估这一极其强大的概念提供了一种自然的方法来避免使用可信的数据管理者（同时比随机响应方法具有更高的准确性），也提供了一种自然的方法来允许多个出于法律原因不能共享其数据集的可信数据管理者，对实际上是合并后的数据集进行查询响应。简而言之，安全函数评估是一种密码学原语，它允许一组 $n$ 个参与方 ${p}_{1},{p}_{2},\ldots ,{p}_{n}$ （其中故障参与方的比例小于某个固定分数，该分数根据故障类型而异；对于“诚实但好奇”的故障，该分数为 1）合作计算任何函数 $f\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ ，其中 ${x}_{i}$ 是参与方 ${p}_{i}$ 的输入或值，以这样一种方式进行计算，即任何故障参与方联盟都无法破坏计算过程，也无法了解非故障参与方的值，除非这些值可以从函数输出和联盟成员的值中推导出来。这两个属性传统上被称为正确性和隐私性。这种隐私概念，我们称之为安全函数评估隐私（SFE 隐私），与差分隐私有很大不同。设 $V$ 是故障参与方持有的值的集合，设 ${p}_{i}$ 是一个非故障参与方。如果 ${x}_{i}$ 可以从 $V \cup  \left\{  {f\left( {{x}_{1},\ldots ,{x}_{n}}\right) }\right\}$ 中推导出来，SFE 隐私允许故障参与方了解 ${x}_{i}$ ；因此，差分隐私不允许精确发布 $f\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ 。然而，用于计算函数 $f$ 的安全函数评估协议可以很容易地修改为 $f$ 的差分隐私协议，只需定义一个新函数 $g$ ，它是在 $f$ 的值上添加拉普拉斯噪声 $\operatorname{Lap}\left( {{\Delta f}/\varepsilon }\right)$ 的结果。原则上，安全函数评估允许对 $g$ 进行评估。由于 $g$ 是差分隐私的，并且将 SFE 隐私属性应用于 $g$ 时表明，除了从 $g\left( {{x}_{1},\ldots ,{x}_{n}}\right)$ 的值和 $V$ 中可以了解到的信息之外，无法了解到关于输入的任何其他信息，因此，只要故障参与者被限制在多项式时间内，就可以确保差分隐私。因此，安全函数评估允许在不使用可信数据管理者的情况下实现差分隐私的计算概念，并且与使用可信数据管理者时所能达到的准确性相比没有损失。特别是，在确保计算差分隐私的同时，可以以恒定的预期误差回答计数查询，而无需可信数据管理者。我们将看到，在不使用密码学的情况下，误差必须为 $\Omega \left( {n}^{1/2}\right)$ ，这证明了在多方情况下，计算假设确实可以提高准确性。

---

<!-- Footnote -->

${}^{1}$ 回想一下，双向边际是指对于每一对属性值，数据库中具有该属性值对的行数的计数。

${}^{2}$ 在“诚实但好奇”的情况下，我们可以为任何参与方 ${P}_{j}$ 设定 $V = \left\{  {x}_{j}\right\}$ 。

<!-- Footnote -->

---

### 9.1 多项式时间的数据管理者

在本节中，我们表明，在标准的密码学假设下，要创建一个合成数据库，使其能够对适当选择的一类计数查询给出准确答案，同时确保哪怕是最基本的隐私概念，在计算上也是困难的。

这一结果有几个扩展；例如，查询集较小（但数据域仍然很大）的情况，以及数据域较小（但查询集很大）的情况。此外，对于某些自然的查询族，如对应于合取的查询族，也得到了类似的负面结果。

我们将使用“合成（syntheticize）”这一术语来表示以保护隐私的方式生成合成数据库的过程${}^{3}$。因此，本节的结果涉及合成过程的计算难度。我们所定义的隐私概念将远弱于差分隐私，因此合成的难度意味着以差分隐私的方式生成合成数据库也具有难度。具体而言，如果即使避免完整泄露输入项都很困难，我们就称合成是困难的。也就是说，总会有某些项完全暴露。

---

<!-- Footnote -->

${}^{3}$ 在第6节中，合成器的输入是一个概要；这里我们从一个数据库开始，它是一个简单的概要。

<!-- Footnote -->

---

请注意，如果相反，泄露少量输入项不被视为隐私泄露，那么通过发布输入项的一个随机子集就可以轻松实现合成。这个“合成数据库”的实用性来自采样边界：在很大概率上，即使对于大量计数查询，这个子集也能保留实用性。

在引入复杂性假设时，我们需要一个安全参数来表示大小；例如，集合的大小、消息的长度、解密密钥的比特数等等，以及表示计算难度。安全参数用$\kappa$表示，代表“合理”的大小和工作量。例如，假设对一个大小为安全参数的（任意固定）多项式的集合进行穷举搜索是可行的。

计算复杂性是一个渐近概念——我们关注的是随着对象（数据全域、数据库、查询族）的大小增长，任务的难度如何增加。因此，例如，我们不仅需要考虑单一大小数据库的分布（在本专著的其余部分我们称之为$n$），还需要考虑由安全参数索引的分布族。与此相关，当我们引入复杂性时，我们倾向于“弱化”断言：伪造签名并非不可能——也许会有运气！相反，我们假设没有高效算法能以不可忽略的概率成功，其中“高效”和“不可忽略”是根据安全参数定义的。在我们的直观讨论中，我们将忽略这些细节，但在正式的定理陈述中会保留它们。

非正式地说，如果对于任何高效的（所谓的）合成器，从该分布中抽取的数据库在很大概率上，至少有一个数据库项可以从所谓的合成器的输出中提取出来，那么数据库的一个分布就很难合成（相对于某个查询族$\mathcal{Q}$）。当然，为了避免平凡情况，我们还要求当这个泄露的项从输入数据库中排除（例如，用一个随机的不同项替换）时，它能从输出中提取出来的概率非常小。这意味着任何高效的（所谓的）合成器确实在很强的意义上损害了输入项的隐私。

下面的定义9.1将形式化我们对合成器的实用性要求。有三个参数：$\alpha$描述了准确性要求（在$\alpha$范围内被认为是准确的）；$\gamma$描述了成功合成允许不准确的查询比例，$\beta$将是失败的概率。

对于一个产生合成数据库的算法$A$，如果对于$1 - \gamma$比例的查询$q \in  \mathcal{Q}$有$\left| {q\left( {A\left( x\right) }\right)  - q\left( x\right) }\right|  \leq  \alpha$，我们就说输出$A\left( x\right)$对于查询集$\mathcal{Q}$是$\left( {\alpha ,\gamma }\right)$ - 准确的。

定义9.1 $\left( {\left( {\alpha ,\beta ,\gamma }\right) \text{-Utility}}\right)$。设$\mathcal{Q}$是一个查询集，$\mathcal{X}$是一个数据全域。如果对于任何$n$项数据库$x$，合成器$A$对于$\mathcal{Q}$和$\mathcal{X}$具有$\left( {\alpha ,\beta ,\gamma }\right)$ - 实用性：

$$
\Pr \left\lbrack  {A\left( x\right) \text{ is }\left( {\alpha ,\gamma }\right) \text{-accurate for }\mathcal{Q}}\right\rbrack   \geq  1 - \beta 
$$

其中概率是基于$A$的随机选择。

设$\mathcal{Q} = {\left\{  {\mathcal{Q}}_{n}\right\}  }_{n = 1,2,\ldots }$为查询族集合，$\mathcal{X} = {\left\{  {\mathcal{X}}_{n}\right\}  }_{n = 1,2,\ldots }$为数据全域集合。若一个算法的运行时间为关于$\left( {n,\log \left( \left| {\mathcal{Q}}_{n}\right| \right) ,\log \left( \left| {\mathcal{X}}_{n}\right| \right) }\right)$的多项式，则称该算法是高效的。

在接下来的定义中，我们将描述一族分布难以合成意味着什么。更具体地说，我们将说明生成提供$\left( {\alpha ,\gamma }\right)$ - 精度的合成数据库困难意味着什么。和往常一样，我们必须将其表述为一个渐近性陈述。

定义9.2（$(\left( {\mu ,\alpha ,\beta ,\gamma ,\mathcal{Q}}\right)$ - 难以合成的数据库分布）。设$\mathcal{Q} = {\left\{  {\mathcal{Q}}_{n}\right\}  }_{n = 1,2,\ldots }$为查询族集合，$\mathcal{X} =$ ${\left\{  {\mathcal{X}}_{n}\right\}  }_{n = 1,2,\ldots }$为数据全域集合，且设$\mu ,\alpha ,\beta ,\gamma  \in  \left\lbrack  {0,1}\right\rbrack$。设$n$为数据库大小，$\mathcal{D}$为分布集合，其中${\mathcal{D}}_{n}$是关于从${X}_{n}$中选取的$n + 1$个项的集合。

我们用$\left( {x,i,{x}_{i}^{\prime }}\right)  \sim  {\mathcal{D}}_{n}$表示这样一个实验：选择一个$n$ - 元素数据库，从$\left\lbrack  n\right\rbrack$中均匀选取一个索引$i$，并从${\mathcal{X}}_{n}$中选取一个额外元素${x}_{i}^{\prime }$。从${\mathcal{D}}_{n}$中抽取的一个样本会得到一对数据库：$x$以及将$x$的第$i$个元素（在规范排序下）替换为${x}_{i}^{\prime }$后的结果。因此，我们认为${\mathcal{D}}_{n}$指定了一个关于$n$ - 项数据库（及其相邻数据库）的分布。

我们称$\mathcal{D}$是$\left( {\mu ,\alpha ,\beta ,\gamma ,\mathcal{Q}}\right)$ - 难以合成的，如果存在一个高效算法$T$，使得对于任何所谓的高效合成器$A$，以下两个条件成立：

1. 在数据库 $x \sim  \mathcal{D}$ 的选择以及 $A$ 和 $T$ 的随机硬币抛掷结果上，以概率 $1 - \mu$ 而言，如果 $A\left( x\right)$ 对 $1 - \gamma$ 比例的查询保持 $\alpha$ -效用，那么 $T$ 可以从 $A\left( x\right)$ 中恢复 $x$ 的某一行：

概率

$\left( {x,i,{x}_{i}^{\prime }}\right)  \sim  {D}_{n}$

$A,T$ 的硬币抛掷结果

$\left\lbrack  {\left( {A\left( x\right) \text{ maintains }\left( {\alpha ,\beta ,\gamma }\right) \text{-utility}}\right) \text{ and }\left( {x \cap  T\left( {A\left( x\right) }\right)  = \varnothing }\right) }\right\rbrack   \leq  \mu$

2. 对于每一个高效算法 $A$，以及每一个 $i \in  \left\lbrack  n\right\rbrack$，如果我们从 $D$ 中抽取 $\left( {x,i,{x}_{i}^{\prime }}\right)$，并将 ${x}_{i}$ 替换为 ${x}_{i}^{\prime }$ 以形成 ${x}^{\prime },T$，那么除了以极小的概率外，无法从 $A\left( {x}^{\prime }\right)$ 中提取 ${x}_{i}$：

$$
\begin{array}{l} \;\mathop{\Pr }\limits_{{\left( {x,i,{x}_{i}^{\prime }}\right)  \sim  {D}_{n}}}\left\lbrack  {{x}_{i} \in  T\left( {A\left( {x}^{\prime }\right) }\right) }\right\rbrack   \leq  \mu . \\  \text{ coin flips of }A,T \\  \end{array}
$$

稍后，我们将关注能生成任意概要（不一定是合成数据库）的离线机制。在这种情况下，我们将关注难以清理（而非难以合成）的相关概念，为此我们只需去掉 $A$ 生成合成数据库这一要求。

### 9.2 一些难以合成的分布

我们现在构造三种难以合成的分布。

一个签名方案由三元组（可能是随机化的）算法（Gen、Sign、Verify）给出：

- Gen：${1}^{\mathbb{N}} \rightarrow  {\left\{  {\left( \mathrm{{SK}},\mathrm{{VK}}\right) }_{n}\right\}  }_{n = 1,2,\ldots }$ 用于生成一个由（秘密）签名密钥和（公开）验证密钥组成的对。它仅以一元形式表示的安全参数 $\kappa  \in  \mathbb{N}$ 作为输入，并生成一个从 ${\left( \mathrm{{SK}},\mathrm{{VK}}\right) }_{\kappa }$ 中抽取的对，${\left( \mathrm{{SK}},\mathrm{{VK}}\right) }_{\kappa }$ 是由 $\kappa$ 索引的（签名、验证）密钥对的分布；我们分别用 ${p}_{s}\left( \kappa \right) ,{p}_{v}\left( \kappa \right) ,\ell s\left( \kappa \right)$ 表示签名密钥、验证密钥和签名的长度。

- Sign：${\mathrm{{SK}}}_{\kappa } \times  \{ 0,1{\} }^{\ell \left( \kappa \right) } \rightarrow  \{ 0,1{\} }^{\ell s\left( \kappa \right) }$ 以从 ${\left( \mathrm{{SK}},\mathrm{{VK}}\right) }_{\kappa }$ 中抽取的密钥对中的签名密钥和长度为 $\ell \left( \kappa \right)$ 的消息 $m$ 作为输入，并生成 $m$ 的签名；

- Verify：${\mathrm{{VK}}}_{\kappa } \times  \{ 0,1{\} }^{ * } \times  \{ 0,1{\} }^{\ell \left( \kappa \right) } \rightarrow  \{ 0,1\}$ 以验证密钥、字符串 $\sigma$ 和长度为 $\ell \left( \kappa \right)$ 的消息 $m$ 作为输入，并检查 $\sigma$ 是否确实是在给定验证密钥下 $m$ 的有效签名。

密钥、消息长度和签名长度在 $\kappa$ 上均为多项式。

所需的安全性概念是，给定任意多项式（关于 $\kappa$）数量的有效（消息，签名）对，伪造任何新的签名是困难的，即使是对先前已签名消息的新签名（请回想一下，签名算法可能是随机化的，因此在相同的签名密钥下，同一消息可能存在多个有效签名）。这样的签名方案可以从任何单向函数构造出来。通俗地说，单向函数是易于计算的函数—— $f\left( x\right)$ 可以在关于 $x$ 的长度（比特数）的多项式时间内计算出来，但难以求逆：对于每个概率多项式时间算法，在安全参数 $\kappa$ 的多项式时间内运行，在 $f$ 的定义域中随机选择 $x$ 时，找到 $f\left( x\right)$ 的任何有效原像的概率，增长速度比 $\kappa$ 的任何多项式的倒数都慢。

难以合成分布 I：固定一个任意的签名方案。计数查询集合 ${\mathcal{Q}}_{\kappa }$ 为每个验证密钥 ${vk} \in  {\mathrm{{VK}}}_{\kappa }$ 包含一个计数查询 ${q}_{vk}$。数据全域 ${\mathcal{X}}_{\kappa }$ 由所有可能的（消息，签名）对组成，这些对的形式是使用 ${\mathrm{{VK}}}_{\kappa }$ 中的密钥对长度为 $\ell \left( \kappa \right)$ 的消息进行签名得到的。

数据库上的分布 ${\mathcal{D}}_{\kappa }$ 由以下采样过程定义。运行签名方案生成器 $\operatorname{Gen}\left( {1}^{\kappa }\right)$ 以获得（私钥，公钥）。在 $\{ 0,1{\} }^{\ell \left( \kappa \right) }$ 中随机选择 $n = \kappa$ 条消息，并对每条消息运行签名过程，得到一组由密钥 ${sk}$ 签名的 $n$ 个（消息，签名）对。这就是数据库 $x$。请注意，数据库中的所有消息都使用相同的签名密钥进行签名。

数据全域项 $\left( {m,\sigma }\right)$ 满足谓词 ${q}_{vk}$ 当且仅当 $\operatorname{Verify}\left( {{vk},m,\sigma }\right)  = 1$，即根据验证密钥 ${vk}$，$\sigma$ 是 $m$ 的有效签名。

设 $x{ \in  }_{R}{\mathcal{D}}_{\kappa }$ 为一个数据库，设 ${sk}$ 为所使用的签名密钥，对应的验证密钥为 ${vk}$。假设合成器生成了 $y$，那么 $y$ 的几乎所有行在 ${vk}$ 下都必须是有效签名（因为查询 ${vk}$ 中 $x$ 的分数计数为 1）。根据签名方案的不可伪造性，所有这些签名都必须来自输入数据库 $x -$，因为多项式时间受限的管理者在时间 poly $\left( \kappa \right)$ 内无法生成新的有效（消息，签名）对。更正式地（只是稍微更正式），一个高效算法能够生成一个可以用密钥 ${vk}$ 验证但不在 $x$ 中的（消息，签名）对的概率是可以忽略不计的，因此，一个高效合成器生成的任何 $y$ 极有可能只包含 $x\frac{4}{3}$ 的行。这与（任何合理的）隐私概念相矛盾。

在这种构造中，${\mathcal{Q}}_{\kappa }$（验证密钥集合）和${\mathcal{X}}_{\kappa }$（(消息,签名)对集合）都很大（相对于$\kappa$是超多项式的）。当这两个集合都较小时，就可以高效地进行合成数据集的差分隐私生成。也就是说，存在一个差分隐私合成器，其运行时间相对于$n = \kappa ,\left| {\mathcal{Q}}_{\kappa }\right|$和$\left| {\mathcal{X}}_{\kappa }\right|$是多项式的：使用拉普拉斯机制计算带噪声的计数以获得概要，然后运行第6节中的合成器。因此，当这两个集合的大小相对于$\kappa$是多项式时，合成器的运行时间相对于$\kappa$也是多项式的。

我们现在简要讨论将第一个困难性结果推广到其中一个集合较小（但另一个仍然很大）的情况。

难以合成的分布II：在上述数据库分布中，我们选择了一个单一的(sk,vk)密钥对，并生成了一个消息数据库，所有消息都使用${sk}$进行签名；通过要求合成器在${sk}$下生成一个新签名，使得合成后的数据库能够为查询${q}_{vk}$提供准确答案，从而得到了困难性。为了在查询集合的大小仅相对于安全参数是多项式时获得合成的困难性，我们再次使用数字签名，用唯一的密钥进行签名，但我们无法为每个可能的验证密钥${vk}$设置一个查询，因为这些密钥数量太多。

---

<!-- Footnote -->

${}^{4}$量化顺序很重要，否则合成器可能会将签名密钥硬编码进去。我们首先固定合成器，然后运行生成器并构建数据库。概率是基于实验中的所有随机性：密钥对的选择、数据库的构建以及合成器使用的随机性。

<!-- Footnote -->

---

为了解决这个问题，我们做了两个改变：

1. 数据库行现在的形式为(验证密钥, 消息, 签名)。更准确地说，数据全域由(key,message,signature)三元组$\mathcal{X} = \left\{  {\left( {{vk},m,s}\right)  : {vk} \in  {\mathrm{{VK}}}_{\kappa },m \in  }\right.$ $\{ 0,1{\} }^{\ell \left( \kappa \right) },s \in  \{ 0,1{\} }^{\ell s\left( \kappa \right) }\}$组成。

2. 我们向查询类中精确添加$2{p}_{v}\left( \kappa \right)$个查询，其中${p}_{v}\left( \kappa \right)$是运行生成算法$\operatorname{Gen}\left( {1}^{\kappa }\right)$产生的验证密钥的长度。查询的形式为(i,b)，其中$1 \leq  i \leq  {p}_{v}\left( \kappa \right)$且$b \in  \{ 0,1\}$。查询“(i,b)”的含义是，“数据库行中形式为(vk,m,s)且$\operatorname{Verify}\left( {{vk},m,s}\right)  = 1$并且${vk}$的第$i$位是$b$的行所占的比例是多少？” 通过用根据单个密钥${vk}$签名的消息填充数据库，我们确保当$v{k}_{i} = b$时，对于所有$1 \leq  i \leq  p\left( \kappa \right)$，这些查询的响应应该接近1，而当$v{k}_{i} = 1 - b$时，应该接近0。

考虑到这一点，数据库上难以合成的分布是通过以下采样过程构建的：生成一个签名 - 验证密钥对$\left( {{sk},{vk}}\right)  \leftarrow  \operatorname{Gen}\left( {1}^{\kappa }\right)$，并从$\{ 0,1{\} }^{\ell \left( \kappa \right) }$中均匀选择$n = \kappa$条消息${m}_{1},\ldots ,{m}_{n}$。数据库$x$将有$n$行；对于$j \in  \left\lbrack  n\right\rbrack$，第$j$行是验证密钥、第$j$条消息及其有效签名，即元组$\left( {{vk},{m}_{j},\operatorname{Sign}\left( {{m}_{j},{sk}}\right) }\right)$。接下来，从$\left\lbrack  n\right\rbrack$中均匀选择$i$。为了生成第$\left( {n + 1}\right)$项${x}_{i}^{\prime }$，只需生成一个新的消息 - 签名对（使用相同的密钥${sk}$）。

难以合成的分布III：为了证明多项式（关于$\kappa$）大小的消息空间（但超多项式大小的查询集）情况下的难度，我们使用伪随机函数。粗略地说，这些是具有简短描述的多项式时间可计算函数，仅根据其输入 - 输出行为，无法有效地将它们与真正的随机函数（其描述很长）区分开来。只有当我们坚持为所有查询保持实用性时，这个结果才表明合成的难度。实际上，如果我们只关心确保平均实用性，那么第6节中描述的计数查询的基本生成器在全域$\mathcal{X}$是多项式大小时，即使$\mathcal{Q}$是指数大的，也能产生一种有效的合成算法。

设${\left\{  {f}_{s}\right\}  }_{s \in  \{ 0,1{\} }^{\kappa }}$是一个从$\left\lbrack  \ell \right\rbrack$到$\left\lbrack  \ell \right\rbrack$的伪随机函数族，其中$\ell  \in  \operatorname{poly}\left( \kappa \right)$。更具体地说，我们需要$\left\lbrack  \ell \right\rbrack$中所有元素对的集合“小”，但大于$\kappa$；这样，描述该函数族中一个函数的$\kappa$位字符串比描述一个将$\left\lbrack  \ell \right\rbrack$映射到$\left\lbrack  \ell \right\rbrack$的随机函数所需的$\ell {\log }_{2}\ell$位要短。这样的伪随机函数族可以从任何单向函数构造出来。

我们的数据全域将是$\left\lbrack  \ell \right\rbrack$中所有元素对的集合：$\mathcal{X} = \{ \left( {a,b}\right)  : a,b \in  \left\lbrack  \ell \right\rbrack  \} .{\mathcal{Q}}_{\kappa }$将包含两种类型的查询：

1. 对于该函数族中的每个函数${\left\{  {f}_{s}\right\}  }_{s \in  \{ 0,1{\} }^{\kappa }}$，都会有一个查询。全域元素$\left( {a,b}\right)  \in  \mathcal{X}$满足查询$s$当且仅当${f}_{s}\left( a\right)  = b$。

2. 将有相对较少数量（比如$\kappa$）的真正随机查询。这样的查询可以通过为每个$\left( {a,b}\right)  \in  \mathcal{X}$随机选择(a,b)是否满足该查询来构造。

难以合成的分布生成方式如下。首先，我们随机选择一个字符串 $s \in  \{ 0,1{\} }^{\kappa }$，它指定了我们函数族中的一个函数。接下来，对于从 $\left\lbrack  \ell \right\rbrack$ 中无放回随机选取的 $n = \kappa$ 个不同值 ${a}_{1},\ldots ,{a}_{n}$，我们生成宇宙元素 $\left( {a,{f}_{s}\left( a\right) }\right)$。

其直觉很简单，仅依赖于第一种类型的查询，并且不利用 ${a}_{i}$ 的独特性。给定一个根据我们的分布生成的数据库 $x$，其中伪随机函数由 $s$ 给出，合成器必须创建一个合成数据库，（几乎）其所有行都必须满足查询 $s$。直觉是它无法可靠地找到不出现在 $x$ 中的输入 - 输出对。更准确地说，对于任意元素 $a \in  \left\lbrack  \ell \right\rbrack$，使得 $x$ 中没有形式为 $\left( {a,{f}_{s}\left( a\right) }\right)$ 的行，${f}_{s}$ 的伪随机性表明，一个高效的合成器找到 ${f}_{s}\left( a\right)$ 的概率最多只比 $1/\ell$ 略大一点。从这个意义上说，伪随机性给我们带来的性质与我们从数字签名中获得的性质类似，尽管稍弱一些。

当然，对于任何给定的 $a \in  \left\lbrack  \ell \right\rbrack$，合成器确实可以以概率 $1/\ell$ 猜出值 ${f}_{s}\left( a\right)$，因此如果没有第二种类型的查询，显然没有什么能阻止它忽略 $x$，选择任意的 $a$，并输出一个包含 $n$ 个 (a, b) 副本的数据库，其中 $b$ 是从 $\left\lbrack  \ell \right\rbrack$ 中均匀随机选取的。现在的直觉是，这样的合成数据库会给出错误的比例 - 要么是零，要么是一，而真正随机查询的正确答案应该约为 $1/2 -$。

形式上，我们有：

定理 9.1。设 $f : \{ 0,1{\} }^{\kappa } \rightarrow  \{ 0,1{\} }^{\kappa }$ 是一个单向函数。对于每个 $a > 0$，以及每个整数 $n = \operatorname{poly}\left( \kappa \right)$，存在一个大小为 $\exp \left( {\operatorname{poly}\left( \kappa \right) }\right)$ 的查询族 $\mathcal{Q}$、一个大小为 $O\left( {n}^{2 + {2a}}\right)$ 的数据宇宙 $\mathcal{X}$，以及一个大小为 $n$ 的数据库上的分布，该分布对于 $\alpha  \leq$ $1/3,\beta  \leq  1/{10}$ 和 $\mu  = 1/{40}{n}^{1 + a}$ 是 $\left( {\mu ,\alpha ,\beta ,0,\mathcal{Q}}\right)$ - 难以合成的（即，对于最坏情况的查询难以合成）。

上述定理表明了使用合成数据进行数据清理的难度。然而，请注意，当查询集较小时，人们总是可以简单地为每个查询发布带噪声的计数。我们得出结论，对于小查询类（具有大数据宇宙）进行数据清理是一项将高效合成与高效概要生成（具有任意输出的数据清理）区分开来的任务。

#### 9.2.1 一般概要的难度结果

上一节的难度结果仅适用于合成器——创建合成数据库的离线机制。更通用形式的隐私保护离线机制（我们一直称之为离线查询发布机制或概要生成器）的难度与叛徒追踪方案的存在之间存在着紧密的联系。叛徒追踪方案是一种内容分发方法，在该方法中，（短）密钥字符串以某种方式分发给订阅者，使得发送者可以广播加密消息，任何订阅者都可以解密这些消息，并且由恶意订阅者联盟构建的任何有用的“盗版”解码器都可以追溯到至少一个合谋者。

一个（私钥、无状态）叛徒追踪方案由算法设置（Setup）、加密（Encrypt）、解密（Decrypt）和追踪（Trace）组成。设置算法为广播者生成一个密钥${bk}$和$N$个订阅者密钥${k}_{1},\ldots ,{k}_{N}$。加密算法使用广播者的密钥${bk}$对给定的比特进行加密。解密算法使用任何一个订阅者密钥对给定的密文进行解密。追踪算法获取密钥${bk}$并以预言机方式访问一个（盗版、无状态）解密盒，然后输出用于创建盗版盒的密钥${k}_{i}$的索引$i \in  \{ 1,\ldots ,N\}$。

叛徒追踪方案的一个重要参数是其抗合谋性：如果只要用于创建盗版解码器的密钥不超过$t$个，追踪就保证有效，那么该方案就是$t$ - 抗合谋的。当$t = N$时，即使所有订阅者联合起来试图创建一个盗版解码器，追踪仍然有效。下面是一个更完整的定义。

定义9.3。如上所述的方案（设置、加密、解密、追踪）是一个t - 抗合谋叛徒追踪方案，如果（i）它生成的密文是语义安全的（粗略地说，多项式时间算法无法区分0的加密和1的加密），并且（ii）没有多项式时间敌手$A$能以不可忽略的概率（在设置、$A$和追踪的随机硬币上）在以下游戏中“获胜”：

$A$接收用户数量$N$和一个安全参数$\kappa$，并（自适应地）请求最多$t$个用户$\left\{  {{i}_{1},\ldots ,{i}_{t}}\right\}$的密钥。然后敌手输出一个盗版解码器Dec。使用密钥${bk}$并以黑盒方式${}^{5}$访问Dec来运行追踪算法；它输出一个用户的名称$i \in  \left\lbrack  N\right\rbrack$或错误符号$\bot$。我们说敌手$A$“获胜”，如果Dec在解密密文方面有不可忽略的优势（甚至比创建一个可用的盗版解密设备的条件更弱），并且追踪的输出不在$\left\{  {{i}_{1},\ldots ,{i}_{t}}\right\}$中，这意味着敌手避免了被检测。

---

<!-- Footnote -->

${}^{5}$以黑盒方式访问一个算法意味着无法访问该算法的内部结构；只能向算法提供输入并观察其输出。

<!-- Footnote -->

---

叛徒追踪方案为何意味着计数查询发布存在难度结果的直观解释如下。固定一个叛徒追踪方案。我们必须描述那些查询发布在计算上困难的数据库和计数查询。

对于任何给定的$n = \kappa$，数据库$x \in  {\left\{  \{ 0,1{\} }^{d}\right\}  }^{n}$将包含来自$n$个合谋用户的叛逆者追踪方案的用户密钥；这里$d$是在输入${1}^{\kappa }$上运行设置算法时获得的解密密钥的长度。查询族${\mathcal{Q}}_{\kappa }$将针对每个可能的密文$c$有一个查询${q}_{c}$，询问“对于多少比例的行$i \in  \left\lbrack  n\right\rbrack$，密文$c$在第$i$行的密钥下解密为1？” 请注意，由于每个用户都可以解密，如果发送者分发比特1的加密$c$，答案将是1：所有行都将$c$解密为1，因此这样的行的比例为1。相反，如果发送者分发比特0的加密${c}^{\prime }$，答案将是0：因为没有行将${c}^{\prime }$解密为1，所以将${c}^{\prime }$解密为1的行的比例为0。因此，对于查询${q}_{c}$（其中$c$是1比特消息$b$的加密）的准确答案就是$b$本身。

现在，假设存在一种针对$\mathcal{Q}$中的查询的高效离线差分隐私查询发布机制。合谋者可以使用该算法高效地生成数据库的概要，使数据分析师能够高效地计算查询${q}_{c}$的近似答案。如果这些近似值并非无意义，那么分析师可以使用它们进行正确解密。也就是说，合谋者可以利用这一点来制造一个盗版解码器盒。但叛逆者追踪确保了，对于任何这样的盒子，追踪算法可以恢复至少一个用户的密钥，即数据库的一行。这违反了差分隐私，与存在一种用于发布$\mathcal{Q}$的高效差分隐私算法的假设相矛盾。

这一方向已被用于排除针对特定类别的${2}^{\widetilde{O}\left( \sqrt{n}\right) }$计数查询的高效离线清理器的存在；这可以扩展到排除针对从第二个（大）类中自适应抽取的$\widetilde{\Theta }\left( {n}^{2}\right)$计数查询的高效在线清理器的存在。

计数查询的离线查询发布困难意味着叛逆者追踪的直觉在于，未能保护隐私会立即产生某种形式的可追踪性；也就是说，在为一组行（解密密钥）提供（近似）功能等价物的同时保护每一行（解密密钥）的隐私的难度——即制造一个不可追踪的解码器的难度——正是我们在叛逆者追踪方案中所寻求的。

更详细地说，给定一个难以清理的数据库分布和计数查询族，随机抽取的$n$项数据库可以充当“主密钥”，其中用于解密消息的秘密是该数据库上随机查询的计数。对于随机选择的多对数(n)个查询的子集$S$，从数据库中随机抽取的多对数(n)行的集合（很可能）能很好地近似$S$中的所有查询。因此，可以通过将数据库随机划分为$n/$个多对数(n)行的多对数(n)集合，并将每个集合分配给不同的用户来获得各个用户的密钥。这些集合足够大，以至于在压倒性概率下，它们在例如多对数(n)个随机查询集合上的计数都接近原始数据库的计数。

为了完成这个论证，我们设计了一种加密方案，其中解密等同于计算小的随机查询集合上的近似计数。由于根据定义，盗版解密盒可以进行解密，因此盗版盒可以用于计算近似计数。如果我们将这个盒子视为数据库的清理结果，我们可以得出结论（因为清理是困难的），解密盒可以“追溯”到用于创建它的密钥（数据库项）。

### 9.3 多项式时间敌手

定义9.4（计算差分隐私）。当且仅当对于所有仅相差一行的数据库 $x,y$，以及所有非均匀多项式（关于 $\kappa$）算法 $T$，随机算法 ${C}_{\kappa } : {\mathcal{X}}^{n} \rightarrow  Y$ 是 $\varepsilon$ -计算差分隐私的。

$$
\Pr \left\lbrack  {T\left( {{C}_{\kappa }\left( x\right) }\right)  = 1}\right\rbrack   \leq  {e}^{\varepsilon }\Pr \left\lbrack  {T\left( {{C}_{\kappa }\left( y\right) }\right)  = 1}\right\rbrack   + \nu \left( \kappa \right) ,
$$

其中 $\nu \left( \cdot \right)$ 是任何增长速度比任何多项式的倒数都慢的函数，并且算法 ${C}_{\kappa }$ 在 $n$、$\log \left| \mathcal{X}\right|$ 和 $\kappa$ 的多项式时间内运行。

直观地说，这意味着如果对手被限制在多项式时间内，那么计算差分隐私机制提供的隐私程度与 $\left( {\varepsilon ,\nu \left( \kappa \right) }\right)$ -差分隐私算法相同。一般来说，消除 $\nu \left( \kappa \right)$ 项是没有希望的；例如，当涉及加密时，总是有一些（极小的）机会猜出解密密钥。

一旦我们假设对手被限制在多项式时间内，我们就可以使用安全多方计算的强大技术来提供分布式在线查询发布算法，用模拟可信策展人的分布式协议取代可信服务器。因此，例如，一组医院，每家医院都持有许多患者的数据，可以协作对其患者的联合数据进行统计分析，同时确保每个患者的差分隐私。一个更激进的影响是，个人可以维护自己的数据，选择参与或不参与每个特定的统计查询或研究，同时确保自己数据的差分隐私。

我们已经看到了一种分布式解决方案，至少对于计算 $n$ 位之和的问题：随机响应。这种解决方案不需要计算假设，并且预期误差为 $\Theta \left( \sqrt{n}\right)$。相比之下，使用密码学假设允许进行更准确和广泛的分析，因为通过模拟策展人，它可以运行拉普拉斯机制的分布式实现，该机制具有恒定的预期误差。

这就引出了一个自然的问题，即是否存在某种不依赖于密码学假设的其他方法，在分布式环境中比随机响应具有更高的准确性。或者更一般地说，计算差分隐私所能实现的与“传统”差分隐私所能实现的之间是否存在差异？也就是说，密码学是否确实为我们带来了一些好处？

在多方环境中，答案是肯定的。仍然将我们的注意力限制在对 $n$ 位求和上，我们有：

定理9.2。对于 $\varepsilon  < 1$，每个用于计算 $n$ 位（每方一位）之和的 $n$ -方 $\left( {\varepsilon ,0}\right)$ -差分隐私协议在高概率下会产生误差 $\Omega \left( {n}^{1/2}\right)$。

如果 $\delta  \in$ $o\left( {1/n}\right)$，对于 $\left( {\varepsilon ,\delta }\right)$ -差分隐私也有类似的定理成立。

证明。（概要）设 ${X}_{1},\ldots ,{X}_{n}$ 是均匀独立的位。协议的记录 $T$ 是一个随机变量 $T = T\left( {{P}_{1}\left( {X}_{1}\right) ,\ldots ,}\right.$ ${P}_{n}\left( {X}_{n}\right)$，其中对于 $i \in  \left\lbrack  n\right\rbrack$，玩家 $i$ 的协议表示为 ${P}_{i}$。在 $T = t$ 的条件下，位 ${X}_{1},\ldots ,{X}_{n}$ 仍然是独立的位，每个位的偏差为 $O\left( \varepsilon \right)$。此外，通过差分隐私、${X}_{i}$ 的均匀性和贝叶斯定律，我们有：

$$
\frac{\Pr \left\lbrack  {{X}_{i} = 1 \mid  T = t}\right\rbrack  }{\Pr \left\lbrack  {{X}_{i} = 0 \mid  T = t}\right\rbrack  } = \frac{\Pr \left\lbrack  {T = t \mid  {X}_{i} = 1}\right\rbrack  }{\Pr \left\lbrack  {T = t \mid  {X}_{i} = 0}\right\rbrack  } \leq  {e}^{\varepsilon } < 1 + {2\varepsilon }.
$$

为完成证明，我们注意到 $n$ 个独立比特（每个比特都有恒定偏差）的和，以很高的概率落在任何大小为 $o\left( \sqrt{n}\right)$ 的区间之外。因此，以很高的概率，和 $\mathop{\sum }\limits_{i}{X}_{i}$ 不在区间 $\left\lbrack  {\operatorname{output}\left( \mathrm{T}\right)  - o\left( {n}^{1/2}\right) ,\operatorname{output}\left( \mathrm{T}\right)  + o\left( {n}^{1/2}\right) }\right\rbrack$ 内。

一个更复杂的证明表明，即使在两方的情况下，计算差分隐私（computational differential privacy）和普通差分隐私（ordinary differential privacy）之间也存在差异。在可信策展人（trusted curator）的情况下，计算假设是否能为我们带来任何好处，这是一个引人入胜的开放性问题。初步结果是否定的：对于少量实值查询，即查询数量不随安全参数增长的情况，存在一类自然的效用度量，包括 ${L}_{p}$ 距离和均方误差，对于这些度量，任何计算上私密的机制都可以转换为一个统计上私密的机制，该机制大致同样高效，并且能实现几乎相同的效用。

### 9.4 参考文献注释

多项式时间有界策展人的负面结果以及与叛徒追踪（traitor tracing）的联系归功于 Dwork 等人 [28]。Ullman [82] 进一步研究了与叛徒追踪的联系，他表明，假设单向函数存在，以差分隐私回答 ${n}^{2 + o\left( 1\right) }$ 个任意线性查询在计算上是困难的（即使在不考虑隐私的情况下，答案很容易计算）。在《我们的数据，我们自己》（“Our Data, Ourselves”）中，Dwork、Kenthapadi、McSherry、Mironov 和 Naor 使用安全函数评估技术代替可信策展人，考虑了差分隐私前身的分布式版本 [21]。[64] 中开始了对计算差分隐私的正式研究，定理 9.2 中多方和单策展人情况下 $\left( {\varepsilon ,0}\right)$ -差分隐私所能达到的准确性之间的差异归功于 McGregor 等人 [58]。关于在可信策展人情况下，对对手的计算假设是否能带来任何好处的初步结果归功于 Groce 等人 [37]。

从任何单向函数构造伪随机函数（pseudorandom functions）归功于 Håstad 等人 [40]。

## 10 差分隐私与机制设计

博弈论中最引人入胜的领域之一是机制设计（mechanism design），它是一门设计激励措施以促使人们按你期望的方式行事的科学。差分隐私已被证明在几个意想不到的方面与机制设计有着有趣的联系。它提供了一种量化和控制隐私损失的工具，如果机制设计者试图操纵的人关心隐私，这一点很重要。然而，它还提供了一种限制机制结果对任何单个个体选择的敏感性的方法，事实证明，即使在没有隐私问题的情况下，这也是一种强大的工具。在本节中，我们简要概述其中一些观点。

机制设计是指当算法的输入由个体的、自利的参与者控制，而不是由算法设计者自己控制时的算法设计问题。该算法将其接收到的输入映射到某个结果，参与者对这些结果有偏好。困难在于，如果参与者误报数据能使算法输出一个不同的、更偏好的结果，他们可能会这样做，因此机制设计者必须设计算法，使参与者始终有动机报告他们的真实数据。

机制设计的关注点与隐私算法设计的关注点非常相似。在这两种情况下，算法的输入都被认为属于某个对结果有偏好的第三方 ${}^{1}$。在机制设计中，我们通常认为个体从机制的结果中获得一些明确的价值。在隐私算法设计中，我们通常认为个体因机制结果（的后果）而遭受一些明确的损害。实际上，我们可以给出一个与标准定义等价的差分隐私的效用理论定义，但它明确了与个体效用的联系：

定义10.1。如果对于每个函数 $f : R \rightarrow  {\mathbb{R}}_{ + }$ ，以及每对相邻数据库 $x,y \in  {\mathbb{N}}^{\left| \mathcal{X}\right| }$ ，算法 $A : {\mathbb{N}}^{\left| \mathcal{X}\right| } \rightarrow  R$ 满足 $\epsilon$ -差分隐私：

$$
\exp \left( {-\epsilon }\right) {\mathbb{E}}_{z \sim  A\left( y\right) }\left\lbrack  {f\left( z\right) }\right\rbrack   \leq  {\mathbb{E}}_{z \sim  A\left( x\right) }\left\lbrack  {f\left( z\right) }\right\rbrack   \leq  \exp \left( \epsilon \right) {\mathbb{E}}_{z \sim  A\left( y\right) }\left\lbrack  {f\left( z\right) }\right\rbrack  .
$$

我们可以将 $f$ 视为一个将结果映射到任意主体对这些结果的效用的函数。根据这种解释，如果一个机制对于每个主体都承诺，无论其效用函数是什么，他们参与该机制对其预期未来效用的影响不会超过 $\exp \left( \epsilon \right)$ 倍，那么该机制就是 $\epsilon$ -差分隐私的。

现在让我们简要定义一下机制设计中的一个问题。一个机制设计问题由几个要素定义。有 $n$ 个主体 $i \in  \left\lbrack  n\right\rbrack$ ，以及一组结果 $\mathcal{O}$ 。每个主体都有一个类型 ${t}_{i} \in  \mathcal{T}$ ，该类型只有她自己知道，并且存在一个关于结果的效用函数 $u : \mathcal{T} \times  \mathcal{O} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 。主体 $i$ 从结果 $o \in  \mathcal{O}$ 中获得的效用是 $u\left( {{t}_{i},o}\right)$ ，我们通常将其缩写为 ${u}_{i}\left( o\right)$ 。我们将用 $t \in  {\mathcal{T}}^{n}$ 表示所有 $n$ 个主体类型的向量，其中 ${t}_{i}$ 表示主体 $i$ 的类型， ${t}_{-i} \equiv  \left( {{t}_{1},\ldots ,{t}_{i - 1},{t}_{i + 1},\ldots ,{t}_{n}}\right)$ 表示除主体 $i$ 之外所有主体的类型向量。主体 $i$ 的类型完全决定了她对结果的效用——也就是说，两个主体 $i \neq  j$ ，如果 ${t}_{i} = {t}_{j}$ ，那么他们对每个结果的评估将相同：对于所有的 $o \in  \mathcal{O}$ ，都有 ${u}_{i}\left( o\right)  = {u}_{j}\left( o\right)$ 。

---

<!-- Footnote -->

${}^{1}$ 在隐私设置中，数据库管理员（如医院）可能已经可以访问数据本身，但在努力保护隐私时，仍然会采取行动来保护数据所有者的利益。

<!-- Footnote -->

---

机制 $M$ 以一组报告的类型作为输入，每个参与者提供一个类型，并选择一个结果。也就是说，机制是一个映射 $M : {\mathcal{T}}^{n} \rightarrow  \mathcal{O}$ 。参与者会策略性地选择报告他们的类型，以优化自己的效用，可能会考虑（他们认为）其他参与者会怎么做。特别是，他们不必向机制报告自己的真实类型。如果无论对手报告什么类型，一个参与者总是有动机报告某个类型，那么报告该类型就被称为占优策略。如果对于每个参与者来说，报告自己的真实类型是占优策略，那么该机制就被称为诚实的，或者等价地，占优策略诚实的。

定义10.2。给定一个机制 $M : {\mathcal{T}}^{n} \rightarrow  \mathcal{O}$ ，如果对于每对类型 ${t}_{i},{t}_{i}^{\prime } \in  T$ ，以及每个类型向量 ${t}_{-i}$ ，如实报告是参与者 $i$ 的 $\epsilon$ -近似占优策略：

$$
u\left( {{t}_{i},M\left( {{t}_{i},{t}_{-i}}\right) }\right)  \geq  u\left( {{t}_{i},M\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  - \epsilon .
$$

如果如实报告是每个参与者的 $\epsilon$ -近似占优策略，我们就说 $M$ 是 $\epsilon$ -近似占优策略诚实的。如果 $\epsilon  = 0$ ，那么 $M$ 就是完全诚实的。

也就是说，如果无论其他参与者报告什么，没有主体可以通过歪曲自己的类型来提高自己的效用，那么该机制就是诚实的。

在这里，我们可以立即观察到与差分隐私定义的句法联系。我们可以将类型空间$T$与数据全域$X$等同起来。因此，该机制的输入由一个大小为$n$的数据库组成，该数据库包含每个参与者的报告。事实上，当一个参与者考虑她是应该如实报告自己的类型${t}_{i}$还是说谎并误报为类型${t}_{i}^{\prime }$时，她是在决定该机制应该接收两个数据库中的哪一个：$\left( {{t}_{1},\ldots ,{t}_{n}}\right)$或$\left( {{t}_{1},\ldots ,{t}_{i - 1},{t}_{i}^{\prime },{t}_{i + 1},\ldots ,{t}_{n}}\right)$。请注意，这两个数据库仅在参与者$i$的报告上有所不同！也就是说，它们是相邻数据库。因此，差分隐私提供了近似真实性的保证！

### 10.1 差分隐私作为一种解决方案概念

研究差分隐私与博弈论之间联系的起点之一是观察到差分隐私是比近似真实性更强的条件。注意到对于$\epsilon  \leq  1,\exp \left( \epsilon \right)  \leq  1 + {2\epsilon }$，因此以下命题是显而易见的。

命题10.1。如果一个机制$M$是$\epsilon$ - 差分隐私的，那么$M$也是${2\epsilon }$ - 近似占优策略真实的。

作为一种解决方案概念，它具有一些策略防操纵机制所不具备的鲁棒性属性。根据差分隐私的组合性质，${2\epsilon }$ - 差分隐私机制的组合仍然是${4\epsilon }$ - 近似占优策略真实的。相比之下，一般策略防操纵机制的激励属性在组合下可能无法保留。

差分隐私作为一种解决方案概念的另一个有用属性是它可以推广到群体隐私：假设$t$和${t}^{\prime } \in$ ${\mathcal{T}}^{n}$不是相邻的，而是在$k$个索引上有所不同。回想一下，根据群体隐私，对于任何参与者$i : {\mathbb{E}}_{o \sim  M\left( t\right) }\left\lbrack  {{u}_{i}\left( o\right) }\right\rbrack   \leq$ $\exp \left( {k\epsilon }\right) {\mathbb{E}}_{o \sim  M\left( {t}^{\prime }\right) }\left\lbrack  {{u}_{i}\left( o\right) }\right\rbrack$。也就是说，当$k \ll  1/\epsilon$时，最多$k$种类型的变化最多使期望输出改变$\approx  \left( {1 + {k\epsilon }}\right)$。因此，差分隐私机制使如实报告成为即使对于$k$个参与者的联盟也是${2k\epsilon }$ - 近似占优策略——即，差分隐私自动提供了对合谋的鲁棒性。同样，这与一般的占优策略真实机制形成对比，一般的占优策略真实机制通常不提供防止合谋的保证。

值得注意的是，差分隐私在非常一般的设置中无需使用货币就能实现这些属性！相比之下，在不允许货币转移的情况下，精确占优策略真实机制的集合非常有限。

最后，我们指出将差分隐私作为一种解决方案概念存在的一个缺点：如实报告自己的类型不仅是一种近似占优策略，任何报告都是近似占优策略！也就是说，差分隐私使得结果近似独立于任何单个参与者的报告。在某些情况下，这个缺点可以得到缓解。例如，假设$M$是一个差分隐私机制，但参与者的效用函数被定义为机制结果和参与者报告类型${t}_{i}^{\prime }$的函数：形式上，我们将结果空间视为${\mathcal{O}}^{\prime } = \mathcal{O} \times  T$。当参与者向机制报告类型${t}_{i}^{\prime }$，并且机制选择结果$o \in  \mathcal{O}$时，参与者所体验到的效用由结果${o}^{\prime } = \left( {o,{t}_{i}^{\prime }}\right)$控制。现在考虑底层效用函数$u : T \times  {\mathcal{O}}^{\prime } \rightarrow  \left\lbrack  {0,1}\right\rbrack$。假设我们固定机制的一个选择$o$，如实报告是一种占优策略——也就是说，对于所有类型${t}_{i},{t}_{i}^{\prime }$，以及所有结果$o \in  \mathcal{O}$：

$$
u\left( {{t}_{i},\left( {o,{t}_{i}}\right) }\right)  \geq  u\left( {{t}_{i},\left( {o,{t}_{i}^{\prime }}\right) }\right) .
$$

那么，向一个$\epsilon$ - 差分隐私机制$M : {T}^{n} \rightarrow  \mathcal{O}$如实报告仍然是一种${2\epsilon }$近似占优策略，因为对于参与者$i$可能考虑的任何虚假报告${t}_{i}^{\prime }$，我们有：

$$
u\left( {{t}_{i},\left( {M\left( t\right) ,{t}_{i}}\right) }\right)  = {\mathbb{E}}_{o \sim  M\left( t\right) }\left\lbrack  {u\left( {{t}_{i},\left( {o,{t}_{i}}\right) }\right) }\right\rbrack  
$$

$$
 \geq  \left( {1 + {2\epsilon }}\right) {\mathbb{E}}_{o \sim  M\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\left\lbrack  {u\left( {{t}_{i},\left( {o,{t}_{i}}\right) }\right) }\right\rbrack  
$$

$$
 \geq  {\mathbb{E}}_{o \sim  M\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\left\lbrack  {u\left( {{t}_{i},\left( {o,{t}_{i}^{\prime }}\right) }\right) }\right\rbrack  
$$

$$
 = u\left( {{t}_{i},\left( {M\left( {{t}_{i}^{\prime },{t}_{-i}}\right) ,{t}_{i}^{\prime }}\right) }\right) \text{.}
$$

然而，我们不再有“每个报告都是近似占优策略”这一情况，因为参与者$i$的效用可以任意依赖于${o}^{\prime } = \left( {o,{t}_{i}^{\prime }}\right)$，并且只有$o$（而不是参与者$i$的报告${t}_{i}^{\prime }$本身）是差分隐私的。我们在这里考虑的所有例子都会是这种情况。

### 10.2 差分隐私作为机制设计的工具

在本节中，我们将展示如何将差分隐私的方法作为一种工具来设计新颖的机制。

#### 10.2.1 热身：数字商品拍卖

作为热身，让我们考虑差分隐私在机制设计中首次应用的一个简单特殊情况。考虑一场数字商品拍卖，即卖家拥有一种商品的无限供应，且生产的边际成本为零，例如一款软件或其他数字媒体。有$n$个对该商品有单位需求的买家，每个买家的估值${v}_{i} \in  \left\lbrack  {0,1}\right\rbrack$未知。通俗地说，投标人$i$的估值${v}_{i}$代表买家$i$愿意为该商品支付的最高金额。投标人的估值没有先验分布，因此一个自然的收入基准是最优固定价格的收入。在价格$p \in  \left\lbrack  {0,1}\right\rbrack$下，每个${v}_{i} \geq  p$的投标人$i$都会购买。因此，拍卖人的总收入为

$$
\operatorname{Rev}\left( {p,v}\right)  = p \cdot  \left| \left\{  {i : {v}_{i} \geq  p}\right\}  \right| .
$$

最优收入是最优固定价格的收入：$\mathrm{{OPT}} =$ $\mathop{\max }\limits_{p}\operatorname{Rev}\left( {p,v}\right)$。这种情况已经得到了深入研究：对于精确占优策略如实机制，已知的最佳结果是一个能实现至少$\mathrm{{OPT}} - O\left( \sqrt{n}\right)$收入的机制。

我们展示了指数机制的一个简单应用如何实现至少$\mathrm{{OPT}} - O\left( \frac{\log n}{\epsilon }\right)$的收入。也就是说，该机制用精确性换取近似如实性，但实现了指数级更好的收入保证。当然，它也继承了前面讨论过的差分隐私的优点，如抗合谋性和可组合性。

思路是从指数机制中选择一个价格，将该价格所能获得的收入作为我们的“质量得分”。假设我们将指数机制的取值范围设定为$\mathcal{R} = \{ \alpha ,{2\alpha },\ldots ,1\}$。该范围的大小为$\left| \mathcal{R}\right|  = 1/\alpha$。如果我们将价格选择范围限制在$\mathcal{R}$内，我们在潜在收入方面损失了多少呢？不难看出

$$
{\mathrm{{OPT}}}_{\mathcal{R}} \equiv  \mathop{\max }\limits_{{p \in  \mathcal{R}}}\operatorname{Rev}\left( {p,v}\right)  \geq  \mathrm{{OPT}} - {\alpha n}.
$$

这是因为，如果${p}^{ * }$是实现最优收入的价格，而我们使用的价格为$p$，且满足${p}^{ * } - \alpha  \leq  p \leq  {p}^{ * }$，那么在最优价格下购买的每个买家都会继续购买，并且每个买家给我们带来的收入最多减少$\alpha$。由于最多有$n$个买家，因此总损失收入最多为${\alpha n}$。

那么我们如何对指数机制进行参数化呢？我们有一族离散范围$\mathcal{R}$，由$\alpha$参数化。对于一个值向量$v$和一个价格$p \in  \mathcal{R}$，我们将质量函数定义为$q\left( {v,p}\right)  =$ $\operatorname{Rev}\left( {v,p}\right)$。注意，由于每个值${v}_{i} \in  \left\lbrack  {0,1}\right\rbrack$，我们可以将注意力限制在价格$p \leq  1$上，因此，$q$的敏感度为$\Delta  = 1$：改变一个投标者的估值最多只能使固定价格下的收益改变${v}_{i} \leq  1$。因此，如果我们要求$\epsilon$ - 差分隐私，根据定理3.11，我们可以得到，在高概率下，指数机制会返回某个价格$p$，使得

$$
\operatorname{Rev}\left( {p,v}\right)  \geq  \left( {\mathrm{{OPT}} - {\alpha n}}\right)  - O\left( {\frac{1}{\epsilon }\ln \left( \frac{1}{\alpha }\right) }\right) .
$$

选择我们的离散化参数$\alpha$来最小化两种误差来源，我们发现这个机制在高概率下能为我们找到一个实现收益的价格

$$
\operatorname{Rev}\left( {p,v}\right)  \geq  \mathrm{{OPT}} - O\left( \frac{\log n}{\epsilon }\right) .
$$

隐私参数$\epsilon$应该选择什么合适的水平呢？请注意，在这里，我们并不一定将隐私本身视为计算的目标。相反，$\epsilon$是一种在收益保证和代理人偏离激励上限之间进行权衡的方式。在经济学中关于大市场的文献里，当无法实现精确的真实性时，一个常见的目标是“渐近真实性”——也就是说，随着市场规模$n$的增大，任何代理人偏离其真实报告的最大激励趋于0。为了在这里实现类似的结果，我们所需要做的就是将$\epsilon$设为代理人数量$n$的某个递减函数。例如，如果我们取$\epsilon  = 1/\log \left( n\right)$，那么我们就得到了一个渐近精确真实的机制（即，随着市场规模增大，对真实性的近似变得精确）。我们还可以问，随着$n$的增大，我们对最优收益的近似程度如何。请注意，我们对最优收益的近似只是加法性的，因此，即使在这样设置$\epsilon$的情况下，只要$\mathrm{{OPT}}$随着人口规模$n$的增长比$\log {\left( n\right) }^{2}$增长得更快，我们仍然可以保证至少有$\left( {1 - o\left( 1\right) }\right) \mathrm{{OPT}}$的收益。

最后，注意我们可以使每个代理人$i$的报告值${v}_{i}$具有约束力。换句话说，只要${v}_{i} \geq  p$，我们就可以将一个物品分配给代理人$i$，并按照所选的张贴价格$p$收取费用。如果我们这样做，该机制是近似真实的，因为价格是使用差分隐私机制选取的。此外，并非每个报告都是近似占优策略：如果一个代理人高报，她可能会被迫以高于其真实价值的价格购买商品。

#### 10.2.2 近似真实的均衡选择机制

我们现在考虑近似真实的均衡选择问题。我们回顾一下纳什均衡的定义：假设每个玩家都有一组行动$\mathcal{A}$，并且可以选择执行任何行动${a}_{i} \in  \mathcal{A}$。此外，假设结果仅仅是代理人可能选择执行的行动选择，因此代理人的效用函数定义为$u : \mathcal{T} \times  {\mathcal{A}}^{n} \rightarrow  \left\lbrack  {0,1}\right\rbrack$。那么：

定义10.3。一组行动$a \in  {\mathcal{A}}^{n}$是一个$\epsilon$ - 近似纳什均衡，如果对于所有玩家$i$和所有行动${a}_{i}^{\prime }$：

$$
{u}_{i}\left( a\right)  \geq  {u}_{i}\left( {{a}_{i}^{\prime },{a}_{-i}}\right)  - \epsilon 
$$

换句话说，假设其他代理人按照$a$行动，那么每个代理人同时都在对其他代理人的行为做出（近似）最优反应。

大致来说，问题如下：假设我们有一个博弈，其中每个玩家都知道自己的收益，但不知道其他玩家的收益（即玩家不知道其他参与者的类型）。因此，玩家们不知道这个博弈的均衡结构。即使他们知道，也可能存在多个均衡，不同的参与者偏好不同的均衡。中介提供的机制能否激励参与者如实报告他们的效用并遵循其选择的均衡呢？

例如，想象一个城市，其中（比如说）谷歌导航是占主导地位的服务。每天早上，每个人输入他们的起点和目的地，收到一组路线指引，并根据这些指引选择自己的路线。是否有可能设计一种导航服务，使得：每个参与者都受到激励，既（1）如实报告，又（2）遵循所提供的驾驶路线？虚报起点和终点，以及如实报告起点和终点但随后选择不同（更短）的路线都应受到抑制。

直观地说，我们的两个期望目标存在冲突。在上述通勤示例中，如果我们要保证每个玩家都受到激励如实遵循建议的路线，那么我们必须根据玩家的报告计算所讨论博弈的一个均衡。另一方面，要做到这一点，我们给某个玩家$i$的建议路线必须依赖于其他玩家报告的位置/目的地对。这种矛盾在激励方面会带来问题：如果我们根据玩家的报告计算博弈的一个均衡，那么一个参与者可能会通过虚报而获益，导致我们计算出错误博弈的均衡。

然而，如果参与者$i$的报告对参与者$j \neq  i$的行动只有微小的影响，那么这个问题将在很大程度上得到缓解。在这种情况下，参与者$i$很难通过对其他玩家的影响获得优势。然后，假设每个人都如实报告了他们的类型，该机制将计算出正确博弈的一个均衡，根据定义，每个参与者$i$遵循建议的均衡行动就是最优选择。换句话说，如果我们能在差分隐私的约束下计算出博弈的一个近似均衡，那么如实报告，然后采取协调设备建议的行动将是一个纳什均衡。稍加思考就会发现，在小型博弈中，私下计算均衡的目标是不可能实现的，因为在小型博弈中，一个参与者的效用是其他参与者行动（以及效用函数）的高度敏感函数。但在大型博弈中情况如何呢？

形式上，假设我们有一个$n$个玩家的博弈，其行动集为$\mathcal{A}$，每个类型为${t}_{i}$的参与者都有一个效用函数${u}_{i} : {\mathcal{A}}^{n} \rightarrow  \left\lbrack  {0,1}\right\rbrack$。我们称这个博弈是$\Delta$ - 大型的，如果对于所有玩家$i \neq  j$、行动向量$a \in  {\mathcal{A}}^{n}$和行动对${a}_{j},{a}_{j}^{\prime } \in  \mathcal{A}$：

$$
\left| {{u}_{i}\left( {{a}_{j},{a}_{-j}}\right)  - {u}_{i}\left( {{a}_{j}^{\prime },{a}_{-j}}\right) }\right|  \leq  \Delta .
$$

换句话说，如果某个参与者$j$单方面改变他的行动，那么他对任何其他参与者$i \neq  j$的收益的影响至多为$\Delta$。请注意，如果参与者$j$改变他自己的行动，那么他的收益可能会任意改变。从这个意义上说，许多博弈都是“大型”的。在上述通勤示例中，如果爱丽丝改变她的上班路线，她可能会大幅增加或减少自己的通勤时间，但对任何其他参与者鲍勃的通勤时间只会有极小的影响。本节的结果在$\Delta  = O\left( {1/n}\right)$的情况下最强，但更普遍地也成立。

首先，我们可能会问，我们是否根本就需要隐私——在一个大规模博弈中，任何根据所报告的类型计算博弈均衡的算法，是否都具有我们所期望的稳定性呢？答案是否定的。举一个简单的例子，假设有$n$个人，每个人都必须选择去海滩（B）还是去山区（M）。人们私下了解自己的类型——每个人的效用取决于他自己的类型、他的行动，以及去海滩的其他人的比例$p$。海滩型的人如果去海滩，会得到${10p}$的收益；如果去山区，则会得到$5\left( {1 - p}\right)$的收益。山区型的人去海滩会得到${5p}$的收益，去山区会得到${10}\left( {1 - p}\right)$的收益。请注意，这是一个大规模（即低敏感度）的博弈——每个玩家的收益对其他玩家的行动不敏感。此外，无论类型的实际情况如何，“每个人都去海滩”和“每个人都去山区”都是该博弈的均衡。考虑这样一种机制，它试图实现以下社会选择规则——“如果海滩型的人数少于总人数的一半，就把所有人送到海滩，反之亦然”。显然，如果山区型的人占多数，那么每个山区型的人都有动机谎报为海滩型；反之亦然。因此，即使这个博弈是“大规模”的，并且参与者的行动不会显著影响其他参与者的收益，但仅仅根据所报告的类型配置计算均衡，一般来说并不能产生近似真实的机制。

然而，事实证明，可以给出一种具有以下性质的机制：它获取每个参与者的类型${t}_{i}$，然后计算由所报告的类型定义的博弈的$\alpha$ - 近似相关均衡（在某些情况下，可以将这一结果加强为计算基础博弈的近似纳什均衡）。它从相关均衡中抽取一个行动配置$a \in  {\mathcal{A}}^{n}$，并向每个参与者$i$报告行动${a}_{i}$。该算法保证，对于所有参与者$i$，除$i$之外的所有参与者的报告的联合分布${a}_{-i}$在参与者$i$所报告的类型上是差分隐私的。当算法计算基础博弈的相关均衡时，这一保证足以实现一种受限形式的近似真实性：有选择加入或退出该机制（但如果选择加入则不能谎报其类型）的参与者没有退出的动机，因为没有参与者$i$可以通过退出显著改变其他参与者的行动分布。此外，鉴于他选择加入，没有参与者有动机不遵循他所得到的建议行动，因为他的建议是相关均衡的一部分。当该机制计算基础博弈的纳什均衡时，那么即使参与者在选择加入时能够向机制谎报其类型，该机制也是真实的。

---

<!-- Footnote -->

${}^{2}$相关均衡是由行动配置的联合分布${\mathcal{A}}^{n}$定义的。对于从该分布中抽取的一个行动配置$a$，如果只告诉参与者$i$行动${a}_{i}$，那么在给定${a}_{-i}$的条件分布下，采取行动${a}_{i}$是最优反应。$\alpha$ - 近似相关均衡是指偏离该均衡最多能使参与者的效用提高$\alpha$。

<!-- Footnote -->

---

更具体地说，当这些机制在满足$\epsilon$ - 差分隐私的同时计算出一个$\alpha$ - 近似纳什均衡时，每个遵循诚实行为（即，首先选择参与并报告其真实类型，然后遵循建议的行动）的参与者都会形成一个$\left( {{2\epsilon } + \alpha }\right)$ - 近似纳什均衡。这是因为，从隐私角度来看，报告你的真实类型是一种${2\epsilon }$ - 近似占优策略，并且假设每个人都报告了他们的真实类型，该机制会计算出真实博弈的一个$\alpha$ - 近似均衡，因此根据定义，遵循建议的行动是一种$\alpha$ - 近似最优反应。存在一些机制可用于计算具有$\alpha  = O\left( \frac{1}{\sqrt{n}\epsilon }\right)$ 的大型博弈中的$\alpha$ - 近似均衡。因此，通过设置$\epsilon  = O\left( \frac{1}{{n}^{1/4}}\right)$ ，这为……提供了一种$\eta$ - 近似真实的均衡选择机制

$$
\eta  = {2\epsilon } + \alpha  = O\left( \frac{1}{{n}^{1/4}}\right) .
$$

换句话说，它为大型博弈中的均衡行为协调提供了一种机制，该机制在博弈规模上渐近真实，而且无需货币转移。

#### 10.2.3 实现精确真实性

到目前为止，我们已经讨论了在大规模群体博弈中渐近真实的机制。然而，如果我们坚持要使用精确占优策略真实的机制，同时保留我们目前所讨论机制的一些优良特性，例如，这些机制不需要能够提取货币支付，该怎么办呢？差分隐私能在此发挥作用吗？答案是肯定的——在本节中，我们将讨论一个框架，该框架将差分隐私机制作为构建模块，用于设计无需货币的精确真实机制。

基本思路简单而优雅。正如我们所见，指数机制在保留差分隐私的同时，通常能提供出色的效用保证。这虽然不能产生一个精确真实的机制，但它让每个参与者几乎没有动机偏离真实行为。如果我们能将其与第二种机制结合起来会怎样呢？第二种机制不一定需要有良好的效用保证，但能给每个参与者一个严格的正向激励来如实报告，即一种本质上只惩罚非真实行为的机制。然后，我们可以在运行这两种机制之间进行随机选择。如果我们对惩罚机制赋予足够的权重，那么我们就能继承其严格真实性的特性。分配给指数机制的剩余权重则有助于最终机制的效用特性。我们希望，由于指数机制一开始就近似策略无懈可击，随机化机制可以对严格真实的惩罚机制赋予较小的权重，从而具有良好的效用特性。

为了设计惩罚机制，我们必须在一个稍微非标准的环境中进行工作。我们可以将机制建模为不仅选择一个结果，然后让参与者选择对该结果的反应，这两者共同决定了他的效用，而不是简单地选择一个结果。然后，机制将有权根据参与者报告的类型限制其允许的反应。形式上，我们将在以下框架中进行工作：

定义10.4（环境）。一个环境是一个包含$n$ 个参与者的集合$N$ 、一个类型集合${t}_{i} \in  \mathcal{T}$ 、一个有限的结果集合$\mathcal{O}$ 、一个反应集合$R$ 和一个效用函数$u : T \times  \mathcal{O} \times  R \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 。

我们用${r}_{i}\left( {t,s,{\widehat{R}}_{i}}\right)  \in  \arg \mathop{\max }\limits_{{r \in  {\widehat{R}}_{i}}}{u}_{i}\left( {t,s,r}\right)$ 来表示，如果$i$ 属于类型$t$ ，他在对备选方案$s$ 的选择${\widehat{R}}_{i} \subseteq  R$ 中的最优反应。

一个直接揭示机制$\mathcal{M}$ 定义了一个按如下方式进行的博弈：

1. 每个参与者$i$ 报告一个类型${t}_{i}^{\prime } \in  \mathcal{T}$ 。

2. 该机制为每个参与者$i$ 选择一个备选方案$s \in  \mathcal{O}$ 和一个反应子集${\widehat{R}}_{i} \subseteq  R$ 。

3. 每个参与者 $i$ 选择一个反应 ${r}_{i} \in  {\widehat{R}}_{i}$ 并获得效用 $u\left( {{t}_{i},s,{r}_{i}}\right)$。

参与者会采取行动以最大化自身效用。注意，由于在第三步之后没有进一步的交互，理性的参与者会选择 ${r}_{i} = {r}_{i}\left( {{t}_{i},s,{\widehat{R}}_{i}}\right)$，因此我们可以将此步骤视为非策略性步骤而忽略。设 $\mathcal{R} = {2}^{R}$。那么一个机制就是一个随机映射 $\mathcal{M} : \mathcal{T} \rightarrow  \mathcal{O} \times  {\mathcal{R}}^{n}$。

让我们考虑功利主义福利准则：$F\left( {t,s,r}\right)  =$ $\frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}u\left( {{t}_{i},s,{r}_{i}}\right)$。注意，该准则的敏感度为 $\Delta  = 1/n$，因为每个参与者的效用都在区间 $\left\lbrack  {0,1}\right\rbrack$ 内。因此，如果我们简单地选择一个结果 $s$ 并允许每个参与者做出他们的最优反应，指数机制就是一个 $\epsilon$ - 差分隐私机制。根据定理 3.11，该机制以高概率实现至少为 OPT $- O\left( \frac{\log \left| \mathcal{O}\right| }{\epsilon n}\right)$ 的社会福利。我们将这个具有质量得分 $F$、范围 $\mathcal{O}$ 和隐私参数 $\epsilon$ 的指数机制实例记为 ${\mathcal{M}}_{\epsilon }$。

其思路是在指数机制（具有良好的社会福利性质）和一个严格诚实机制（惩罚虚假报告，但社会福利性质较差）之间进行随机选择。如果我们进行适当的混合，就可以得到一个具有合理社会福利保证的严格诚实机制。

以下是一个这样的惩罚机制，它很简单，但对于给定的问题不一定是最优的：

定义 10.5。承诺机制 ${M}^{P}\left( {t}^{\prime }\right)$ 从 $s \in  \mathcal{O}$ 中均匀随机选择一个结果，并设定 ${\widehat{R}}_{i} = \left\{  {{r}_{i}\left( {{t}_{i}^{\prime },s,{R}_{i}}\right) }\right\}$，即它随机选择一个结果，并强制每个人按照他们报告的类型就是真实类型的方式做出反应。

将一个环境的差距定义为

$$
\gamma  = \mathop{\min }\limits_{{i,{t}_{i} \neq  {t}_{i}^{\prime },{t}_{-i}}}\mathop{\max }\limits_{{s \in  \mathcal{O}}}\left( {u\left( {{t}_{i},s,{r}_{i}\left( {{t}_{i},s,{R}_{i}}\right) }\right)  - u\left( {{t}_{i},s,{r}_{i}\left( {{t}_{i}^{\prime },s,{R}_{i}}\right) }\right) }\right) ,
$$

即 $\gamma$ 是参与者和类型在误报的最坏情况成本（关于 $s$）上的一个下界。注意，对于每个参与者，这种最坏情况至少以概率 $1/\left| \mathcal{O}\right|$ 出现。因此，我们有以下简单的观察结果：

引理 10.2。对于所有的 $i,{t}_{i},{t}_{i}^{\prime },{t}_{-i}$：

$$
u\left( {{t}_{i},{\mathcal{M}}^{P}\left( {{t}_{i},{t}_{-i}}\right) }\right)  \geq  u\left( {{t}_{i},{\mathcal{M}}^{P}\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  + \frac{\gamma }{\left| \mathcal{O}\right| }.
$$

注意，承诺机制是严格诚实的：每个个体至少有 $\frac{\gamma }{\left| \mathcal{O}\right| }$ 的激励不去说谎。

这表明存在一个具有良好社会福利保证的严格诚实机制：

定义 10.6。惩罚指数机制 ${\mathcal{M}}_{\epsilon }^{P}\left( t\right)$ 由参数 $0 \leq  q \leq  1$ 定义，它以概率 $1 - q$ 选择指数机制 ${\mathcal{M}}_{\epsilon }\left( t\right)$，以互补概率 $q$ 选择惩罚机制 ${\mathcal{M}}^{P}\left( t\right)$。

观察到，根据期望的线性性质，对于所有的 ${t}_{i},{t}_{i}^{\prime },{t}_{-i}$，我们有：

$$
u\left( {{t}_{i},{\mathcal{M}}_{\epsilon }^{P}\left( {{t}_{i},{t}_{-i}}\right) }\right)  = \left( {1 - q}\right)  \cdot  u\left( {{t}_{i},{\mathcal{M}}_{\epsilon }\left( {{t}_{i},{t}_{-i}}\right) }\right)  + q \cdot  u\left( {{t}_{i},{\mathcal{M}}^{P}\left( {{t}_{i},{t}_{-i}}\right) }\right) 
$$

$$
 \geq  \left( {1 - q}\right) \left( {u\left( {{t}_{i},{\mathcal{M}}_{\epsilon }\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  - {2\epsilon }}\right) 
$$

$$
 + q\left( {u\left( {{t}_{i},{\mathcal{M}}^{P}\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  + \frac{\gamma }{\left| \mathcal{O}\right| }}\right) 
$$

$$
 = u\left( {{t}_{i},{\mathcal{M}}_{\epsilon }^{P}\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  - \left( {1 - q}\right) {2\epsilon } + q\frac{\gamma }{\left| \mathcal{O}\right| }
$$

$$
 = u\left( {{t}_{i},{\mathcal{M}}_{\epsilon }^{P}\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\right)  - {2\epsilon } + q\left( {{2\epsilon } + \frac{\gamma }{\left| \mathcal{O}\right| }}\right) .
$$

以下两个定理展示了该机制的激励和社会福利性质。

定理10.3。若${2\epsilon } \leq  \frac{q\gamma }{\left| \mathcal{O}\right| }$，则${\mathcal{M}}_{\epsilon }^{P}$是严格真实的。

注意，我们也为此机制提供了效用保证。设置参数$q$，使我们得到一个真实机制：

$$
{\mathbb{E}}_{s,\widehat{R} \sim  {\mathcal{M}}_{\epsilon }^{P}}\left\lbrack  {F\left( {t,s,r\left( {t,s,\widehat{R}}\right) }\right) }\right\rbrack  
$$

$$
 \geq  \left( {1 - q}\right)  \cdot  {\mathbb{E}}_{s,\widehat{R} \sim  {\mathcal{M}}_{\epsilon }}\left\lbrack  {F\left( {t,s,r\left( {t,s,\widehat{R}}\right) }\right) }\right\rbrack  
$$

$$
 = \left( {1 - \frac{{2\epsilon }\left| \mathcal{O}\right| }{\gamma }}\right)  \cdot  {\mathbb{E}}_{s,\widehat{R} \sim  {\mathcal{M}}_{\epsilon }}\left\lbrack  {F\left( {t,s,r\left( {t,s,\widehat{R}}\right) }\right) }\right\rbrack  
$$

$$
 \geq  \left( {1 - \frac{{2\epsilon }\left| \mathcal{O}\right| }{\gamma }}\right)  \cdot  \left( {\mathop{\max }\limits_{{t,s,r}}F\left( {t,s,r}\right)  - O\left( {\frac{1}{\epsilon n}\log \left| \mathcal{O}\right| }\right) }\right) 
$$

$$
 \geq  \mathop{\max }\limits_{{t,s,r}}F\left( {t,s,r}\right)  - \frac{{2\epsilon }\left| \mathcal{O}\right| }{\gamma } - O\left( {\frac{1}{\epsilon n}\log \left| \mathcal{O}\right| }\right) .
$$

设置

$$
\epsilon  \in  O\left( \sqrt{\frac{\log \left| \mathcal{O}\right| \gamma }{\left| \mathcal{O}\right| n}}\right) 
$$

我们发现：

$$
{\mathbb{E}}_{s,\widehat{R} \sim  {\mathcal{M}}_{\epsilon }^{P}}\left\lbrack  {F\left( {t,s,r\left( {t,s,\widehat{R}}\right) }\right) }\right\rbrack   \geq  \mathop{\max }\limits_{{t,s,r}}F\left( {t,s,r}\right)  - O\left( \sqrt{\frac{\left| \mathcal{O}\right| \log \left| \mathcal{O}\right| }{\gamma n}}\right) .
$$

注意，在这个计算中，我们假设$\epsilon  \leq  \gamma /\left( {2\left| \mathcal{O}\right| }\right)$，使得$q = \frac{{2\epsilon }\left| \mathcal{O}\right| }{\gamma } \leq  1$且该机制定义明确。对于足够大的$n$，这是成立的。也就是说，我们已经证明：

定理10.4。对于足够大的$n,{M}_{\epsilon }^{P}$，可实现社会福利达到

至少

$$
\mathrm{{OPT}} - O\left( \sqrt{\frac{\left| \mathcal{O}\right| \log \left| \mathcal{O}\right| }{\gamma n}}\right) .
$$

注意，这个机制无需支付就能保证真实性！

现在让我们考虑这个框架的一个应用：设施选址博弈。假设一个城市想建造$k$家医院，以最小化每个市民与其最近医院之间的平均距离。为简化问题，我们做一个温和的假设，即城市建在单位线段的离散化上。形式上，令$L\left( m\right)  = \left\{  {0,\frac{1}{m},\frac{2}{m},\ldots ,1}\right\}$表示步长为$1/m.\left| {L\left( m\right) }\right|  = m + 1$的离散单位线段。令对于所有$i$有$\mathcal{T} = {R}_{i} = L\left( m\right)$，且令$\left| \mathcal{O}\right|  = L{\left( m\right) }^{k}$。定义参与者$i$的效用为：

$$
u\left( {{t}_{i},s,{r}_{i}}\right)  = \left\{  \begin{array}{ll}  - \left| {{t}_{i} - {r}_{i}}\right| , & \text{ If }{r}_{i} \in  s \\   - 1, & \text{ otherwise. } \end{array}\right. 
$$

---

<!-- Footnote -->

${}^{3}$如果不是这种情况，我们可以轻松拆除然后重建城市。

<!-- Footnote -->

---

换句话说，参与者与线段上的点相关联，而一个结果是为$k$个设施中的每一个分配线段上的一个位置。参与者可以通过决定去哪个设施来对一组设施做出反应，他们做出这种决定的成本是他们自己的位置（即他们的类型）与他们选择的设施之间的距离。注意，这里${r}_{i}\left( {{t}_{i},s}\right)$是最近的设施${r}_{i} \in  s$。

我们可以实例化定理10.4。在这种情况下，我们有：$\left| \mathcal{O}\right|  =$ ${\left( m + 1\right) }^{k}$和$\gamma  = 1/m$，因为任意两个位置${t}_{i} \neq  {t}_{i}^{\prime }$之间的差异至少为$1/m$。因此，我们有：

定理10.5。针对设施选址博弈实例化的${M}_{\epsilon }^{P}$是严格真实的，并且至少能实现社会福利：

$$
\mathrm{{OPT}} - O\left( \sqrt{\frac{{km}{\left( m + 1\right) }^{k}\log m}{n}}\right) .
$$

对于少量的设施$k$，这已经非常好了，因为我们预期$\mathrm{{OPT}} = \Omega \left( 1\right)$。

### 10.3 针对注重隐私的参与者的机制设计

在上一节中，我们看到差分隐私作为一种为只关心机制所选结果的参与者设计机制的工具是有用的。在这里，我们主要将隐私视为在传统机制设计中实现目标的一种工具。作为一种附带影响，这些机制也保护了所报告的参与者类型的隐私。这本身是一个有价值的目标吗？为什么我们希望我们的机制保护参与者类型的隐私呢？

稍加思考便会发现，参与者（agent）可能会在意隐私。实际上，通过基本的内省可以推测，在现实世界中，参与者重视对某些“敏感”信息保密的能力，例如健康信息或性取向。在本节中，我们将探讨如何对这种隐私价值进行建模的问题，以及文献中采用的各种方法。

鉴于参与者可能对隐私有偏好，即便对于像福利最大化这类我们已经可以在不考虑隐私的情况下解决的任务，也值得考虑设计将保护隐私作为额外目标的机制。正如我们将看到的，确实有可能将维克里 - 克拉克 - 格罗夫斯（VCG）机制进行推广，以在任何社会选择问题中私下近似优化社会福利，同时在隐私参数和近似参数之间实现平滑权衡，并且保证精确的占优策略真实性。

然而，我们可能希望更进一步。当存在对隐私有偏好的参与者时，如果我们希望设计出具有真实性的机制，就必须以某种方式在他们的效用函数中对其隐私偏好进行建模，然后设计出相对于这些新的“考虑隐私”的效用函数具有真实性的机制。正如我们在差分隐私中所看到的，将隐私建模为机制本身的一种属性是最为自然的。因此，我们的效用函数不仅仅是结果的函数，而是结果和机制本身的函数。在几乎所有的模型中，参与者对结果的效用都被视为线性可分的，也就是说，对于每个参与者 $i$ ，

$$
{u}_{i}\left( {o,\mathcal{M},t}\right)  \equiv  {\mu }_{i}\left( o\right)  - {c}_{i}\left( {o,\mathcal{M},t}\right) .
$$

这里 ${\mu }_{i}\left( o\right)$ 表示参与者 $i$ 对结果 $o$ 的效用，而 ${c}_{i}\left( {o,\mathcal{M},t}\right)$ 表示当使用机制 $\mathcal{M}$ 选择结果 $o$ 时参与者 $i$ 所经历的（隐私）成本。

我们首先考虑也许是最简单（也是最天真）的隐私成本函数 ${c}_{i}$ 模型。回想一下，对于 $\epsilon  \ll  1$ ，差分隐私承诺对于每个参与者 $i$ ，以及对于每个可能的效用函数 ${f}_{i}$ 、类型向量 $t \in  {\mathcal{T}}^{n}$ 和偏差 ${t}_{i}^{\prime } \in  \mathcal{T}$ ：

$$
\left| {{\mathbb{E}}_{o \sim  M\left( {{t}_{i},{t}_{-i}}\right) }\left\lbrack  {{f}_{i}\left( o\right) }\right\rbrack   - {\mathbb{E}}_{o \sim  M\left( {{t}_{i}^{\prime },{t}_{-i}}\right) }\left\lbrack  {{f}_{i}\left( o\right) }\right\rbrack  }\right|  \leq  {2\epsilon }{\mathbb{E}}_{o \sim  M\left( t\right) }\left\lbrack  {{f}_{i}\left( o\right) }\right\rbrack  .
$$

如果我们将 ${f}_{i}$ 视为代表参与者 $i$ 的“预期未来效用”，那么将参与者 $i$ 的数据在 $\epsilon$ - 差分隐私计算中被使用的成本建模为与 $\epsilon$ 呈线性关系是很自然的。也就是说，我们认为参与者 $i$ 由某个值 ${v}_{i} \in  \mathbb{R}$ 参数化，并采用：

$$
{c}_{i}\left( {o,\mathcal{M},t}\right)  = \epsilon {v}_{i},
$$

其中 $\epsilon$ 是使得 $\mathcal{M}$ 具有 $\epsilon$ - 差分隐私的最小值。这里我们假设 ${v}_{i}$ 代表类似 ${\mathbb{E}}_{o \sim  M\left( t\right) }\left\lbrack  {{f}_{i}\left( o\right) }\right\rbrack$ 的一个量。在这种情况下， ${c}_{i}$ 不依赖于结果 $o$ 或类型分布 $t$ 。

使用这种简单的隐私度量方法，我们讨论隐私数据分析中的一个基本问题：当数据所有者重视他们的隐私并坚持为此获得补偿时，如何收集数据。在这种情况下，除了支付款项之外，参与者没有其他他们看重的“结果”，只有因隐私损失而产生的负效用。然后，我们将讨论这种（以及其他）隐私损失负效用度量方法的缺点，以及在更一般的机制设计场景中，当参与者 ${do}$ 从机制的结果中获得效用时的隐私问题。

#### 10.3.1 维克里 - 克拉克 - 格罗夫斯（VCG）机制的隐私推广

假设我们有一个一般的社会选择问题，由结果空间 $\mathcal{O}$ 和一组参与者 $N$ 定义，这些参与者对由 ${u}_{i} : \mathcal{O} \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 给出的结果有任意偏好。我们可能希望选择一个结果 $o \in  \mathcal{O}$ 来最大化社会福利 $F\left( o\right)  = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}{u}_{i}\left( o\right)$。众所周知，在任何这样的设定中，维克里 - 克拉克 - 格罗夫斯（VCG）机制可以实现恰好使社会福利最大化的结果 ${o}^{ * }$，同时收取费用，使得如实报告成为占优策略。如果我们想在保护隐私的同时实现相同的结果，该怎么办呢？隐私参数 $\epsilon$ 必须如何与我们对最优社会福利的近似程度进行权衡呢？

回想一下，我们可以使用指数机制来选择一个结果 $o \in  \mathcal{O}$，其质量得分是 $F$。对于隐私参数 $\epsilon$，这将给出一个分布 ${\mathcal{M}}_{\epsilon }$，定义为 $\Pr \left\lbrack  {{\mathcal{M}}_{\epsilon } = o}\right\rbrack   \propto$ $\exp \left( \frac{{\epsilon F}\left( o\right) }{2n}\right)$。此外，该机制具有良好的社会福利性质：以概率 $1 - \beta$，它会选择某个 $o$，使得：$F\left( o\right)  \geq$ $F\left( {o}^{ * }\right)  - \frac{2}{\epsilon n}\left( {\ln \frac{\left| \mathcal{O}\right| }{\beta }}\right)$。但正如我们所见，差分隐私仅能保证 $\epsilon$ - 近似的真实性。

然而，可以证明 ${\mathcal{M}}_{\epsilon }$ 是以下精确优化问题的解：

$$
{\mathcal{M}}_{\epsilon } = \arg \mathop{\max }\limits_{{\mathcal{D} \in  \Delta \mathcal{O}}}\left( {{\mathbb{E}}_{o \sim  \mathcal{D}}\left\lbrack  {F\left( o\right) }\right\rbrack   + \frac{2}{\epsilon n}H\left( \mathcal{D}\right) }\right) ,
$$

其中 $H$ 表示分布 $\mathcal{D}$ 的香农熵（Shannon Entropy）。换句话说，指数机制是恰好使期望社会福利加上由 $2/\left( {\epsilon n}\right)$ 加权的分布熵最大化的分布。这一点很重要，原因如下：已知任何在任何有限范围内恰好使期望参与者效用最大化的机制（称为分布范围最大机制）都可以与支付相结合，从而使如实报告成为精确的占优策略。指数机制是恰好使期望社会福利加上熵最大化的分布。换句话说，如果我们假设添加了一个额外的参与者，其效用恰好是分布的熵，那么指数机制在分布范围内是最大的。因此，它可以与支付相结合，使得如实报告对所有参与者（特别是 $n$ 个真实参与者）成为占优策略。此外，可以证明如何以保护隐私的方式收取费用。结论是，对于任何社会选择问题，社会福利可以以一种既保护差分隐私又完全真实的方式进行近似。

#### 10.3.2 敏感调查者问题

在本节中，我们考虑一位数据分析师的问题，他希望使用一群个体的私有数据进行研究。然而，他必须说服这些个体交出他们的数据！个体因隐私损失而产生成本。数据分析师可以通过保证差分隐私并对他们的损失进行补偿来减轻这些成本，同时试图获取具有代表性的数据样本。

考虑敏感调查员爱丽丝面临的以下典型问题。她的任务是对一组 $n$ 个个体 $N$ 进行调查，以确定个体 $i \in  N$ 中满足某些属性 $P\left( i\right)$ 的比例。她的最终目标是发现该统计量的真实值 $s = \frac{1}{n}\left| {\{ i \in  N : P\left( i\right) \} }\right|$，但如果无法做到这一点，她会满足于某个估计值 $\widehat{s}$，使得误差 $\left| {\widehat{s} - s}\right|$ 最小化。我们将采用基于大偏差界限的准确性概念，如果 $\Pr \left\lbrack  {\left| {\widehat{s} - s}\right|  \geq  \alpha }\right\rbrack   \leq  \frac{1}{3}$，则称调查机制是 $\alpha$ -准确的。不可避免的问题是，个体重视他们的隐私，不会免费参与调查。当个体与爱丽丝互动时，他们会因隐私损失而产生一定成本，必须为此损失获得补偿。更糟糕的是，这些个体是理性（即自私）的主体，如果误报成本能带来经济收益，他们很可能会向爱丽丝误报自己的成本。这使得爱丽丝的问题完全属于机制设计领域，要求爱丽丝制定一个在统计准确性和成本之间进行权衡的方案，同时还要管理个体的激励机制。

顺便说一下，这个典型问题与任何使用潜在敏感数据集合的组织都广泛相关。例如，这包括使用搜索日志来提供搜索查询补全、使用浏览历史来改善搜索引擎排名、使用社交网络数据来选择展示广告和推荐新链接，以及现在网络上提供的无数其他数据驱动的服务。在所有这些情况下，都是从敏感数据集合的统计属性中获取价值，以换取某种报酬 4

以固定价格交换收集数据可能会导致对总体统计量的有偏估计，因为这样的方案只会从那些对隐私的重视程度低于所提供价格的个体那里收集数据。然而，如果不与主体进行互动，我们就无法知道应该提供什么价格，才能让足够多的人参与，从而保证我们收集到的答案只有很小的偏差。因此，为了获得该统计量的准确估计，自然会考虑使用拍卖的方式购买隐私数据，以此来确定这个价格。在为隐私数据进行拍卖时，必须面对两个明显的障碍，还有一个不太明显但更具隐患的额外障碍。第一个障碍是，必须对“隐私”进行定量形式化，以便用于衡量主体在对其数据进行各种操作时的成本。在这里，差分隐私提供了一个明显的工具。对于较小的 $\epsilon$ 值，因为 $\exp \left( \epsilon \right)  \approx  \left( {1 + \epsilon }\right)$，所以正如前面所讨论的，一个简单（但可能比较天真）的初步模型是将每个主体视为在参与隐私研究时具有一定的线性成本。我们在这里假设每个主体 $i$ 都有一个未知的隐私价值 ${v}_{i}$，并且当他的隐私数据以 $\epsilon$ -差分隐私的方式被使用时，会经历 $\operatorname{cost}{c}_{i}\left( \epsilon \right)  = \epsilon {v}_{i}$ 5 第二个障碍是，我们的目标是与统计准确性进行权衡，而在机制设计中，统计准确性这一目标尚未得到充分研究。

---

<!-- Footnote -->

${}^{4}$ 报酬不一定是明确的和/或以美元计价的——例如，它可能是使用“免费”服务。

<!-- Footnote -->

---

最后一个更隐蔽的障碍是，个人因隐私泄露而付出的代价可能与其私人数据本身高度相关！假设我们只知道鲍勃非常重视其艾滋病状况的隐私，但并不明确知晓他是否患有艾滋病。这本身就具有泄露性，因为鲍勃的艾滋病状况可能与其对隐私的重视程度相关，而知道他为保护隐私愿意付出高昂代价，会让我们更新对其私人数据的推测。更关键的是，假设在一项艾滋病患病率调查的第一步，我们要求每个人报告他们对隐私的重视程度，然后打算通过拍卖来决定购买哪些人的数据。如果参与者如实报告，我们可能会发现报告的数值自然形成两个聚类：低重视程度者和高重视程度者。在这种情况下，我们甚至在收集任何数据或支付任何费用之前，就可能已经了解到了一些关于总体统计数据的信息——因此，参与者已经付出了代价。结果，参与者可能会虚报他们对隐私的重视程度，这可能会给调查结果带来偏差。这种现象使得直接披露机制存在问题，并将这个问题与经典机制设计区分开来。

有了一种量化参与者 $i$ 允许其数据被 $\epsilon$ -差分隐私算法 $\left( {{c}_{i}\left( \epsilon \right)  = \epsilon {v}_{i}}\right)$ 使用所造成损失的方法后，我们几乎可以描述敏感调查者问题的结果了。回想一下，差分隐私算法是一种映射 $M : {\mathcal{T}}^{n} \rightarrow  \mathcal{O}$，适用于一般类型空间 $\mathcal{T}$。接下来需要明确定义类型空间 $\mathcal{T}$ 究竟是什么。我们将考虑两种模型。在这两种模型中，我们都会为每个人关联一个比特 ${b}_{i} \in  \{ 0,1\}$，它表示这个人是否满足敏感谓词 $P\left( i\right)$，以及一个隐私价值 ${v}_{i} \in  {\mathbb{R}}^{ + }$。

---

<!-- Footnote -->

${}^{5}$ 正如我们稍后将讨论的，这个假设可能存在问题。

<!-- Footnote -->

---

1. 在不敏感价值模型中，我们通过将类型空间设为 $\mathcal{T} = \{ 0,1\}$ 来计算隐私机制的 $\epsilon$ 参数：即，我们仅根据机制对敏感比特 ${b}_{i}$ 的处理方式来衡量隐私成本，而忽略它对所报告的隐私价值 ${v}_{i}6$ 的处理方式。

2. 在敏感价值模型中，我们通过将类型空间设为 $\mathcal{T} = \left( {\{ 0,1\}  \times  {\mathbb{R}}^{ + }}\right)$ 来计算隐私机制的 $\epsilon$ 参数：即，我们根据机制对每个人的 $\left( {{b}_{i},{v}_{i}}\right)$ 这一对信息的处理方式来衡量隐私。

直观地说，不敏感价值模型假设个人忽略了其隐私价值与私人比特之间的相关性可能导致的潜在隐私损失，而敏感价值模型则假设个人认为这些相关性处于最坏情况，即他们的隐私价值 ${v}_{i}$ 与他们的私人比特 ${b}_{i}$ 一样具有泄露性。众所周知，在不敏感价值模型中，可以推导出近似最优的直接披露机制，以实现高精度和低成本。相比之下，在敏感价值模型中，没有任何个体理性的直接披露机制能够实现任何非平凡的精度。

这导致了一种有些不尽如人意的情况。敏感价值模型捕捉到了我们真正想要处理的微妙问题，但在这个模型中却得到了一个不可能的结果！以令人满意的方式绕过这个结果（例如，通过改变模型或机制的能力）仍然是一个引人入胜的开放性问题。

#### 10.3.3 更好的隐私成本衡量方法

在上一节中，我们采用了一个简单的建模假设，即参与一个$\epsilon$ - 差分隐私机制$M$所产生的成本为${c}_{i}\left( {o,\mathcal{M},t}\right)  = \epsilon {v}_{i}$，其中${v}_{i}$为某个数值。这种衡量方法存在几个问题。首先，尽管差分隐私保证了任何参与者的效用损失上限是一个与$\epsilon$（近似）呈线性关系的量，但没有理由认为参与者的成本下限也是这样一个量。也就是说，虽然采用${c}_{i}\left( {o,\mathcal{M},t}\right)  \leq  \epsilon {v}_{i}$是有充分理由的，但几乎没有依据将这个不等式变成等式。其次，（事实证明）任何仅作为$\epsilon$的确定性函数（不仅仅是线性函数）的隐私衡量方法都会导致有问题的行为预测。

---

<!-- Footnote -->

${}^{6}$ 也就是说，处理报告值的映射部分不必是差分隐私的。

<!-- Footnote -->

---

那么，我们还可以如何对${c}_{i}$进行建模呢？一种自然的衡量方法是参与者报告的类型$i$与机制结果之间的互信息。为了使这个定义明确，我们必须处于一个每个参与者的类型${t}_{i}$都从一个已知的先验分布${t}_{i} \sim  \mathcal{T}$中抽取的环境中。每个参与者的策略是一个映射${\sigma }_{i} : \mathcal{T} \rightarrow  \mathcal{T}$，它根据参与者的真实类型确定他所报告的类型。然后我们可以定义

$$
{c}_{i}\left( {o,\mathcal{M},\sigma }\right)  = I\left( {\mathcal{T};\mathcal{M}\left( {{t}_{-i},\sigma \left( \mathcal{T}\right) }\right) ,}\right. 
$$

其中$I$是表示参与者$i$类型先验的随机变量$\mathcal{T}$与表示机制结果的随机变量$\mathcal{M}\left( {{t}_{-i},\sigma \left( \mathcal{T}\right) }\right)$之间的互信息，这里的机制结果是在给定参与者$i$的策略下得到的。

这种衡量方法具有很大的吸引力，因为它表示了机制的输出与参与者$i$的真实类型之间的“关联程度”。然而，除了需要一个关于参与者类型的先验分布之外，我们还可以观察到这种隐私损失衡量方法所导致的一个有趣的悖论。考虑这样一个世界，其中有两种三明治面包：黑麦面包（R）和小麦面包（W）。此外，在这个世界中，三明治偏好是非常私密且令人尴尬的。类型$\mathcal{T}$的先验分布在$\mathrm{R}$和$\mathrm{W}$上是均匀的，并且机制$\mathcal{M}$只是给参与者$i$提供他声称喜欢的那种类型的三明治。现在考虑两种可能的策略，${\sigma }_{\text{truthful }}$和${\sigma }_{\text{random }}$。${\sigma }_{\text{truthful }}$对应于如实报告三明治偏好（随后会吃到喜欢的三明治类型），而${\sigma }_{\text{random }}$则独立于真实类型随机报告（结果只有一半的时间能吃到喜欢的三明治）。使用随机策略的成本是$I\left( {\mathcal{T};\mathcal{M}\left( {{t}_{-i},{\sigma }_{\text{random }}\left( \mathcal{T}\right) }\right)  = 0}\right.$，因为输出与参与者$i$的类型无关。另一方面，如实报告的成本是$I\left( {\mathcal{T};\mathcal{M}\left( {{t}_{-i},{\sigma }_{\text{truthful }}\left( \mathcal{T}\right) }\right)  = 1}\right.$，因为现在三明治的结果是参与者$i$类型的恒等函数。然而，从任何外部观察者的角度来看，这两种策略是无法区分的！在这两种情况下，参与者$i$都会得到一个均匀随机的三明治。那么为什么有人会选择随机策略呢？只要对手认为他们是随机选择的，他们就应该选择诚实策略。

另一种不需要关于参与者类型先验分布的方法如下。我们可以将参与者建模为具有一个满足以下条件的成本函数${c}_{i}$：

$$
\left| {{c}_{i}\left( {o,\mathcal{M},t}\right) }\right|  = \ln \left( {\mathop{\max }\limits_{{{t}_{i},{t}_{i}^{\prime } \in  \mathcal{T}}}\frac{\Pr \left\lbrack  {\mathcal{M}\left( {{t}_{i},{t}_{-i}}\right)  = o}\right\rbrack  }{\Pr \left\lbrack  {\mathcal{M}\left( {{t}_{i}^{\prime },{t}_{-i}}\right)  = o}\right\rbrack  }}\right) .
$$

注意，如果$\mathcal{M}$是$\epsilon$ - 差分隐私的，那么

$$
\mathop{\max }\limits_{{t \in  {\mathcal{T}}^{n}}}\mathop{\max }\limits_{{o \in  \mathcal{O}}}\mathop{\max }\limits_{{{t}_{i},{t}_{i}^{\prime } \in  \mathcal{T}}}\ln \left( \frac{\Pr \left\lbrack  {\mathcal{M}\left( {{t}_{i},{t}_{-i}}\right)  = o}\right\rbrack  }{\Pr \left\lbrack  {\mathcal{M}\left( {{t}_{i}^{\prime },{t}_{-i}}\right)  = o}\right\rbrack  }\right)  \leq  \epsilon .
$$

也就是说，我们可以将差分隐私视为对所有可能结果的最坏情况隐私损失进行界定，而这里提出的度量仅考虑实际实现的结果 $o$（以及类型向量 $t$）的隐私损失。因此，对于任何差分隐私机制 $\mathcal{M},\left| {{c}_{i}\left( {o,\mathcal{M},t}\right) }\right|  \leq  \epsilon$，对于所有 $o,t$ 都成立，但重要的是，成本可能因结果而异。

然后，我们可以考虑以下用于最大化社会福利 $F\left( o\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}{u}_{i}\left( o\right) .{}^{7}$ 的分配规则。我们讨论 $\left| \mathcal{O}\right|  = 2$ 的情况（这不需要支付），但也可以分析一般情况（有支付），该情况可以针对任何社会选择问题私下实现维克里 - 克拉克 - 格罗夫斯（VCG）机制。

1. 对于每个结果 $o \in  \mathcal{O}$，从分布 $\Pr \left\lbrack  {{r}_{o} = x}\right\rbrack   \propto  \exp \left( {-\epsilon \left| x\right| }\right)$ 中选择一个随机数 ${r}_{o}$。

2. 输出 ${o}^{ * } = \arg \mathop{\max }\limits_{{o \in  \mathcal{O}}}\left( {F\left( o\right)  + {r}_{o}}\right)$。

上述机制是$\epsilon$ -差分隐私的，并且对于注重隐私的参与者而言是真实的，只要对于每个参与者$i$，以及对于两个结果$o,{o}^{\prime } \in  \mathcal{O},\left| {{\mu }_{i}\left( o\right)  - {\mu }_{i}\left( {o}^{\prime }\right) }\right|  > {2\epsilon }$都成立。请注意，只要参与者对结果的效用是不同的，那么对于足够小的$\epsilon$，这将是成立的。分析过程是通过考虑随机变量${r}_{o}$的任意固定实现，以及第$i$个参与者偏离真实报告的任意偏差${t}_{i}^{\prime }$。有两种情况：在第一种情况下，这种偏差不会改变机制的结果$o$。在这种情况下，参与者对结果的效用${\mu }_{i}$以及他因隐私损失而产生的成本${c}_{i}$都完全不会改变，因此参与者不会从偏离中获益。在第二种情况下，如果当参与者$i$偏离时结果从$o$变为${o}^{\prime }$，那么必然有${\mu }_{i}\left( {o}^{\prime }\right)  < {\mu }_{i}\left( o\right)  - {2\epsilon }$。然而，根据差分隐私，$\left| {{c}_{i}\left( {o,\mathcal{M},t}\right)  - {c}_{i}\left( {{o}^{\prime },\mathcal{M},t}\right) }\right|  \leq  {2\epsilon }$，因此隐私成本的变化不足以使其变得有利。

---

<!-- Footnote -->

${}^{7}$ 这种分配规则与指数机制极为相似，实际上可以修改为与之完全相同。

<!-- Footnote -->

---

最后，通常考虑的对隐私成本进行建模的最保守方法如下。给定一个$\epsilon$ -差分隐私机制$\mathcal{M}$，仅假设

$$
{c}_{i}\left( {o,\mathcal{M},t}\right)  \leq  \epsilon {v}_{i}
$$

对于某个数${v}_{i}$。这与我们之前考虑的线性成本函数类似，但关键在于，这里我们仅假设一个上界。我们到目前为止所考虑的所有其他隐私成本模型都满足这一假设。可以证明，许多将差分隐私算法与能够限制用户选择的惩罚机制相结合的机制，就像我们在第10.2.3节中考虑的那些机制一样，只要值${v}_{i}$是有界的，在存在有隐私偏好的参与者的情况下，它们仍能保持其真实性属性。

### 10.4 参考文献注释

本节基于派伊（Pai）和罗斯（Roth）的综述[70]以及罗斯的综述[73]。差分隐私与机制设计之间的联系最初由杰森·哈特林（Jason Hartline）提出，并由麦克谢里（McSherry）和塔尔瓦尔（Talwar）在他们的开创性著作《通过差分隐私进行机制设计》[61]中进行了研究，他们在该著作中考虑了将差分隐私应用于设计近似真实的数字商品拍卖。在数字商品场景下，关于精确真实机制的最佳结果归功于巴尔坎（Balcan）等人[2]。

使用差分隐私作为工具来设计精确真实机制的问题最初由尼斯姆（Nissim）、斯莫罗丁斯基（Smorodinsky）和滕内霍尔茨（Tennenholtz）在[69]中进行了探索，他们也是首次对将差分隐私（单独使用）作为一种解决方案概念提出批评的人。本节中使用差分隐私来获得精确真实机制的示例直接取自[69]。敏感调查员问题最初由戈什（Ghosh）和罗斯[36]考虑，并由[56, 34, 75, 16]进行了扩展。弗莱舍尔（Fleischer）和吕（Lyu）[34]考虑了本节中讨论的贝叶斯场景，而利格特（Ligett）和罗斯[56]考虑了带有要么接受要么放弃提议的最坏情况场景，两者都是为了绕过[36]中的不可能结果。戈什和利格特考虑了一个相关模型，其中参与决策（和隐私保证）仅在均衡状态下确定[35]。

在将隐私明确视为其效用函数一部分的参与者存在的情况下进行机制设计的问题，最早由肖（Xiao）的具有影响力的研究[85]提出，他考虑了（在其他隐私成本度量中）互信息成本函数。在此之后，陈（Chen）等人[15]和尼斯姆（Nissim）等人[67]展示了在两种不同的模型中，即使对于重视隐私的参与者，有时也可以设计出诚实机制。陈冲（Chen Chong）、卡什（Kash）、莫兰（Moran）和瓦德汉（Vadhan）考虑了我们在本节中讨论的基于结果的成本函数，而尼斯姆、奥兰迪（Orlandi）和斯莫罗丁斯基（Smorodinsky）在$\epsilon  >$中考虑了仅用线性函数对每个参与者的成本进行上界约束的保守模型。根据互信息来衡量隐私的“三明治悖论”归因于尼斯姆、奥兰迪和斯莫罗丁斯基。

黄（Huang）和坎南（Kannan）证明，通过添加支付可以使指数机制变得完全诚实[49]。凯恩斯（Kearns）、派伊（Pai）、罗斯（Roth）和厄尔曼（Ullman）展示了如何通过在大型博弈中私下计算相关均衡，利用差分隐私来推导渐近诚实的均衡选择机制[54]。罗杰斯（Rogers）和罗斯[71]强化了这些结果，他们展示了如何在大型拥塞博弈中私下计算近似纳什均衡，这使得该机制具有更强的激励特性。这两篇论文都使用了“联合差分隐私”的解决方案概念，该概念要求对于每个参与者$i$，发送给其他参与者的消息的联合分布$j \neq  i$在$i$的报告中是差分隐私的。这个解决方案概念在其他隐私机制设计场景中也被证明是有用的，包括许（Hsu）等人[47]提出的用于计算隐私匹配的算法。

## 11 差分隐私与机器学习

数据分析中最有用的任务之一是机器学习：自动找到一个简单规则以准确预测从未见过的数据的某些未知特征的问题。许多机器学习任务可以在差分隐私的约束下执行。事实上，隐私约束不一定与机器学习的目标相冲突，两者都旨在从数据所来自的分布中提取信息，而不是从单个数据点中提取。在本节中，我们将概述一些关于隐私机器学习的最基本结果，而不试图全面涵盖这个广阔的领域。

机器学习的目标通常与隐私数据分析的目标相似。学习者通常希望学习一些能够解释数据集的简单规则。然而，她希望这个规则具有泛化能力——也就是说，她所学习的规则不仅要能正确描述她手头的数据，还要能够正确描述从同一分布中抽取的新数据。一般来说，这意味着她希望学习一个能够捕捉手头数据集的分布信息的规则，且这种方式不太依赖于任何单个数据点。当然，这正是隐私数据分析的目标——揭示隐私数据集的分布信息，而不泄露数据集中任何单个个体的过多信息。因此，机器学习和隐私数据分析密切相关也就不足为奇了。事实上，正如我们将看到的，我们通常能够以与非隐私机器学习几乎相同的准确性和几乎相同数量的示例来执行隐私机器学习。

让我们首先简要定义机器学习问题。在这里，我们将遵循瓦利安特（Valiant）的PAC（可能近似正确）机器学习模型。设$\mathcal{X} = \{ 0,1{\} }^{d}$为“无标签示例”的域。将每个$x \in  \mathcal{X}$视为包含$d$个布尔属性的向量。我们将认为向量$x \in  \mathcal{X}$与标签$y \in  \{ 0,1\}$配对。

定义11.1。一个带标签的示例是一个对$\left( {x,y}\right)  \in  \mathcal{X} \times  \{ 0,1\}$：一个向量与一个标签配对。

一个学习问题被定义为带标签示例上的一个分布$\mathcal{D}$。目标是找到一个函数$f : \mathcal{X} \rightarrow  \{ 0,1\}$，它能正确标记从该分布中抽取的几乎所有示例。

定义11.2。给定一个函数$f : \mathcal{X} \rightarrow  \{ 0,1\}$和带标签示例上的一个分布$\mathcal{D}$，$f$在$\mathcal{D}$上的错误率为：

$$
\operatorname{err}\left( {f,\mathcal{D}}\right)  = \mathop{\Pr }\limits_{{\left( {x,y}\right)  \sim  \mathcal{D}}}\left\lbrack  {f\left( x\right)  \neq  y}\right\rbrack  
$$

我们还可以定义$f$在有限样本$D$上的错误率：

$$
\operatorname{err}\left( {f,D}\right)  = \frac{1}{\left| D\right| }\left| {\{ \left( {x,y}\right)  \in  D : f\left( x\right)  \neq  y\} }\right| .
$$

学习算法可以观察从$\mathcal{D}$中抽取的一些带标签的示例，其目标是找到一个函数$f$，使其在$\mathcal{D}$上测量时的错误率尽可能小。衡量学习算法质量的两个参数是其运行时间，以及为了找到一个好的假设它需要观察的示例数量。

定义11.3。如果对于每个$\alpha ,\beta  > 0$，都存在一个$m = \operatorname{poly}\left( {d,1/\alpha ,\log \left( {1/\beta }\right) }\right)$，使得对于带标签示例上的每个分布$\mathcal{D}$，算法$A$将从$\mathcal{D}$中抽取的$m$个带标签示例作为输入，并输出一个假设$f \in  C$，使得以概率$1 - \beta$：

$$
\operatorname{err}\left( {f,\mathcal{D}}\right)  \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  + \alpha 
$$

如果$\mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  = 0$，则称学习者在可实现的设定下运行（即，该类中存在某个函数可以完美地为数据打标签）。否则，称学习者在不可知设定下运行。如果$A$的运行时间也是关于$d,1/\alpha$和$\log \left( {1/\beta }\right)$的多项式，则称该学习者是高效的。如果存在一个算法可以PAC学习$C$，则称$C$是PAC可学习的。

上述学习定义允许学习者直接访问带标签的示例。有时考虑这样的学习模型也很有用，即算法只能通过神谕（oracle）访问关于$\mathcal{D}$的一些含噪信息。

定义11.4。统计查询是某个函数$\phi  : \mathcal{X} \times  \{ 0,1\}  \rightarrow$$\left\lbrack  {0,1}\right\rbrack$。对于带标签示例分布$\mathcal{D}$，容差为$\tau$的统计查询神谕是一个神谕${\mathcal{O}}_{\mathcal{D}}^{\tau }$，使得对于每个统计查询$\phi$：

$$
\left| {{\mathcal{O}}_{\mathcal{D}}^{\tau }\left( \phi \right)  - {\mathbb{E}}_{\left( {x,y}\right)  \sim  \mathcal{D}}\left\lbrack  {\phi \left( {x,y}\right) }\right\rbrack  }\right|  \leq  \tau 
$$

换句话说，一个统计查询（SQ）神谕将统计查询$\phi$作为输入，并输出一个值，该值保证在从$\mathcal{D}$中抽取的示例上$\phi$的期望值的$\pm  \tau$范围内。

统计查询学习模型是为了对存在噪声情况下的学习问题进行建模而引入的。

定义11.5。如果对于每个$\alpha ,\beta  > 0$，都存在一个$m = \operatorname{poly}\left( {d,1/\alpha ,\log \left( {1/\beta }\right) }\right)$，使得算法$A$最多向${\mathcal{O}}_{\mathcal{D}}^{\tau }$进行$m$次容差为$\tau  = 1/m$的查询，并且以概率$1 - \beta$输出一个假设$f \in  C$，使得：

$$
\operatorname{err}\left( {f,\mathcal{D}}\right)  \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  + \alpha 
$$

请注意，除了通过统计查询（SQ）神谕之外，SQ学习算法无法访问$\mathcal{D}$。与PAC学习一样，我们可以讨论SQ学习算法是在可实现设定还是不可知设定下运行，以及该学习算法的计算效率。如果存在一个SQ学习算法可以学习类$C$，则称类$C$是SQ可学习的。

### 11.1 差分隐私机器学习的样本复杂度

关于隐私与学习之间的关系，人们可能提出的第一个问题或许是“何时可以以隐私保护的方式进行机器学习？” 换句话说，你可能会寻求一种概率近似正确（PAC）学习算法，该算法将数据集（隐式假设是从某个分布$\mathcal{D}$中采样得到的）作为输入，然后以隐私保护的方式输出一个假设$f$，该假设在该分布上大概率具有较低的误差。一个更细致的问题可能是“与在没有差分隐私约束的情况下进行学习所需的样本数量相比，以隐私保护的方式进行学习需要额外多少样本？” 类似地，“与非隐私保护学习所需的运行时间相比，以隐私保护的方式进行学习需要额外多少运行时间？” 我们在此将简要概述关于$\left( {\varepsilon ,0}\right)$-差分隐私的已知结果。一般来说，使用高级组合定理可以得到关于$\left( {\varepsilon ,\delta }\right)$-差分隐私的更好结果。

隐私机器学习中的一个基础信息论结果是，即使在不可知设置下，当且仅当非隐私的概率近似正确（PAC）学习可以用多项式数量的样本实现时，隐私的PAC学习才可以用多项式数量的样本实现。事实上，所需样本复杂度的增加相对较小 —— 然而，这个结果并不能保证计算效率。一种实现方法是直接通过指数机制。我们可以用一个范围$R = C$来实例化指数机制，该范围等于要学习的查询类。给定一个数据库$D$，我们可以使用质量得分$q\left( {f,D}\right)  =  - \frac{1}{\left| D\right| }\left| {\{ \left( {x,y}\right)  \in  D : f\left( x\right)  \neq  y\} }\right|$：即，我们试图最小化隐私数据集中错误分类示例的比例。这显然是隐私数据的一个$1/n$敏感函数，因此根据指数机制的效用定理，以概率$1 - \beta$，该机制会返回一个函数$f \in  C$，该函数能正确标记数据库中最优（OPT）$- \frac{2\left( {\log \left| C\right|  + \log \frac{1}{\beta }}\right) }{\varepsilon n}$比例的点。然而，请回想一下，在学习场景中，我们将数据库$D$视为由从某个带标签示例分布$\mathcal{D}$中独立同分布（i.i.d.）抽取的$n$个样本组成。回顾引理4.3中关于采样界限的讨论。切尔诺夫界（Chernoff bound）与联合界（union bound）相结合告诉我们，大概率地，如果$D$由从$\mathcal{D}$中独立同分布抽取的$n$个样本组成，那么对于所有$f \in  C : \left| {\operatorname{err}\left( {f,D}\right)  - \operatorname{err}\left( {f,\mathcal{D}}\right) }\right|  \leq  O\left( \sqrt{\frac{\log \left| C\right| }{n}}\right)$。因此，如果我们希望找到一个假设，其在分布$\mathcal{D}$上的误差在最优误差的$\alpha$范围内，那么抽取一个由$n \geq  \log \left| C\right| /{\alpha }^{2}$个样本组成的数据库$D$，并在$D$上学习最佳分类器${f}^{ * }$就足够了。

现在考虑使用上述指数机制进行隐私的概率近似正确（PAC）学习的问题。回想一下，根据定理3.11，指数机制返回一个效用得分比最优${f}^{ * }$的效用得分差超过一个加性因子$O\left( {\left( {{\Delta u}/\varepsilon }\right) \log \left| C\right| }\right)$的函数$f$的可能性非常小，在这种情况下，效用函数的敏感度${\Delta u}$为$1/n$。也就是说，大概率地，指数机制将返回一个函数$f \in  C$，使得：

$$
\operatorname{err}\left( {f,D}\right)  \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },D}\right)  + O\left( \frac{\left( \log \left| C\right| \right) }{\varepsilon n}\right) 
$$

$$
 \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  + O\left( \sqrt{\frac{\log \left| C\right| }{n}}\right)  + O\left( \frac{\left( \log \left| C\right| \right) }{\varepsilon n}\right) .
$$

因此，如果我们希望找到一个假设，其在分布$\mathcal{D}$上的误差在最优误差的$\alpha$范围内，那么抽取一个由以下数量样本组成的数据库$D$就足够了：

$$
n \geq  O\left( {\max \left( {\frac{\log \left| C\right| }{\varepsilon \alpha },\frac{\log \left| C\right| }{{\alpha }^{2}}}\right) }\right) ,
$$

当$\varepsilon  \geq  \alpha$时，这在渐近意义上并不比非隐私学习所需的数据库规模更大。

这个简单计算${}^{1}$的一个推论是（忽略计算效率），一类函数$C$是概率近似正确（PAC，Probably Approximately Correct）可学习的，当且仅当它是隐私概率近似正确可学习的。

对于一个$\mathrm{{SQ}}$可学习的概念类$C$，我们能得出更强的结论吗？观察可知，如果$C$是高效统计查询（SQ，Statistical Query）可学习的，那么$C$的学习算法只需通过一个统计查询预言机来访问数据，而统计查询预言机非常适合差分隐私：注意，统计查询预言机回答的是基于谓词$\phi \left( {x,y}\right)  \in  \left\lbrack  {0,1}\right\rbrack$定义的期望查询${\mathbb{E}}_{\left( {x,y}\right)  \sim  \mathcal{D}}\left\lbrack  {\phi \left( {x,y}\right) }\right\rbrack$，当在数据库$D$（它是来自$\mathcal{D}$的大小为$n$的样本）上进行估计时，该查询的敏感度仅为$1/n$。此外，学习算法不需要精确地接收答案，而是可以使用任何满足以下性质的答案$a$来运行：$\left| {{\mathbb{E}}_{\left( {x,y}\right)  \sim  \mathcal{D}}\left\lbrack  {\phi \left( {x,y}\right) }\right\rbrack   - a}\right|  \leq  \tau$，也就是说，该算法可以使用低敏感度查询的含噪答案来运行。这样做的好处是，我们可以使用拉普拉斯机制高效地在计算上回答此类查询，但代价是可能需要较大的样本规模。回顾一下，拉普拉斯机制可以以$\left( {\varepsilon ,0}\right)$ - 差分隐私回答${m1}/n$敏感查询，并且期望最坏情况误差为$\alpha  = O\left( \frac{m\log m}{\varepsilon n}\right)$。因此，一个需要以精度$\alpha$回答$m$个查询的${SQ}$学习算法可以使用样本规模为$n = O\left( {\max \left( {\frac{m\log m}{\varepsilon \alpha },\frac{\log m}{{\alpha }^{2}}}\right) }\right)$的样本运行。让我们将其与非隐私${SQ}$学习者所需的样本规模进行比较。如果${SQ}$学习者需要以容差$\alpha$进行$m$个查询，那么根据切尔诺夫界和联合界，样本规模为$O\left( {\log m/{\alpha }^{2}}\right)$就足够了。注意，对于$\varepsilon  = O\left( 1\right)$和误差$\alpha  = O\left( 1\right)$，非隐私算法可能需要的样本数量呈指数级减少。然而，在统计查询学习定义所允许的误差容差$\alpha  \leq  1/m$下，对于$\epsilon  = \Theta \left( 1\right)$，隐私统计查询学习的样本复杂度并不比非隐私统计查询学习的样本复杂度差。

其结果是，从信息论的角度来看，隐私对机器学习几乎没有阻碍。此外，对于任何仅通过统计查询预言机访问数据的算法，通过拉普拉斯机制可以立即实现向隐私学习的转化，并且还能保持计算效率！

---

<!-- Footnote -->

${}^{1}$再结合相应的下界，这些下界表明对于一般的$C$，不可能使用具有$o\left( {\log \left| C\right| /{\alpha }^{2}}\right)$个点的样本进行非隐私的概率近似正确学习。

${}^{2}$事实上，已知的几乎每一类（奇偶校验函数是唯一的例外）概率近似正确可学习的函数也都可以仅使用统计查询预言机进行学习。

<!-- Footnote -->

---

### 11.2 差分隐私在线学习

在本节中，我们考虑一个稍有不同的学习问题，即从专家建议中学习的问题。这个问题似乎与我们在上一节讨论的分类问题有所不同，但实际上，这里介绍的简单算法用途极为广泛，除了分类任务之外，还可用于执行许多其他任务，不过我们在此不做讨论。

想象一下，你正在对赛马进行下注，但不幸的是，你对马匹一无所知！不过，你可以获取一些 $k$ 位专家的意见，他们每天都会预测哪匹马会获胜。每天你可以选择一位专家并听从其建议，而且每天在你下注之后，你会得知哪匹马实际上赢得了比赛。你应该如何决定每天听从哪位专家的建议，又该如何评估自己的表现呢？专家并非完美无缺（事实上，他们可能根本就不擅长预测！），因此，期望你一直甚至大部分时间都做出正确的下注是不合理的，如果没有一位专家能做到这一点的话。然而，你可能有一个更温和的目标：从事后看来，你能否以一种方式对马匹下注，使得你的表现几乎和最佳专家一样好呢？

形式上，一个在线学习算法 $A$ 在以下环境中运行：

1. 每天 $t = 1,\ldots ,T$ ：

(a) $A$ 选择一位专家 ${a}_{t} \in  \{ 1,\ldots ,k\}$

(b) $A$ 观察到每位专家 $i \in  \{ 1,\ldots ,k\}$ 的损失 ${\ell }_{i}^{t} \in  \left\lbrack  {0,1}\right\rbrack$ ，并经历损失 ${\ell }_{{a}_{t}}^{t}$ 。

对于一系列损失 ${\ell }^{ \leq  T} \equiv  {\left\{  {\ell }^{t}\right\}  }_{t = 1}^{T}$ ，我们记为：

$$
{L}_{i}\left( {\ell }^{ \leq  T}\right)  = \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{\ell }_{i}^{t}
$$

表示专家 $i$ 在所有 $T$ 轮中的总平均损失，并记

$$
{L}_{A}\left( {\ell }^{ \leq  T}\right)  = \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{\ell }_{{a}_{t}}^{t}
$$

表示该算法的总平均损失。

该算法的遗憾值定义为其实际产生的损失与事后看来最佳专家的损失之间的差值：

$$
\operatorname{Regret}\left( {A,{\ell }^{ \leq  T}}\right)  = {L}_{A}\left( {\ell }^{ \leq  T}\right)  - \mathop{\min }\limits_{i}{L}_{i}\left( {\ell }^{ \leq  T}\right) .
$$

在线学习的目标是设计这样的算法，保证对于所有可能的损失序列 ${\ell }^{ \leq  T}$ ，即使是对抗性选择的序列，当 $T \rightarrow  \infty$ 时，遗憾值也保证趋于零。事实上，使用乘法权重算法（也有许多其他名称，例如，随机加权多数算法、Hedge 算法、指数梯度下降算法，其中乘法权重算法最为常用）就可以实现这一点。

注记 11.1. 我们在第 4 节中已经见过这个算法了——这只是乘法权重更新规则的另一种形式！事实上，关于私有乘法权重机制的所有结果都可以直接从我们在定理 11.1 中给出的遗憾界推导出来。

<!-- Media -->

算法 15 乘法权重（或随机加权多数（RWM））算法，版本 1。它以损失流 ${\ell }^{1},{\ell }^{2},\ldots$ 作为输入，并输出动作流 ${a}_{1},{a}_{2},\ldots$ 。它由一个更新参数 $\eta$ 进行参数化。

---

$\mathbf{{RWM}}\left( \eta \right)$ :

对于每个 $i \in  \{ 1,\ldots ,k\}$ ，令 ${w}_{i} \leftarrow  1$ 。

对于 $t = 1,\ldots$ 执行以下操作

以与 ${w}_{i}$ 成比例的概率选择动作 ${a}_{t} = i$

观察${\ell }^{t}$并对每个$i \in  \left\lbrack  k\right\rbrack$设置${w}_{i} \leftarrow  {w}_{i} \cdot  \exp \left( {-\eta {\ell }_{i}^{t}}\right)$

结束循环

---

<!-- Media -->

事实证明，这个简单的算法已经有了显著的遗憾界（regret bound）。

定理11.1。对于任意由对手选择的长度为$T,{\ell }^{ \leq  T} = \left( {{\ell }^{1},\ldots ,{\ell }^{T}}\right)$的损失序列，更新参数为$\eta$的随机加权多数算法（Randomized Weighted Majority algorithm）有如下保证：

$$
\mathbb{E}\left\lbrack  {\operatorname{Regret}\left( {\operatorname{RWM}\left( \eta \right) ,{\ell }^{ \leq  T}}\right) }\right\rbrack   \leq  \eta  + \frac{\ln \left( k\right) }{\eta T}, \tag{11.1}
$$

其中$k$是专家的数量。选择$\eta  = \sqrt{\frac{\ln k}{T}}$可得：

$$
\mathbb{E}\left\lbrack  {\operatorname{Regret}\left( {\operatorname{RWM}\left( \eta \right) ,{\ell }^{ \leq  T}}\right) }\right\rbrack   \leq  2\sqrt{\frac{\ln k}{T}}.
$$

这个显著的定理表明，即使面对一个由对手选择的损失序列，随机加权多数算法平均而言可以和事后看来$k$个专家中最好的专家表现一样好，仅减去一个额外的附加项，该项以$O\left( \sqrt{\frac{\ln k}{T}}\right)$的速率趋近于零。换句话说，在最多$T \leq  4\frac{\ln k}{{\alpha }^{2}}$轮之后，随机加权多数算法的遗憾（regret）保证至多为$\alpha$！此外，这个界是最优的。

我们能否在差分隐私（differential privacy）的约束下实现类似的结果呢？在我们提出这个问题之前，我们必须确定输入数据库是什么，以及我们希望以何种粒度保护隐私？由于输入是损失向量的集合${\ell }^{ \leq  T} = \left( {{\ell }^{1},\ldots ,{\ell }^{T}}\right)$，很自然地将${\ell }^{ \leq  T}$视为数据库，并将相邻数据库$\ell \overset{\widehat{ \leq  }T}{ \leq  }$视为在任何单个时间步的整个损失向量上有所不同的数据库：即，对于某个固定的时间步$t,{\widehat{\ell }}^{i} = {\ell }^{i}$，对于所有$i \neq  t$，但${\ell }^{t}$和${\widehat{\ell }}^{t}$可以任意不同。算法的输出是它选择的动作序列${a}_{1},\ldots ,{a}_{T}$，而我们希望以差分隐私的方式输出这个序列。

我们的第一个观察是，随机加权多数算法在每一天$t$以一种熟悉的方式选择一个动作！我们在这里以一种等价的方式重新表述该算法：

它以与$\exp \left( {-\eta \mathop{\sum }\limits_{{j = 1}}^{{t - 1}}{\ell }_{i}^{j}}\right)$成比例的概率选择一个动作${a}_{t}$，这仅仅是质量得分（quality score）为$q\left( {i,{\ell }^{ < T}}\right)  = \mathop{\sum }\limits_{{j = 1}}^{{t - 1}}{\ell }_{i}^{j}$、隐私参数为$\varepsilon  = {2\eta }$的指数机制（exponential mechanism）。注意，因为每个${\ell }_{i}^{t} \in  \left\lbrack  {0,1}\right\rbrack$，质量函数的敏感度为1。因此，在每一轮$t$，随机加权多数算法以一种保留${2\eta }$差分隐私的方式选择一个动作${a}_{t}$，所以为了实现隐私$\varepsilon$，只需设置$\eta  = \varepsilon /2$。

<!-- Media -->

算法16 乘法权重（或随机加权多数（RWM））算法，重新表述。它将损失流${\ell }^{1},{\ell }^{2},\ldots$作为输入，并输出动作流${a}_{1},{a}_{2},\ldots$。它由一个更新参数$\eta$参数化。

---

$\mathbf{{RWM}}\left( \eta \right)$ :

对于$t = 1,\ldots$执行

以与……成比例的概率选择动作${a}_{t} = i$

		$\exp \left( {-\eta \mathop{\sum }\limits_{{j = 1}}^{{t - 1}}{\ell }_{i}^{j}}\right)$

观察 ${\ell }^{t}$

结束循环

---

<!-- Media -->

此外，在算法运行过程中，它将选择一个动作 $T$ 次。如果我们希望算法的整个运行过程对于某些 $\varepsilon$ 和 $\delta$ 满足 $\left( {\varepsilon ,\delta }\right)$ -差分隐私，那么我们可以简单地应用我们的组合定理。回想定理3.20，由于总共有 $T$ 步，如果算法的每一步对于 ${\varepsilon }^{\prime } = \varepsilon /\sqrt{{8T}\ln \left( {1/\delta }\right) }$ 满足 $\left( {{\varepsilon }^{\prime },0}\right)$ -差分隐私，那么整个算法将满足 $\left( {\varepsilon ,\delta }\right)$ 差分隐私。因此，通过设置 $\eta  = {\varepsilon }^{\prime }/2$ 可以直接得到以下定理：

定理11.2。对于长度为 $T$ 的损失序列，带有 $\eta  = \frac{\varepsilon }{\sqrt{{32T}\ln \left( {1/\delta }\right) }}$ 的算法 $\operatorname{RWM}\left( \eta \right)$ 满足 $\left( {\varepsilon ,\delta }\right)$ -差分隐私。

值得注意的是，我们完全没有修改原始的随机加权多数算法就得到了这个定理，而只是通过适当地设置 $\eta$ 。从某种意义上说，我们免费获得了隐私性！因此，我们也可以不加修改地使用定理11.1，即随机加权多数（RWM）算法的效用定理：

定理11.3。对于任何由对手选择的长度为 $T,{\ell }^{ \leq  T} = \left( {{\ell }^{1},\ldots ,{\ell }^{T}}\right)$ 的损失序列，更新参数为 $\eta  = \frac{\varepsilon }{\sqrt{{32T}\ln \left( {1/\delta }\right) }}$ 的随机加权多数算法有如下保证：

$$
\mathbb{E}\left\lbrack  {\operatorname{Regret}\left( {\operatorname{RWM}\left( \eta \right) ,{\ell }^{ \leq  T}}\right) }\right\rbrack   \leq  \frac{\varepsilon }{\sqrt{{32T}\ln \left( {1/\delta }\right) }} + \frac{\sqrt{{32}\ln \left( {1/\delta }\right) }\ln k}{\varepsilon \sqrt{T}}
$$

$$
 \leq  \frac{\sqrt{{128}\ln \left( {1/\delta }\right) }\ln k}{\varepsilon \sqrt{T}},
$$

其中 $k$ 是专家的数量。

由于每个时间步 $t$ 的每轮损失是一个独立选择的随机变量（关于 ${a}_{t}$ 的选择），其取值范围在 $\left\lbrack  {-1,1}\right\rbrack$ 内，我们还可以应用切尔诺夫界（Chernoff bound）来获得高概率保证：

定理11.4。对于任何由对手选择的长度为 $T,{\ell }^{ \leq  T} = \left( {{\ell }^{1},\ldots ,{\ell }^{T}}\right)$ 的损失序列，更新参数为 $\eta  = \frac{\varepsilon }{\sqrt{{32T}\ln \left( {1/\delta }\right) }}$ 的随机加权多数算法产生的动作序列满足，至少以 $1 - \beta$ 的概率：

$$
\operatorname{Regret}\left( {\operatorname{RWM}\left( \eta \right) ,{\ell }^{ \leq  T}}\right)  \leq  \frac{\sqrt{{128}\ln \left( {1/\delta }\right) }\ln k}{\varepsilon \sqrt{T}} + \sqrt{\frac{\ln k/\beta }{T}}
$$

$$
 = O\left( \frac{\sqrt{\ln \left( {1/\delta }\right) }\ln \left( {k/\beta }\right) }{\varepsilon \sqrt{T}}\right) .
$$

这个界几乎和即使不考虑隐私性时所能达到的最佳界（即随机加权多数算法的界）一样好——遗憾界仅大了一个 $\Omega \left( \frac{\sqrt{\ln \left( k\right) \ln \left( {1/\delta }\right) }}{\varepsilon }\right)$ 的因子。（我们注意到，通过使用不同的算法并进行更细致的分析，我们可以去掉这个额外的 $\sqrt{\ln k}$ 因子）。由于我们实际上使用的是相同的算法，当然效率也得以保留。这里我们有一个机器学习中的有力例子，其中隐私性几乎是“免费的”。值得注意的是，就像非隐私算法一样，我们的效用界在算法运行时间越长时会变得更好，同时保持隐私保证不变。 ${}^{3}$

---

<!-- Footnote -->

${}^{3}$ 当然，我们必须适当地设置更新参数，就像我们对非隐私算法所做的那样。当轮数 $T$ 事先已知时，这很容易做到，但当轮数事先未知时，也可以自适应地完成。

<!-- Footnote -->

---

### 11.3 经验风险最小化

在本节中，我们将上一节讨论的随机加权多数算法应用于经验风险最小化问题的一个特殊情况，以学习一个线性函数。我们不假设采用对抗模型，而是假设示例是从某个已知分布中抽取的，并且我们希望从该分布的有限数量的样本中学习一个分类器，以便在从同一分布中抽取的新样本上我们的损失较低。

假设我们有一个关于示例 $x \in  {\left\lbrack  -1,1\right\rbrack  }^{d}$ 的分布 $\mathcal{D}$，对于每个这样的向量 $x \in  {\left\lbrack  -1,1\right\rbrack  }^{d}$，以及对于每个满足 $\parallel \theta {\parallel }_{1} = 1$ 的向量 $\theta  \in  {\left\lbrack  0,1\right\rbrack  }^{d}$，我们将 $\theta$ 在示例 $x$ 上的损失定义为 $\operatorname{Loss}\left( {\theta ,x}\right)  = \langle \theta ,x\rangle$。我们希望找到一个向量 ${\theta }^{ * }$ 来最小化从 $\mathcal{D}$ 中抽取的示例的期望损失：

$$
{\theta }^{ * } = \arg \mathop{\min }\limits_{{\theta  \in  {\left\lbrack  0,1\right\rbrack  }^{d} : \parallel \theta {\parallel }_{1} = 1}}{\mathbb{E}}_{x \sim  \mathcal{D}}\left\lbrack  {\langle \theta ,x\rangle }\right\rbrack  .
$$

这个问题可用于对寻找低误差线性分类器的任务进行建模。通常，我们只能通过从 $\mathcal{D}$ 中独立同分布抽取的一些示例 $S \subset  {\left\lbrack  -1,1\right\rbrack  }^{d}$ 来了解分布 $\mathcal{D}$，这些示例作为我们学习算法的输入。在这里，我们将这个样本 $S$ 视为我们的私有数据库，并将关注我们能够以多高的隐私性来近似 ${\theta }^{ * }$ 作为 $\left| S\right|$ 的函数的误差（学习算法的样本复杂度）。

我们的方法是将该问题简化为借助专家建议进行学习的问题，并应用上一节讨论的随机加权多数算法的隐私版本：

1. 专家将是 $d$ 个标准基向量 $\left\{  {{e}_{1},\ldots ,{e}_{d}}\right\}$，其中 ${e}_{i} = \left( {0,\ldots ,0,\underset{i}{\underbrace{1}},0,\ldots ,0}\right)$。

2. 给定一个示例 $x \in  {\left\lbrack  -1,1\right\rbrack  }^{d}$，我们通过为每个 $i \in  \{ 1,\ldots ,d\}$ 设置 $\ell {\left( x\right) }_{i} = \left\langle  {{e}_{i},x}\right\rangle$ 来定义一个损失向量 $\ell \left( x\right)  \in$ ${\left\lbrack  -1,1\right\rbrack  }^{d}$。换句话说，我们只需设置 $\ell {\left( x\right) }_{i} = {x}_{i}$。

3. 在时间 $t$，我们通过对 $x \sim  \mathcal{D}$ 进行采样并设置 ${\ell }^{t} = \ell \left( x\right)$ 来选择一个损失函数 ${\ell }^{t}$。请注意，如果我们有一个来自 $\mathcal{D}$ 的大小为 $\left| S\right|  = T$ 的样本 $S$，那么我们可以按照上述方式对损失序列运行随机加权多数（RWM）算法，总共进行 $T$ 轮。这将产生一系列输出 ${a}_{1},\ldots ,{a}_{T}$，我们将把我们的最终分类器定义为 ${\theta }^{T} \equiv  \frac{1}{T}\mathop{\sum }\limits_{{i = 1}}^{T}{a}_{i}$。（回想一下，每个 ${a}_{i}$ 都是一个标准基向量 ${a}_{i} \in  \left\{  {{e}_{1},\ldots ,{e}_{d}}\right\}$，因此我们有 ${\begin{Vmatrix}{\theta }^{T}\end{Vmatrix}}_{1} = 1$）。

我们在下面总结该算法：

<!-- Media -->

算法17 一种学习线性函数的算法。它以示例的私有数据库$S \subset  {\left\lbrack  -1,1\right\rbrack  }^{d},S = \left( {{x}_{1},\ldots ,{x}_{T}}\right)$以及隐私参数$\varepsilon$和$\delta$作为输入。

---

线性学习器$\left( {S,\varepsilon ,\delta }\right)$：

令$\eta  \leftarrow  \frac{\varepsilon }{\sqrt{{32T}\ln \left( {1/\delta }\right) }}$

对于从$t = 1$到$T = \left| S\right|$执行以下操作

以与……成比例的概率选择向量${a}_{t} = {e}_{i}$

		$\exp \left( {-\eta \mathop{\sum }\limits_{{j = 1}}^{{t - 1}}{\ell }_{i}^{j}}\right)$

令损失向量为${\ell }^{t} = \left( {\left\langle  {{e}_{1},{x}_{t}}\right\rangle  ,\left\langle  {{e}_{2},{x}_{t}}\right\rangle  ,\ldots ,\left\langle  {{e}_{d},{x}_{t}}\right\rangle  }\right)$。

结束循环

输出${\theta }^{T} = \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{a}_{t}$。

---

<!-- Media -->

我们已经知道线性学习器是具有隐私性的，因为它只是随机加权多数算法在更新参数$\eta$正确时的一个实例：

定理11.5。线性学习器$\left( {S,\varepsilon ,\delta }\right)$具有$\left( {\varepsilon ,\delta }\right)$ - 差分隐私性。

接下来需要分析线性学习器（LinearLearner）的分类准确率，这相当于考虑私有随机加权多数（RWM）算法的遗憾界。

定理11.6。如果$S$由$T$个独立同分布（i.i.d.）样本$x \sim  \mathcal{D}$组成，那么至少以$1 - \beta$的概率，线性学习器输出一个向量${\theta }^{T}$，使得：

$$
{\mathbb{E}}_{x \sim  \mathcal{D}}\left\lbrack  \left\langle  {{\theta }^{T},x}\right\rangle  \right\rbrack   \leq  \mathop{\min }\limits_{{\theta }^{ * }}{\mathbb{E}}_{x \sim  \mathcal{D}}\left\lbrack  \left\langle  {{\theta }^{ * },x}\right\rangle  \right\rbrack   + O\left( \frac{\sqrt{\ln \left( {1/\delta }\right) }\ln \left( {d/\beta }\right) }{\varepsilon \sqrt{T}}\right) ,
$$

其中$d$是专家的数量。证明。根据定理11.4，我们至少以$1 - \beta /2$的概率有以下保证：

$$
\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}\left\langle  {{a}_{t},{x}_{t}}\right\rangle   \leq  \mathop{\min }\limits_{{i \in  \{ 1,\ldots ,d\} }}\left\langle  {{e}_{i},\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{x}_{t}}\right\rangle   + O\left( \frac{\sqrt{\ln \left( {1/\delta }\right) }\ln \left( {d/\beta }\right) }{\varepsilon \sqrt{T}}\right) 
$$

$$
 = \mathop{\min }\limits_{{{\theta }^{ * } \in  {\left\lbrack  0,1\right\rbrack  }^{d} : {\begin{Vmatrix}{\theta }^{ * }\end{Vmatrix}}_{1} = 1}}\left\langle  {{\theta }^{ * },\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{x}_{t}}\right\rangle   + O\left( \frac{\sqrt{\ln \left( {1/\delta }\right) }\ln \left( {d/\beta }\right) }{\varepsilon \sqrt{T}}\right) .
$$

在第一个等式中，我们利用了单纯形上线性函数的最小值在单纯形的一个顶点处取得这一事实。注意到每个${x}_{t} \sim  \mathcal{D}$是相互独立的，并且每个$\left\langle  {{x}_{t},{e}_{i}}\right\rangle$都在$\left\lbrack  {-1,1}\right\rbrack$范围内有界，我们可以两次应用阿祖玛不等式（Azuma's inequality），至少以$1 - \beta /2$的概率对这两个量进行界定：

$$
\left| {\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}\left\langle  {{a}_{t},{x}_{t}}\right\rangle   - \frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{a}_{t},x}\right\rangle  }\right| 
$$

$$
 = \left| {\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}\left\langle  {{a}_{t},{x}_{t}}\right\rangle   - {\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{\theta }^{T},x}\right\rangle  }\right|  \leq  O\left( \sqrt{\frac{\ln \left( {1/\beta }\right) }{T}}\right) 
$$

并且

$$
\mathop{\max }\limits_{{i \in  \{ 1,\ldots ,d\} }}\left| {\left\langle  {{e}_{i},\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{x}_{t}}\right\rangle   - {\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{e}_{i},x}\right\rangle  }\right|  \leq  O\left( \sqrt{\frac{\ln \left( {d/\beta }\right) }{T}}\right) .
$$

因此我们也有：

$$
\mathop{\max }\limits_{{{\theta }^{ * } \in  {\left\lbrack  0,1\right\rbrack  }^{d} : {\begin{Vmatrix}{\theta }^{ * }\end{Vmatrix}}_{1} = 1}}\left| {\left\langle  {{\theta }^{ * },\frac{1}{T}\mathop{\sum }\limits_{{t = 1}}^{T}{x}_{t}}\right\rangle   - {\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{\theta }^{ * },x}\right\rangle  }\right|  \leq  O\left( \sqrt{\frac{\ln d/\beta }{T}}\right) .
$$

将这些不等式结合起来，我们得到关于算法${\theta }^{T}$输出的最终结果：

$$
{\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{\theta }^{T},x}\right\rangle   \leq  \mathop{\min }\limits_{{{\theta }^{ * } \in  {\left\lbrack  0,1\right\rbrack  }^{d} : {\begin{Vmatrix}{\theta }^{ * }\end{Vmatrix}}_{1} = 1}}{\mathbb{E}}_{x \sim  \mathcal{D}}\left\langle  {{\theta }^{ * },x}\right\rangle   + O\left( \frac{\sqrt{\ln \left( {1/\delta }\right) }\ln \left( {d/\beta }\right) }{\varepsilon \sqrt{T}}\right) .
$$

### 11.4 参考文献注释

机器学习的可能近似正确（PAC）模型由瓦利安特（Valiant）于1984年提出[83]，统计查询（SQ）模型由凯恩斯（Kearns）提出[53]。随机加权多数算法最初由利特尔斯特恩（Littlestone）和沃穆思（Warmuth）提出[57]，并且已经以多种形式进行了研究。有关综述，请参阅布卢姆（Blum）和曼苏尔（Mansour）[9]或阿罗拉（Arora）等人的文章[1]。我们用于随机加权多数算法的遗憾界在文献[1]中给出。

机器学习是差分隐私领域最早研究的主题之一，始于布卢姆（Blum）等人的工作[7]，他们表明在统计查询学习框架下运行的算法可以转换为隐私保护算法。差分隐私学习的样本复杂度最早由卡西维斯瓦纳坦（Kasiviswanathan）、李（Lee）、尼斯姆（Nissim）、拉斯霍德尼科娃（Raskhodnikova）和史密斯（Smith）在《我们能私下学习什么？》[52]中进行了考虑，该文献在多项式因子范围内刻画了私有学习的样本复杂度。有关私有学习样本复杂度的更精细分析，请参阅文献[3, 4, 12, 19]。

关于高效机器学习算法也有大量的研究工作，包括著名的支持向量机（SVM）和经验风险最小化器框架[13, 55, 76]。谱学习技术，包括主成分分析（PCA）和低秩矩阵近似也得到了研究[7,14,33,42,43,51]。

从专家建议中进行私有学习最早由德沃克（Dwork）等人考虑[26]。随机加权多数算法在不做修改的情况下（当更新参数设置适当时）具有隐私保护特性这一事实是业内常识（源于高级组合定理[32]），并且已被广泛应用；例如，在文献[48]中。有关私有在线学习的更一般研究，请参阅文献[50]，有关经验风险最小化的更一般研究，请参阅文献[50, 13]。

## 12 其他模型

到目前为止，我们对私有数据分析模型做了一些隐含的假设。例如，我们假设存在某个可信的数据管理者可以直接访问私有数据集，并且我们假设攻击者只能访问算法的输出，而无法访问算法执行过程中的任何内部状态。但如果情况并非如此呢？如果我们不信任任何人查看我们的数据，甚至不信任他们进行隐私保护的数据分析呢？如果某个黑客可能在私有算法运行时访问其内部状态呢？在本节中，我们放宽之前的一些假设并考虑这些问题。

在本节中，我们描述文献中受到关注的一些其他计算模型。

- 局部模型是随机化应答（见第2节）的推广，其动机来自于个体不信任数据管理者处理其数据的情况。虽然可以使用安全多方计算来模拟可信数据管理者的角色以解决这种不信任问题，但也有一些不需要密码学的技术。

接下来的两个模型考虑事件流，每个事件都可能与一个个体相关联。例如，一个事件可能是某个特定的人对任意术语进行的一次搜索。在给定的事件流中，与某个特定个体相关联的（可能有很多）事件可以与和其他个体相关联的事件任意交错。

- 在泛隐私（pan - privacy）中，数据管理者是可信的，但可能会被迫公开非隐私数据，例如，由于传票的要求，或者因为持有信息的实体被另一个可能不太可信的实体收购。因此，在泛隐私中，算法的内部状态也是差分隐私的，内部状态和输出的联合分布同样如此。

- 持续观察模型解决了在持续监测和报告事件统计信息（例如可能预示着即将爆发流行病的非处方药物购买情况）时如何维护隐私的问题。一些研究探讨了持续观察下的泛隐私问题。

### 12.1 本地模型

到目前为止，我们考虑的是数据隐私的集中式模型，在该模型中存在一个可以直接访问私有数据的数据库管理员。如果没有可信的数据库管理员会怎样呢？即使有合适的可信方，也有很多理由不希望私有数据由第三方进行聚合。私有信息聚合数据库的存在本身就增加了这样一种可能性：在未来的某个时候，它可能会落入不可信方的手中，要么是恶意地（通过数据盗窃），要么是组织更替的自然结果。从私有数据所有者的角度来看，更好的模型是本地模型，在该模型中，代理可以以差分隐私的方式（随机地）回答关于他们自己数据的问题，而无需与其他任何人共享这些数据。在谓词查询的背景下，这似乎严重限制了私有机制与数据交互的表达能力：机制可以询问每个用户其数据是否满足给定的谓词，而用户可能会抛硬币决定，以略高于回答错误的概率如实回答。在这个模型中，哪些是可行的呢？

本地隐私模型最初是在学习的背景下引入的。本地隐私模型将随机化响应形式化：不存在私有数据的中央数据库。相反，每个个体保留自己的数据元素（一个大小为 1 的数据库），并且仅以差分隐私的方式回答关于它的问题。形式上，数据库 $x \in  {\mathbb{N}}^{\left| \mathcal{X}\right| }$ 是来自某个域 $\mathcal{X}$ 的 $n$ 个元素的集合，并且每个 ${x}_{i} \in  x$ 由一个个体持有。

定义 12.1（本地随机化器）。一个 $\varepsilon$ - 本地随机化器 $R : \mathcal{X} \rightarrow$ $W$ 是一个以大小为 $n = 1$ 的数据库为输入的 $\varepsilon$ - 差分隐私算法。

在本地隐私模型中，算法只能通过本地随机化器预言机与数据库进行交互：

定义 12.2（LR 预言机）。一个 LR 预言机 $L{R}_{D}\left( {\cdot , \cdot  }\right)$ 以一个索引 $i \in  \left\lbrack  n\right\rbrack$ 和一个 $\varepsilon$ - 本地随机化器 $R$ 为输入，并根据分布 $R\left( {x}_{i}\right)$ 输出一个随机值 $w \in  W$，其中 ${x}_{i} \in  D$ 是数据库中第 $i$ 个个体持有的元素。

定义 12.3（（本地算法））。如果一个算法通过预言机 $L{R}_{D}$ 访问数据库 $D$，并且有以下限制，则该算法是 $\varepsilon$ - 本地的：如果 $L{R}_{D}\left( {i,{R}_{1}}\right) ,\ldots ,L{R}_{D}\left( {i,{R}_{k}}\right)$ 是该算法在索引 $i$ 上对 $L{R}_{D}$ 的调用，其中每个 ${R}_{J}$ 都是一个 ${\varepsilon }_{j}$ - 本地随机化器，那么 ${\varepsilon }_{1} + \cdots  + {\varepsilon }_{k} \leq  \varepsilon$

由于差分隐私是可组合的，很容易看出 $\varepsilon$ - 本地算法是 $\varepsilon$ - 差分隐私的。

观察12.1. $\varepsilon$ -局部算法具有$\varepsilon$ -差分隐私性。

也就是说，一个$\varepsilon$ -局部算法仅使用一系列$\varepsilon$ -差分隐私算法与数据进行交互，其中每个算法仅对大小为1的数据库进行计算。由于除数据所有者外，没有人会接触任何私有数据，因此局部设置更加安全：它不需要可信方，也不存在可能遭受黑客攻击的中央方。由于即使是算法也从未见过私有数据，因此算法的内部状态也始终具有差分隐私性（即，局部隐私意味着泛隐私，将在下一节中描述）。一个自然的问题是局部隐私模型的限制程度如何。在本节中，我们仅非正式地讨论相关结果。感兴趣的读者可以参考本节末尾的参考文献以获取更多信息。我们注意到，局部隐私模型的另一个名称是完全分布式模型。

我们回顾一下第11节中引入的统计查询（SQ）模型的定义。粗略地说，给定一个大小为$n$的数据库$x$，统计查询模型允许算法通过对数据库进行多项式（关于$n$）数量的含噪线性查询来访问该数据库，其中查询答案的误差是$n$的某个逆多项式。形式上：

定义12.4. 统计查询是某个函数$\phi  : \mathcal{X} \times  \{ 0,1\}  \rightarrow$ $\left\lbrack  {0,1}\right\rbrack$。对于具有容差$\tau$的带标签示例分布$\mathcal{D}$的统计查询预言机是一个预言机${\mathcal{O}}_{\mathcal{D}}^{\tau }$，使得对于每个统计

查询$\phi$：

$$
\left| {{\mathcal{O}}_{\mathcal{D}}^{\tau }\left( \phi \right)  - {\mathbb{E}}_{\left( {x,y}\right)  \sim  \mathcal{D}}\left\lbrack  {\phi \left( {x,y}\right) }\right\rbrack  }\right|  \leq  \tau 
$$

换句话说，一个SQ预言机将统计查询$\phi$作为输入，并输出一个保证在从$\mathcal{D}$中抽取的示例上$\phi$的期望值的$\pm  \tau$范围内的值。

定义12.5. 如果对于每个$\alpha ,\beta  > 0$，都存在一个$m = \operatorname{poly}\left( {d,1/\alpha ,\log \left( {1/\beta }\right) }\right)$，使得算法$A$对${\mathcal{O}}_{\mathcal{D}}^{\tau }$进行最多$m$次容差为$\tau  = 1/m$的查询，并且以概率$1 - \beta$输出一个假设$f \in  C$，使得：

$$
\operatorname{err}\left( {f,\mathcal{D}}\right)  \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  + \alpha 
$$

更一般地，如果一个算法（用于执行任何计算）仅通过SQ预言机访问数据，我们可以说它在SQ模型中运行：

定义12.6. 如果存在一个$m$，使得算法$A$对${\mathcal{O}}_{D}^{\tau }$进行最多$m$次容差为$\tau  = 1/m$的查询，并且没有其他访问数据库的方式，则称算法$A$在SQ模型中运行。如果$m$是数据库大小$D$的多项式，则$A$是高效的。

事实证明，在数据库大小和查询数量的多项式因子范围内，任何可以在SQ模型中实现的算法都可以在局部隐私模型中实现并进行隐私分析，反之亦然。我们注意到，在SQ模型中实现算法与在局部模型中进行其隐私分析之间存在区别：我们最终介绍的几乎所有算法都使用含噪线性查询来访问数据，因此可以认为它们在SQ模型中运行。然而，它们的隐私保证是在数据隐私的集中式模型中进行分析的（即，由于分析的某些“全局”部分，如稀疏向量算法）。

在以下总结中，我们还将回顾第11节中引入的PAC学习的定义：

定义12.7。若对于任意的 $\alpha ,\beta  > 0$ ，都存在一个 $m = \operatorname{poly}\left( {d,1/\alpha ,\log \left( {1/\beta }\right) }\right)$ ，使得对于带标签示例上的任意分布 $\mathcal{D}$ ，算法 $A$ 以从 $\mathcal{D}$ 中抽取的 $m$ 个带标签示例作为输入，并输出一个假设 $f \in  C$ ，且以概率 $1 - \beta$ 满足以下条件，则称算法 $A$ 能PAC学习（Probably Approximately Correct learning，概率近似正确学习）函数类 $C$ ：

$$
\operatorname{err}\left( {f,\mathcal{D}}\right)  \leq  \mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  + \alpha 
$$

若 $\mathop{\min }\limits_{{{f}^{ * } \in  C}}\operatorname{err}\left( {{f}^{ * },\mathcal{D}}\right)  = 0$ ，则称学习器在可实现设定下运行（即，该函数类中存在某个函数能完美地对数据进行标签标注）。否则，称学习器在不可知设定下运行。若 $A$ 的运行时间关于 $d,1/\alpha$ 和 $\log \left( {1/\beta }\right)$ 是多项式的，则称该学习器是高效的。若存在一个算法能PAC学习 $C$ ，则称 $C$ 是PAC可学习的。注意，统计查询（Statistical Query，SQ）学习算法和PAC学习算法的主要区别在于，PAC学习算法可以直接访问示例数据库，而SQ学习算法只能通过有噪声的SQ预言机访问数据。

以下是我们对SQ模型局限性的一些理解，以及将其与数据隐私的集中式模型区分开来的问题。

1. 在数据隐私的集中式模型中，使用拉普拉斯机制可以以误差 $O\left( 1\right)$ 回答单个敏感度为1的查询，但在局部数据隐私模型中需要误差 $\Theta \left( \sqrt{n}\right)$ 。

2. 在局部隐私模型中我们能够（恰当地）学习的函数类集合，恰好是在SQ模型中我们能够恰当地学习的函数类集合（在数据库大小和算法查询复杂度的多项式因子范围内）。相比之下，在集中式模型中我们能够（恰当地或不可知地）学习的内容集合，对应于在PAC模型中我们能够学习的内容集合。SQ学习严格弱于PAC学习，但这并不是一个巨大的障碍，因为奇偶函数本质上是唯一有趣的、可PAC学习但不可SQ学习的函数类。我们在此明确提及恰当地学习（即，函数类中存在某个函数能完美地对数据进行标签标注的设定）。在PAC模型中，恰当地学习和不可知地学习在信息论上没有区别，但在SQ模型中区别很大：见下一点。

3. 在局部隐私模型中我们能够发布的查询集合，恰好是在SQ模型中我们能够不可知地学习的查询集合。相比之下，在集中式模型中我们能够发布的内容集合，对应于在PAC模型中我们能够不可知地学习的内容集合。这是一个更大的障碍——即使是合取式（即，边缘分布）在SQ模型中也不可不可知地学习。这是由我们在第5节中使用迭代构造机制看到的从不可知学习（即，区分）到查询发布的信息论约简得出的。

我们注意到，如果我们只关注计算能力受限的对手，那么原则上分布式代理可以使用安全多方计算来模拟集中式设定下的隐私算法。虽然这实际上并不能提供差分隐私保证，但从计算能力受限的对手的角度来看，这种模拟的结果将与差分隐私计算的结果无法区分。然而，一般的安全多方计算协议通常需要大量的消息传递（因此有时运行时间过长），而局部隐私模型中的算法往往非常简单。

### 12.2 泛隐私流模型

泛私有算法的目标是，即使面对偶尔能够观察到算法内部状态的对手，仍能保持差分隐私。入侵可能出于多种原因，包括黑客攻击、传票传唤，或者当为一个目的收集的数据被用于另一个目的时出现的任务蔓延（“想想孩子们！”）。泛私有流算法可以防范所有这些情况。请注意，普通的流算法不一定能防范入侵带来的隐私泄露问题，因为即使是低内存的流算法也可能在内存中保留少量数据项，而这些数据项在入侵时会完全暴露。从技术层面来看，管理员可能知晓（传票传唤）或不知晓（黑客攻击）入侵情况。这些情况可能产生截然不同的影响，因为知晓入侵的管理员可以采取保护措施，比如对某些变量重新进行随机化处理。

#### 12.2.1 定义

我们假设存在一个由全域 $\mathcal{X}$ 中的元素组成的无限长数据流。将查询流上的数据分析作为动机可能会有所帮助，在查询流中，查询会附带发出者的 IP 地址。目前，我们忽略查询文本本身；全域 $\mathcal{X}$ 是潜在 IP 地址的集合。因此，直观地说，用户级隐私保护的是流中某个 IP 地址是否存在，而不考虑它实际出现的次数（如果它确实存在的话）。相比之下，事件级隐私仅保护单个访问的隐私。目前，我们专注于用户级隐私。

与差分隐私算法中常见的情况一样，对手可以任意控制输入流，并且可能拥有从其他来源获得的任意辅助知识。对手还可以拥有任意的计算能力。

我们假设算法会一直运行，直到收到一个特殊信号，此时它会产生（可观察到的）输出。算法可以选择继续运行，并在之后再次响应特殊信号产生额外的输出。由于输出是可观察到的，我们不对特殊信号提供隐私保护。

流算法会经历一系列内部状态，并产生一个（可能无界的）输出序列。用 I 表示算法可能的内部状态集合，用 $\sigma$ 表示可能的输出序列集合。我们假设对手只能观察到内部状态和输出序列；它无法看到流中的数据（尽管它可能拥有关于其中一些数据的辅助知识），并且无法获取输入序列的长度。

定义 12.8（ $\mathcal{X}$ -相邻数据流）。我们认为数据流是无限长的；前缀的长度是有限的。如果数据流 $S$ 和 ${S}^{\prime }$ 仅在单个元素 $u \in  \mathcal{X}$ 的所有出现情况的有无上存在差异，那么它们就是 $\mathcal{X}$ -相邻的。我们类似地定义流前缀的 $\mathcal{X}$ -相邻性。

用户级泛隐私。一个将数据流前缀映射到范围 $\mathrm{I} \times  \sigma$ 的算法 Alg，如果对于所有内部状态集合 ${\mathrm{I}}^{\prime } \subseteq  \mathrm{I}$ 和输出序列集合 ${\sigma }^{\prime } \subseteq  \sigma$，以及所有相邻数据流前缀对 $S,{S}^{\prime }$

$$
\Pr \left\lbrack  {\mathbf{{Alg}}\left( S\right)  \in  \left( {{\mathrm{I}}^{\prime },{\sigma }^{\prime }}\right) }\right\rbrack   \leq  {e}^{\varepsilon }\Pr \left\lbrack  {\mathbf{{Alg}}\left( {S}^{\prime }\right)  \in  \left( {{\mathrm{I}}^{\prime },{\sigma }^{\prime }}\right) }\right\rbrack  ,
$$

其中概率空间是基于算法 Alg 的随机掷币结果。

这个定义仅涉及单次入侵。对于多次入侵，我们必须考虑对内部状态和输出的观察的交织情况。

通过修改相邻性的概念可以得到事件级隐私的放宽定义，大致来说，如果两个流在全域 $\mathcal{X}$ 中单个元素的单个实例上存在差异，即删除/添加了一个元素的一个实例，那么它们就是事件相邻的。显然，事件级隐私的保障力度远低于用户级隐私。

注记 12.1。如果我们假设存在极少量对手不可见的秘密存储，那么对于许多我们无法获得泛私有解决方案的问题，都存在（非泛）私有流解决方案。然而，秘密存储的数量不如其存在本身重要，因为秘密存储容易受到社会压力的影响，而泛隐私旨在保护数据（和管理员）免受这些社会压力的侵害。

泛私有密度估计。相当令人惊讶的是，即使对于许多常见流计算的用户级隐私，也能实现泛隐私。例如，考虑密度估计问题：给定一个数据元素的全集 $\mathcal{X}$ 和一个数据流 $\sigma$，目标是估计 $\mathcal{X}$ 中实际出现在数据流中的元素比例。例如，全集由给定社区中的所有青少年（由 IP 地址表示）组成，目标是了解访问计划生育网站的青少年比例。

用于密度估计的标准低内存流解决方案涉及记录至少一些输入项的确定性计算结果，这种方法本质上不是泛私有的。这里有一个简单但内存需求高的解决方案，它受到随机响应的启发。该算法为每个 IP 地址 $a$（可能在数据流中出现任意次数）维护一个比特 ${b}_{a}$，初始值是均匀随机的。数据流一次处理一个元素。输入 $a$ 时，算法翻转一个偏向 1 的比特；也就是说，这个有偏比特取值为 0 的概率是 $1/2 - \varepsilon$，取值为 1 的概率是 $1/2 + \varepsilon$。该算法执行此过程与 IP 地址 $a$ 在数据流中出现的次数无关。这个算法是 $\left( {\varepsilon ,0}\right)$ -差分隐私的。与随机响应一样，我们可以通过 $z = 2\left( {y - \left| \mathcal{X}\right| /2}\right) /\left| \mathcal{X}\right|$ 来估计“真实” 1 的比例，其中 $y$ 是处理完数据流后表中 1 的实际数量。为了确保泛隐私，算法发布 $z$ 的一个含噪版本。与随机响应一样，误差将在 $1/\sqrt{\left| \mathcal{X}\right| }$ 数量级，当密度较高时会产生有意义的结果。

其他拥有用户级泛私有算法的问题包括：

- 对于任意 $t$，估计恰好出现 $t$ 次的元素比例；

- 估计 $t$ -裁剪均值：大致来说，是所有元素的 $t$ 和该元素在数据流中出现次数的最小值的平均值；

- 估计 $k$ -频繁项（$\mathcal{X}$ 中在数据流中至少出现 $k$ 次的元素）的比例。

这些问题的变体也可以针对完全动态的数据进行定义，在这种数据中计数既可以增加也可以减少。例如，密度估计（数据流中出现的比例是多少？）变成了“有多少（或多大比例）的元素（净）计数等于零？”这些问题也可以通过使用流文献中草图技术的差分隐私变体，以用户级泛隐私的方式解决。

### 12.3 持续观察

数据分析的许多应用涉及重复计算，这要么是因为整个目标是进行监控，例如监控交通状况、搜索趋势或流感发病率。在这类应用中，系统需要持续产生输出。因此，我们需要在持续观察下实现差分隐私的技术。

像往常一样，差分隐私要求每对相邻数据库的输出具有基本相同的分布，但在这种情况下我们应该如何定义相邻性呢？让我们考虑两个示例场景。

假设目标是通过分析 H1N1 自我评估网站的统计数据来监测公共卫生。个人可以与该网站交互，以了解他们正在经历的症状是否可能表明感染了 H1N1 流感。用户填写一些人口统计数据（年龄、邮政编码、性别），并回答有关其症状的问题（体温是否超过 ${100.4}^{ \circ  }\mathrm{F}$？是否喉咙痛？症状持续时间？）。我们预计给定的个人与 H1N1 自我评估网站的交互次数非常少（例如，如果我们将关注范围限制在六个月内）。为简单起见，假设只有一次。在这种情况下，确保事件级隐私就足够了，其隐私目标是隐藏单个事件（一个用户与自我评估网站的交互）的存在与否。

---

<!-- Footnote -->

${}^{1}$ https://h1n1.cloudapp.net 在 2010 年冬季提供了这样一项服务；用户提供的数据在获得用户同意后被存储用于分析。

<!-- Footnote -->

---

再次假设目标是监测公众健康，这次是通过分析提交给医疗搜索引擎的搜索词来实现。在这种情况下，即使我们将关注范围限制在相对较短的时间段内，假设一个人与该网站的交互很少可能不再安全。在这种情况下，我们需要用户级别的隐私保护，确保同时保护用户的整个搜索词集合。

我们将连续观察算法视为在离散的时间间隔内采取步骤；在每个步骤中，算法接收输入、进行计算并产生输出。我们将数据建模为以流的形式到达，每个时间间隔最多有一个数据元素。为了体现现实生活中存在没有任何事情发生的时间段这一事实，空事件在数据流中用一个特殊符号来建模。因此，“ $t$ 个时间段”的直观概念对应于处理流中 $t$ 个元素的序列。

例如，下面计数器原语背后的动机是计算自算法启动以来某件事情发生的次数（计数器非常通用；我们事先不指定它在计数什么）。这通过一个基于 $\{ 0,1\}$ 的输入流来建模。这里，“0”表示“没有事情发生”，“1”表示感兴趣的事件发生了，并且对于 $t = 1,2,\ldots ,T$ ，算法输出流的长度为 $t$ 的前缀中看到的“1”的数量的近似值。

有三种自然的选择：

1. 对每个时间段使用随机响应，并将这个随机值添加到计数器中；

2. 对每个时间步的真实值添加根据 $\operatorname{Lap}\left( {1/\varepsilon }\right)$ 分布的噪声，并将这个扰动后的值添加到计数器中；

3. 在每个时间步计算真实计数，对计数添加根据 $\operatorname{Lap}\left( {T/\varepsilon }\right)$ 分布的噪声，并发布这个含噪计数。

所有这些选择都会导致至少为 $\Omega \left( {\sqrt{T}/\varepsilon }\right)$ 量级的噪声。我们希望通过利用查询集的结构来做得更好。

设 $\mathcal{X}$ 为可能的输入符号的全集。设 $S$ 和 ${S}^{\prime }$ 是从 $\mathcal{X}$ 中抽取的符号的流前缀（即有限流）。那么 $\operatorname{Adj}\left( {S,{S}^{\prime }}\right)$ （“ $S$ 与 ${S}^{\prime }$ 相邻”）当且仅当存在 $a,b \in  \mathcal{X}$ ，使得如果我们将 $S$ 中 $a$ 的某些实例更改为 $b$ 的实例，那么我们得到 ${S}^{\prime }$ 。更正式地说， $\operatorname{Adj}\left( {S,{S}^{\prime }}\right)$ 当且仅当 $\exists a,b \in  \mathcal{X}$ 且 $\exists R \subseteq  \left\lbrack  \left| S\right| \right\rbrack$ ，使得 ${\left. S\right| }_{R : a \rightarrow  b} = {S}^{\prime }$ 。这里， $R$ 是流前缀 $S$ 中的一个索引集， ${\left. S\right| }_{R : a \rightarrow  b}$ 是将这些索引处的所有 $a$ 替换为 $b$ 的结果。注意，相邻的前缀长度总是相同的。

为了实现事件级别的隐私保护，我们将邻接的定义限制在 $\left| R\right|  \leq  1$ 的情况。为了实现用户级别的隐私保护，我们在邻接的定义中不限制 $R$ 的大小。

如上所述，一种选择是在每个时间步发布一个含噪计数；在时间 $t$ 发布的计数反映了数据流长度为 $t$ 的前缀中 1 的近似数量。隐私方面的挑战在于，数据流中的早期元素几乎要接受 $T$ 次统计，因此对于 $\left( {\varepsilon ,0}\right)$ -差分隐私，我们需要添加规模为 $T/\varepsilon$ 的噪声，这是不可接受的。此外，由于 1 是数据流中“有趣”的元素，我们希望失真程度与数据流中出现的 $1\mathrm{\;s}$ 的数量成比例，而不是与数据流的长度成比例。这就排除了对数据流中的每个元素独立应用随机响应的可能性。

以下算法采用了一种将静态算法转换为动态算法的经典方法。

假设 $T$ 是 2 的幂。这些区间是与具有 $T$ 个叶子节点的完全二叉树的标签相对应的自然区间，其中叶子节点从左到右依次用区间 $\left\lbrack  {0,0}\right\rbrack  ,\left\lbrack  {1,1}\right\rbrack  ,\ldots ,\left\lbrack  {T - 1,T - 1}\right\rbrack$ 标记，每个父节点用其两个子节点标记区间的并集来标记。其思路是为每个标签 $\left\lbrack  {s,t}\right\rbrack$ 计算并发布一个含噪计数；也就是说，与标签 $\left\lbrack  {s,t}\right\rbrack$ 对应的发布值是输入数据流中位置 $s,s + 1,\ldots ,t$ 上 1 的含噪计数。为了了解时间 $t \in  \left\lbrack  {0,T - 1}\right\rbrack$ 的近似累积计数，分析人员使用 $t$ 的二进制表示来确定一组最多 ${\log }_{2}T$ 个不相交的区间，这些区间的并集为 $\left\lbrack  {0,t}\right\rbrack$ ，并计算相应发布的含噪计数的总和。见图 12.1。

<!-- Media -->

---

计数器 $\left( {T,\varepsilon }\right)$

初始化。初始化 $\xi  = {\log }_{2}T/\varepsilon$ ，并采样计数器 $\sim  \operatorname{Lap}\left( \xi \right)$ 。

区间。对于 $i \in  \{ 1,\ldots ,\log T\}$ ，将每个字符串 $s \in  \{ 0,1{\} }^{i}$ 与时间区间 $S$ 关联起来

${2}^{\log T - i}$ 个时间段 $\left\{  {s \circ  {0}^{\log T - i},\ldots s \circ  {1}^{\log T - i}}\right\}$ 。该区间从时间 $s \circ  {0}^{\log T - i}$ 开始，到

时间 $s \circ  {1}^{\log T - i}$ 结束。

处理。在时间段 $t \in  \{ 0,1,\ldots ,T - 1\}$ ，设 ${x}_{t} \in  \{ 0,1\}$ 为第 $t$ 个输入位：

1. 对于每个从时间 $t$ 开始的区间 $I$ ，将 ${c}_{I}$ 初始化为一个独立的随机抽样： ${c}_{I} \leftarrow$

		$\operatorname{Lap}\left( {\left( {{\log }_{2}T}\right) /\varepsilon }\right)$ ;

2. 对于每个包含 $t$ 的区间 $I$ ，将 ${x}_{t}$ 加到 ${c}_{I} : {c}_{I} \leftarrow  {c}_{I} + {x}_{t}$ 上；

3. 对于每个在时间 $t$ 结束的区间 $I$ ，输出 ${c}_{I}$ 。

---

图 12.1：事件级隐私计数器算法（非全隐私）。

<!-- Media -->

每个流位置 $t \in  \left\lbrack  {0,T - 1}\right\rbrack$ 最多出现在 $1 + {\log }_{2}T$ 个区间中（因为树的高度为 ${\log }_{2}T$ ），因此流中的每个元素最多影响 $1 + {\log }_{2}T$ 个已发布的含噪计数。因此，根据 $\operatorname{Lap}\left( {\left( {1 + {\log }_{2}T}\right) /\varepsilon }\right)$ 为每个区间计数添加噪声可确保满足 $\left( {\varepsilon ,0}\right)$ -差分隐私。至于准确性，由于任何索引 $t \in  \left\lbrack  {0,T - 1}\right\rbrack$ 的二进制表示会产生一组最多包含 ${\log }_{2}T$ 个区间的不相交集合，其并集为 $\left\lbrack  {0,t}\right\rbrack$ ，我们可以应用下面的引理 12.2 得出结论：预期误差紧密集中在 ${\left( {\log }_{2}T\right) }^{3/2}$ 附近。在所有时间 $t$ 上的最大预期误差为 ${\left( {\log }_{2}T\right) }^{5/3}$ 量级。

引理 12.2。设 ${Y}_{1},\ldots ,{Y}_{k}$ 为服从分布 $\operatorname{Lap}\left( {b}_{i}\right)$ 的独立变量。设 $Y = \mathop{\sum }\limits_{i}{Y}_{i}$ 和 ${b}_{\max } = \mathop{\max }\limits_{i}{b}_{i}$ 。设 $\nu  \geq  \sqrt{\mathop{\sum }\limits_{i}{\left( {b}_{i}\right) }^{2}}$ ，以及 $0 < \lambda  < \frac{2\sqrt{2}{\nu }^{2}}{{b}_{\max }}$ 。则

$$
\Pr \left\lbrack  {Y > \lambda }\right\rbrack   \leq  \exp \left( {-\frac{{\lambda }^{2}}{8{\nu }^{2}}}\right) .
$$

证明。${Y}_{i}$ 的矩生成函数为 $\mathbb{E}\left\lbrack  {\exp \left( {h{Y}_{i}}\right) }\right\rbrack   = 1/(1 -$ $\left. {{h}^{2}{b}_{i}^{2}}\right)$ ，其中 $\left| h\right|  < 1/{b}_{i}$ 。使用不等式 ${\left( 1 - x\right) }^{-1} \leq  1 + {2x} \leq$ $\exp \left( {2x}\right)$ （对于 $0 \leq  x < 1/2$ ），若 $\left| h\right|  < 1/2{b}_{i}$ ，则有 $\mathbb{E}\left\lbrack  {\exp \left( {h{Y}_{i}}\right) }\right\rbrack   \leq  \exp \left( {2{h}^{2}{b}_{i}^{2}}\right)$ 。我们现在针对 $0 < h < 1/\sqrt{2}{b}_{\max }$ 进行计算：

$$
\Pr \left\lbrack  {Y > \lambda }\right\rbrack   = \Pr \left\lbrack  {\exp \left( {hY}\right)  > \exp \left( {h\lambda }\right) }\right\rbrack  
$$

$$
 \leq  \exp \left( {-{h\lambda }}\right) \mathbb{E}\left\lbrack  {\exp \left( {hY}\right) }\right\rbrack  
$$

$$
 = \exp \left( {-{h\lambda }}\right) \mathop{\prod }\limits_{i}\mathbb{E}\left\lbrack  {\exp \left( {h{Y}_{i}}\right) }\right\rbrack  
$$

$$
 \leq  \exp \left( {-{h\lambda } + 2{h}^{2}{\nu }^{2}}\right) .
$$

---

<!-- Footnote -->

${}^{2}$ 该算法可以进行轻微优化（例如，我们从不使用与根节点对应的计数，从而从树中消除一层），并且可以对其进行修改，以处理 $T$ 不是 2 的幂次方的情况，更有趣的是，处理 $T$ 事先未知的情况。

<!-- Footnote -->

---

根据假设，$0 < \lambda  < \frac{2\sqrt{2}{\nu }^{2}}{{b}_{\max }}$ 。我们通过设定 $h = \lambda /4{\nu }^{2} < 1/\sqrt{2}{b}_{\max }.$ 来完成证明。

推论 12.3。设 $Y,\nu ,{\left\{  {b}_{i}\right\}  }_{i},{b}_{\max }$ 如引理 12.2 所定义。对于 $\delta  \in$ ∈(0,1) 和 $\nu  > \max \left\{  {\sqrt{\mathop{\sum }\limits_{i}{b}_{i}^{2}},{b}_{\max }\sqrt{\ln \left( {2/\delta }\right) }}\right\}$ ，我们有 $\Pr \lbrack \left| Y\right|  >$ $\nu \sqrt{8\ln \left( {2/\delta }\right) }\rbrack  \leq  \delta$

在我们的例子中，所有的 ${b}_{i}$ 都是相同的（例如，$b = \left( {{\log }_{2}T}\right) /\varepsilon$ ）。取 $\nu  = \sqrt{k}b$ ，我们得到以下推论：

推论 12.4。对于所有的 $\lambda  < \alpha \left( {\sqrt{k}b}\right)  < 2\sqrt{2}{kb} = 2\sqrt{2k}\nu$ ，

$$
\Pr \left\lbrack  {Y > \lambda }\right\rbrack   \leq  {e}^{-{\alpha }^{2}/8}
$$

请注意，我们采取了不同寻常的步骤，即在计数之前而不是之后向计数中添加噪声。就输出而言，这没有区别（加法满足交换律）。然而，这对算法的内部状态有一个有趣的影响：它们具有差分隐私性！也就是说，假设入侵发生在时间 $t$ ，并考虑任意的 $i \in  \left\lbrack  {0,t}\right\rbrack$ 。由于最多有 ${\log }_{2}T$ 个区间包含步骤 $i$ （在算法中我们取消了与根节点对应的区间），${x}_{i}$ 最多影响 ${\log }_{2}T$ 个含噪声的计数，因此 ${x}_{i}$ 受到保护，防止入侵的原因与它在算法输出中受到保护的原因完全相同。然而，图 12.1 中的算法即使针对单次入侵也不是泛隐私的。这是因为，虽然其内部状态和输出各自独立地具有差分隐私性，但联合分布并不能确保 $\varepsilon$ -差分隐私性。为了理解为什么会这样，考虑一个入侵者，他在时间 $t$ 看到了内部状态，并且知道除 ${x}_{t + 1}$ 之外的整个数据流，设 $I = \left\lbrack  {a,b}\right\rbrack$ 是一个包含 $t$ 和 $t + 1$ 的区间。由于对手知道 ${x}_{\left\lbrack  0,t\right\rbrack  }$ ，它可以从 ${c}_{I}$ 中减去直到时间 $t$ 为止数据流的贡献（即，它从在时间 $t$ 观察到的 ${c}_{I}$ 中减去值 ${x}_{a},{x}_{a + 1},\ldots ,{x}_{t}$ ，所有这些它都知道）。由此，入侵者得知了 ${c}_{I}$ 初始化时所使用的拉普拉斯抽样的值。当 ${c}_{I}$ 在步骤 $b$ 结束时被公布时，对手从公布的值中减去这个初始抽样值，以及 ${x}_{\left\lbrack  a,b\right\rbrack  }$ 中除 ${x}_{t + 1}$ 之外所有元素的贡献，而 ${x}_{t + 1}$ 是它不知道的。剩下的就是未知的 ${x}_{t + 1}$ 。

#### 12.3.1 泛隐私计数

尽管图12.1中的算法可以轻松修改，以确保针对单次入侵的事件级泛隐私性，但我们在此给出一种不同的算法，以便引入一种强大的双射技术，该技术已被证明在其他应用中很有用。该算法在其内部状态中维护一个单一的噪声计数器（或累加器），以及每个时间间隔的噪声值。在任何给定时间段$t$的输出是累加器和包含$t$的时间间隔的噪声值之和。当一个时间间隔$I$结束时，其关联的噪声值${\eta }_{I}$将从内存中删除。

定理12.5。图12.2中的计数器算法，在使用参数$T,\varepsilon$运行且最多遭受一次入侵时，会产生一个$\left( {\varepsilon ,0}\right)$ - 泛隐私计数器，该计数器至少以$1 - \beta$的概率在其$T$个输出上的最大误差为$O\left( {\log \left( {1/\beta }\right)  \cdot  {\log }^{2.5}T/\varepsilon }\right)$。我们还注意到，在每一轮单独（而非所有轮同时）中，除了$\beta$的概率外，误差的幅度最多为$O(\log \left( {1/\beta }\right)$。${\log }^{1.5}T/\varepsilon )$。

证明。准确性的证明与图12.1中算法的证明相同，依赖于推论12.4。我们在此重点关注泛隐私性的证明。

在原子步骤${t}^{ * }$和${t}^{ * } + 1$之间的入侵期间，即紧接在输入流中处理元素${t}^{ * }$之后

<!-- Media -->

---

泛隐私计数器$\left( {T,\varepsilon }\right)$

初始化。初始化$\xi  = \left( {1 + \log T}\right) /\varepsilon$，并采样计数器$\sim  \operatorname{Lap}\left( \xi \right)$。

时间间隔。对于$i \in  \{ 1,\ldots ,\log T\}$，将每个字符串$s \in  \{ 0,1{\} }^{i}$与时间间隔$S$关联起来

${2}^{\log T - i}$个时间段$\left\{  {s \circ  {0}^{\log T - i},\ldots s \circ  {1}^{\log T - i}}\right\}$。该时间间隔从时间$s \circ  {0}^{\log T - i}$开始，结束于

时间$s \circ  {1}^{\log T - i}$。

处理。在时间段$t \in  \{ 0,1,\ldots ,T - 1\}$，设${x}_{t} \in  \{ 0,1\}$为第$t$个输入位：

1. 计数器$\leftarrow$ 计数器$+ {x}_{t}$；

2. 对于每个在时间$t$开始的时间间隔$I$，采样噪声${\eta }_{I} \sim  \operatorname{Lap}\left( \xi \right)$；

3. 设${I}_{1},\ldots ,{I}_{\log T}$为包含$t$的$\log T$个时间间隔。输出计数器$+ \mathop{\sum }\limits_{{i = 1}}^{{\log T}}{\eta }_{{I}_{i}}$。

4. 对于每个在时间$t$结束的时间间隔$I$，删除${\eta }_{I}$。

---

图12.2：事件级泛隐私计数器算法。

<!-- Media -->

(请记住，我们从 0 开始对元素进行编号)，对手的视角包括：(1) 含噪声的累积计数（在变量“count”中）；(2) 入侵发生时内存中的区间噪声值 ${\eta }_{S}$；以及 (3) 轮次 $0,1,\ldots ,t$ 中算法所有输出的完整序列。考虑相邻数据库 $x$ 和 ${x}^{\prime }$，它们在时间 $t$ 上有所不同，不失一般性地说，假设 ${x}_{t} = 1$ 和 ${x}_{t}^{\prime } = 0$，并且在时间段 ${t}^{ * } \geq  t$ 之后立即发生入侵（我们将在下面讨论 ${t}^{ * } < t$ 的情况）。我们将描述在 $x$ 上执行和在 ${x}^{\prime }$ 上执行时所使用的噪声值向量之间的一一对应关系，使得相应的噪声值在 $x$ 和 ${x}^{\prime }$ 上诱导出相同的对手视角，并且相邻噪声值的概率仅相差一个 ${e}^{\varepsilon }$ 乘法因子。这意味着满足 $\varepsilon$ - 差分泛隐私。

根据假设，当输入为 $x$ 时，时间段 ${t}^{ * } \geq  t$ 刚结束后的真实计数比输入为 ${x}^{\prime }$ 时更大。固定输入流为 $x$ 时的任意一次执行 ${E}_{x}$。这相当于固定了算法的随机性，进而固定了所生成的噪声值。我们将通过描述其噪声值与 ${E}_{x}$ 中的噪声值有何不同来描述相应的执行 ${E}_{{x}^{\prime }}$。

程序变量 Counter 用拉普拉斯噪声进行初始化。通过在 ${E}_{{x}^{\prime }}$ 中将此噪声增加 1，步骤 ${t}^{ * }$ 刚结束时 Counter 的值在 ${E}_{{x}^{\prime }}$ 和 ${E}_{x}$ 中相同。时间段 ${t}^{ * }$ 刚结束时内存中的噪声变量与输入无关；这些在 ${E}_{{x}^{\prime }}$ 中将保持不变。我们将通过改变一组 $\log T$ 区间噪声值 ${\eta }_{S}$（对手入侵时这些值不在内存中），使 ${E}_{{x}^{\prime }}$ 中的输出序列与 ${E}_{x}$ 中的输出序列相同，从而使得直到 $t - 1$ 的所有轮次中所有噪声值的总和不变，但从轮次 $t$ 开始，数据库 ${x}^{\prime }$ 的噪声值总和比 $x$ 的大 1。由于我们增加了 Counter 的初始化噪声，现在需要将时间段 $0,\ldots ,t - 1$ 的区间噪声值总和减少 1，并保持从时间段 $t$ 开始的区间噪声值总和不变。

为此，我们找到一个并集为 $\{ 0,\ldots ,t - 1\}$ 的不相交区间集合。总是存在这样的集合，并且其大小至多为 $\log T$。我们可以通过迭代的方式构造它，对于从 $\left\lfloor  {\log \left( {t - 1}\right) }\right\rfloor$ 递减到 0 的 $i$，选择大小为 ${2}^{i}$ 且包含在 $\{ 0,\ldots ,t - 1\}$ 中但不包含在先前选择的区间内的区间（如果存在这样的区间）。给定这个不相交区间集合，我们还注意到它们都在时间 $t - 1 < t \leq  {t}^{ * }$ 结束，因此对手入侵时（恰好在时间段 ${t}^{ * }$ 之后）它们的噪声不在内存中。总体而言（同时考虑改变 Counter 的初始噪声值），对手看到的完整视角是相同的，并且用于 $x$ 和 ${x}^{\prime }$ 的（集合的）噪声值的概率至多相差一个 ${e}^{\varepsilon }$ 乘法因子。

注意，我们假设了${t}^{ * } \geq  t$。如果${t}^{ * } < t$，那么在${E}_{{x}^{\prime }}$中添加到计数器的初始噪声将与在${E}_{x}$中相同，并且我们需要在从$t$到$T$的每个时间段内，将区间噪声的总和加1（$t$时刻之前的区间噪声总和保持不变）。这与上述操作一样，通过找到一个最多包含$\log T$个区间的不相交集合来精确覆盖$\{ t,\ldots ,T - 1\}$。当在${t}^{ * } < t$时刻发生入侵时，这些区间的噪声值尚未存储在内存中，证明过程类似。

#### 12.3.2 一个（关于$T$的）对数下界

鉴于定理12.5中的上界，其中误差仅与$T$呈多项式对数关系，很自然会问是否存在某种内在的依赖关系。在本节中，我们将证明对$T$的对数依赖确实是内在的。

定理12.6。任何用于对$T$轮进行计数的差分隐私事件级算法的误差必须为$\Omega \left( {\log T}\right)$（即使在$\varepsilon  = 1$的情况下也是如此）。

证明。设$\varepsilon  = 1$。为了推出矛盾，假设存在一个用于长度为$T$的流的差分隐私事件级计数器，该计数器保证在至少$2/3$的概率下，其在所有时间段的计数误差最大不超过$\left( {{\log }_{2}T}\right) /4$。设$k =$ $\left( {{\log }_{2}T}\right) /4$。我们按如下方式构造一个包含$T/k$个输入的集合$S$。将$T$个时间段划分为$T/k$个连续的阶段，每个阶段的长度为$k$（可能最后一个阶段除外）。对于$i = 1,\ldots ,T/k$，第$i$个输入${x}^{i} \in  S$除了在第$i$个阶段外，其他位置的输入位均为0。即，${x}^{i} =$ ${0}^{k \cdot  i} \circ  {1}^{k} \circ  {0}^{k \cdot  \left( {\left( {T/k}\right)  - \left( {i + 1}\right) }\right) }$。

对于$1 \leq  i \leq  T/k$，如果在第$i$个阶段之前输出小于$k/2$，并且在第$i$个阶段结束时输出至少为$k/2$，我们就说该输出与$i$匹配。根据准确性，在输入为${x}^{i}$时，输出应至少以$2/3$的概率与$i$匹配。根据$\varepsilon$差分隐私，这意味着对于每个满足$i \neq  j$的$i,j \in  \left\lbrack  {T/k}\right\rbrack$，在输入为${x}^{i}$时，输出应至少以

$$
{e}^{-{2\varepsilon } \cdot  k} = {e}^{-\varepsilon \log \left( {T}^{1/2}\right) }
$$

$$
 = {e}^{-\log \left( {T}^{1/2}\right) } = 1/\sqrt{T}.
$$

这是一个矛盾，因为对于不同的$j$，输出与$j$匹配的事件是不相交的，但在输入为${x}^{i}$时，它们的概率之和超过了1。

### 12.4 查询发布的平均情况误差

在第4节和第5节中，我们考虑了用于解决私有查询发布问题的各种机制，当时我们关注的是最坏情况误差。也就是说，给定一个大小为$\left| \mathcal{Q}\right|  = k$的查询类$\mathcal{Q}$，我们希望恢复一个答案向量$\widehat{a} \in  {\mathbb{R}}^{k}$，使得对于每个查询${f}_{i} \in  \mathcal{Q}$，在某个最坏情况误差率$\alpha$下满足$\left| {{f}_{i}\left( x\right)  - {\widehat{a}}_{i}}\right|  \leq  \alpha$。换句话说，如果我们用$a \in  {\mathbb{R}}^{k}$表示真实答案的向量，其中${a}_{i} \equiv  {f}_{i}\left( x\right)$，那么我们需要一个形如$\parallel a - \widehat{a}{\parallel }_{\infty } \leq  \alpha$的界。在本节中，我们考虑一种弱化的效用保证，针对${\ell }_{2}$（而非${\ell }_{\infty }$）误差：一个形如$\parallel a - \widehat{a}{\parallel }_{2} \leq  \alpha$的界。这种形式的界并不能保证我们对每个查询都有低误差，但它确实保证了平均而言，我们的误差较小。

尽管这种界比最坏情况误差的界更弱，但该机制特别简单，并且它利用了一种我们此前未曾见过的、对查询发布问题的优雅几何视角。

回想一下，我们可以将数据库$x$视为一个向量$x \in  {\mathbb{N}}^{\left| \mathcal{X}\right| }$，其中$\parallel x{\parallel }_{1} = n$。类似地，我们也可以将查询${f}_{i} \in  \mathcal{Q}$视为向量${f}_{i} \in  {\mathbb{N}}^{\left| \mathcal{X}\right| }$，使得${f}_{i}\left( x\right)  = \left\langle  {{f}_{i},x}\right\rangle$。因此，将我们的查询类$\mathcal{Q}$视为一个矩阵$A \in  {\mathbb{R}}^{k \times  \left| \mathcal{X}\right| }$会很有帮助，其中$A$的第$i$行就是向量${f}_{i}$。然后我们可以看到，用矩阵表示法，我们的答案向量$a \in  {\mathbb{R}}^{k}$为：

$$
A \cdot  x = a.
$$

让我们考虑将$A$视为线性映射时的定义域和值域。用${B}_{1} = \left\{  {x \in  {\mathbb{R}}^{\left| \mathcal{X}\right| } : \parallel x{\parallel }_{1} = 1}\right\}$表示$\left| \mathcal{X}\right|$维空间中的单位${\ell }_{1}$球。注意到$x \in  n{B}_{1}$，因为$\parallel x{\parallel }_{1} = n$。我们将$n{B}_{1}$称为“数据库空间”。记$K = A{B}_{1}$。类似地注意到，对于所有的$x \in  n{B}_{1},a = A \cdot  x \in  {nK}$。我们将${nK}$称为“答案空间”。我们对$K$做几点观察：注意到因为${B}_{1}$是中心对称的，所以$K -$也是中心对称的，即$K =  - K$。还要注意到$K \subset  {\mathbb{R}}^{k}$是一个凸多面体，其顶点$\pm  {A}^{1},\ldots , \pm  {A}^{\left| \mathcal{X}\right| }$等于$A$的列向量及其负向量。

以下算法极其简单：它只是使用拉普拉斯机制（Laplace mechanism）独立地回答每个查询，然后将结果投影回答案空间。换句话说，它为每个查询添加独立的拉普拉斯噪声（Laplace noise），正如我们所见，这本身会导致与$k$呈线性关系的失真（或者，如果我们放宽到$\left( {\varepsilon ,\delta }\right)$ -差分隐私（differential privacy），至少与$\sqrt{k}$呈线性关系）。然而，得到的答案向量$\widetilde{a}$可能与数据库空间中的任何数据库$y \in  n{B}_{1}$都不一致。因此，它不是返回$\widetilde{a}$，而是返回某个与$\widetilde{a}$尽可能接近的一致答案向量$\widehat{a} \in  {nK}$。正如我们将看到的，这个投影步骤提高了机制的准确性，同时对隐私没有影响（因为这只是后处理！）

我们首先观察到投影（Project）是差分隐私的。

定理12.7。对于任何$A \in  {\left\lbrack  0,1\right\rbrack  }^{k \times  \left| \mathcal{X}\right| }$，投影$\left( {x,A,\varepsilon }\right)$保留$\left( {\varepsilon ,\delta }\right)$ -差分隐私。

<!-- Media -->

算法18 $K$ -投影拉普拉斯机制（Projected Laplace Mechanism）。它以矩阵$A \in  {\left\lbrack  0,1\right\rbrack  }^{k \times  \left| \mathcal{X}\right| }$、数据库$x \in  n{B}_{1}$以及隐私参数$\varepsilon$和$\delta$作为输入。

---

投影$\left( {x,A,\varepsilon ,\delta }\right)$：

设$a = A \cdot  x$

对于每个$i \in  \left\lbrack  k\right\rbrack$，采样${\nu }_{i} \sim  \operatorname{Lap}\left( {\sqrt{{8k}\ln \left( {1/\delta }\right) }/\varepsilon }\right)$，并设$\widetilde{a} = a + \nu$。

输出$\widehat{a} = \arg \mathop{\min }\limits_{{\widehat{a} \in  {nK}}}\parallel \widehat{a} - \widetilde{a}{\parallel }_{2}^{2}$。

---

<!-- Media -->

证明。我们只需注意到$\widetilde{a}$是拉普拉斯机制对敏感度为1的查询$k$的输出，根据定理3.6和3.20，它是$\left( {\varepsilon ,\delta }\right)$ -差分隐私的。最后，由于$\widehat{a}$是从$\widetilde{a}$导出的，且没有进一步访问私有数据，根据差分隐私的后处理保证（命题2.1），$\widehat{a}$的发布是差分隐私的。

定理12.8。对于任何线性查询类$A$和数据库$x$，设$a = A \cdot  x$表示真实答案向量。设$\widehat{a}$表示机制投影的输出：$\widehat{a} = \operatorname{Project}\left( {x,A,\varepsilon }\right)$。至少以$1 - \beta$的概率：

$$
\parallel a - \widehat{a}{\parallel }_{2}^{2} \leq  \frac{{kn}\sqrt{{192}\ln \left( {1/\delta }\right) \ln \left( {2\left| \mathcal{X}\right| /\beta }\right) }}{\varepsilon }.
$$

为了证明这个定理，我们将引入凸几何中的几个简单概念。对于一个凸体$K \subset  {\mathbb{R}}^{k}$，其极体${K}^{ \circ  }$定义为${K}^{ \circ  } = \left\{  {y \in  {\mathbb{R}}^{k} : \langle y,x\rangle  \leq  1\text{for all}x \in  K}\right\}$。由凸体$K$定义的闵可夫斯基范数（Minkowski Norm）为

$$
\parallel x{\parallel }_{K} \equiv  \min \{ r \in  \mathbb{R}\text{ such that }x \in  {rK}\} .
$$

$\parallel x{\parallel }_{K}$的对偶范数是由$K$的极体诱导的闵可夫斯基范数，即$\parallel x{\parallel }_{{K}^{ \circ  }}$。该范数还具有以下形式：

$$
\parallel x{\parallel }_{{K}^{ \circ  }} = \mathop{\max }\limits_{{y \in  K}}\langle x,y\rangle .
$$

我们将使用的关键事实是赫尔德不等式（Holder's Inequality），所有中心对称的凸体$K$都满足该不等式：

$$
\left| {\langle x,y\rangle }\right|  \leq  \parallel x{\parallel }_{K}\parallel y{\parallel }_{{K}^{ \circ  }}.
$$

定理12.8的证明。证明将分两步进行。首先，我们将证明：$\parallel a - \widehat{a}{\parallel }_{2}^{2} \leq  2\langle \widehat{a} - a,\widetilde{a} - a\rangle$，然后我们将使用赫尔德不等式来界定第二个量。

引理12.9。

$$
\parallel a - \widehat{a}{\parallel }_{2}^{2} \leq  2\langle \widehat{a} - a,\widetilde{a} - a\rangle 
$$

证明。我们计算：

$$
\parallel \widehat{a} - a{\parallel }_{2}^{2} = \langle \widehat{a} - a,\widehat{a} - a\rangle 
$$

$$
 = \langle \widehat{a} - a,\widetilde{a} - a\rangle  + \langle \widehat{a} - a,\widehat{a} - \widetilde{a}\rangle 
$$

$$
 \leq  2\langle \widehat{a} - a,\widetilde{a} - a\rangle .
$$

该不等式可通过以下计算得出：

$$
\langle \widehat{a} - a,\widetilde{a} - a\rangle  = \parallel \widetilde{a} - a{\parallel }_{2}^{2} + \langle \widehat{a} - \widetilde{a},\widetilde{a} - a\rangle 
$$

$$
 \geq  \parallel \widehat{a} - \widetilde{a}{\parallel }_{2}^{2} + \langle \widehat{a} - \widetilde{a},\widetilde{a} - a\rangle 
$$

$$
 = \langle \widehat{a} - \widetilde{a},\widehat{a} - a\rangle ,
$$

其中最后一个不等式成立是因为根据$\widehat{a}$的选择，对于所有${a}^{\prime } \in$ ${nK} : \parallel \widetilde{a} - \widehat{a}{\parallel }_{2}^{2} \leq  {\begin{Vmatrix}\widetilde{a} - {a}^{\prime }\end{Vmatrix}}_{2}^{2}.$

我们现在可以完成证明。回想一下，根据定义，$\widetilde{a} - a = \nu$是拉普拉斯机制添加的独立同分布拉普拉斯噪声向量。根据引理12.9和赫尔德不等式，我们有：

$$
\parallel a - \widehat{a}{\parallel }_{2}^{2} \leq  2\langle \widehat{a} - a,\nu \rangle 
$$

$$
 \leq  2\parallel \widehat{a} - a{\parallel }_{K}\parallel \nu {\parallel }_{{K}^{ \circ  }}.
$$

我们分别界定这两项。由于根据定义$\widehat{a},a \in  {nK}$，我们有$\max \left( {\parallel \widehat{a}{\parallel }_{K},\parallel a{\parallel }_{K}}\right)  \leq  n$，因此根据三角不等式，$\parallel \widehat{a} -$ $a\parallel K \leq  {2n}$。

接下来，注意到由于$\parallel \nu {\parallel }_{{K}^{ \circ  }} = \mathop{\max }\limits_{{y \in  K}}\langle y,\nu \rangle$，并且由于线性函数在多面体上的最大值在顶点处取得，我们有：$\parallel \nu {\parallel }_{{K}^{ \circ  }} = \mathop{\max }\limits_{{i \in  \left\lbrack  \left| \mathcal{X}\right| \right\rbrack  }}\left| \left\langle  {{A}^{i},\nu }\right\rangle  \right|$。

因为每个${A}^{i} \in  {\mathbb{R}}^{k}$都满足${\begin{Vmatrix}{A}^{i}\end{Vmatrix}}_{\infty } \leq  1$，并且回想一下，对于任何标量$q$，如果$Z \sim  \operatorname{Lap}\left( b\right)$，那么${qZ} \sim  \operatorname{Lap}\left( {qb}\right)$，我们可以应用引理12.2来界定拉普拉斯随机变量的加权和$\left\langle  {{A}^{i},\nu }\right\rangle$。这样做，我们得到至少以概率$1 - \beta$：

$$
\mathop{\max }\limits_{{i \in  \left\lbrack  \left| \mathcal{X}\right| \right\rbrack  }}\left| \left\langle  {{A}^{i},\nu }\right\rangle  \right|  \leq  \frac{{8k}\sqrt{\ln \left( {1/\delta }\right) \ln \left( {\left| \mathcal{X}\right| /\beta }\right) }}{\epsilon }.
$$

综合上述所有界，我们得到以概率$1 - \beta$：

$$
\parallel a - \widehat{a}{\parallel }_{2}^{2} \leq  \frac{{16nk}\sqrt{\ln \left( {1/\delta }\right) \ln \left( {\left| \mathcal{X}\right| /\beta }\right) }}{\epsilon }.
$$

让我们解释一下这个界。注意到$\parallel a - \widehat{a}{\parallel }_{2}^{2} = \mathop{\sum }\limits_{{i = 1}}^{k}{\left( {a}_{i} - {\widehat{a}}_{i}\right) }^{2}$，因此这是所有查询的平方误差之和的界。因此，该机制的每个查询的平均平方误差仅为：

$$
\frac{1}{k}\mathop{\sum }\limits_{{i = 1}}^{k}{\left( {a}_{i} - {\widehat{a}}_{i}\right) }^{2} \leq  \frac{{16n}\sqrt{\ln \left( {1/\delta }\right) \ln \left( {\left| \mathcal{X}\right| /\beta }\right) }}{\epsilon }.
$$

相比之下，私有乘法权重机制保证了$\mathop{\max }\limits_{{i \in  \left\lbrack  k\right\rbrack  }}\left| {{a}_{i} - {\widehat{a}}_{i}}\right|  \leq  \widetilde{O}\left( {\sqrt{n}\log {\left| \mathcal{X}\right| }^{1/4}/{\varepsilon }^{1/2}}\right)$，因此与投影拉普拉斯机制的均方误差保证相匹配，其界限为：$\widetilde{O}\left( {n\sqrt{\log \left| \mathcal{X}\right| }/\varepsilon }\right)$。然而，乘法权重机制（尤其是其隐私分析）比投影拉普拉斯机制复杂得多！特别是，$K$ - 投影拉普拉斯机制的私有部分仅仅是拉普拉斯机制本身，并且不需要查询之间的协调。有趣的是——事实证明，这是必要的——协调发生在投影阶段。由于投影是在后期处理中进行的，因此不会产生进一步的隐私损失；实际上，它可以由数据分析师自己（必要时在线）进行。

### 12.5 参考文献注释

数据隐私的局部模型源于随机响应，这一概念最早由华纳（Warner）在1965年提出 [84]。局部模型由卡西维斯瓦纳坦（Kasiviswanathan）等人 [52] 在学习的背景下进行了形式化，他们证明了局部模型中的私有学习等同于统计查询（SQ）模型中的非私有学习。古普塔（Gupta）等人 [38] 证明了在局部模型中可以发布的查询集恰好等于在SQ模型中可以进行不可知学习的查询集。

泛隐私（Pan - Privacy）由德沃尔（Dwork）等人 [27] 引入，并由米尔（Mir）等人 [62] 进一步探索。泛私有密度估计以及使用哈希的低内存变体出现在 [27] 中。

持续观察下的隐私由德沃尔（Dwork）等人 [26] 引入；我们用于持续观察下计数的算法以及误差下界均来自该论文。陈（Chan）等人 [11] 也给出了类似的算法。引理12.2中给出的拉普拉斯随机变量和的测度集中不等式的证明来自 [11]。

用于实现低平均误差的投影拉普拉斯机制由尼科洛夫（Nikolov）等人 [66] 提出，他们还针对任何查询类给出了（平均误差）查询发布问题的实例最优算法。这项工作扩展了由哈德特（Hardt）和塔尔瓦尔（Talwar） [45] 开创的关于差分隐私与几何之间联系的一系列研究，并由巴斯卡拉（Bhaskara）等人 [5] 和德沃尔（Dwork）等人 [30] 进一步拓展。

德沃尔（Dwork）、诺尔（Naor）和瓦德汉（Vadhan）证明了无状态和有状态差分隐私机制能够回答（具有非平凡误差）的查询数量之间存在指数级差距 [29]。得到的经验教训是——协调对于准确且私密地回答大量查询至关重要——这似乎排除了投影拉普拉斯机制中独立添加噪声的可能性。该算法的有状态性出现在投影步骤中，从而解决了这一矛盾。

## 13 反思

### 13.1 迈向隐私实践

差分隐私的设计初衷是针对互联网规模的数据集。类似于第8节中的重建攻击可以由一个多项式时间有界的对手在大小为$n$的数据库上仅询问$O\left( n\right)$个查询来实施。当$n$达到数亿级别，并且每个查询需要线性量级的计算时，即使查询可以并行化，这样的攻击也是不现实的。这一观察促成了差分隐私的早期发展：如果对手被限制在亚线性数量的计数查询范围内，那么每个查询添加$o\left( \sqrt{n}\right)$噪声——小于采样误差！——就足以保护隐私（推论3.21）。

在不破坏统计效用的前提下，差分隐私（Differential Privacy）能在多大程度上应用于较小的数据集，甚至是针对大型数据库中一小部分数据的定向攻击呢？首先，一项分析可能需要进行大量查询，其数量开始接近这个较小数据集的规模。其次，现在让$n$表示较小数据集或小型数据库的大小，让$k$表示查询的数量，当$n$较小时，数量级为$\sqrt{k}/n$的分数误差就难以忽略。第三，高级组合定理中的$\sqrt{\ln \left( {1/\delta }\right) }/\varepsilon$因子变得很重要。考虑到当噪声为$o\left( \sqrt{n}\right)$时的重构攻击，对于任意一组$k \approx  n$低敏感度查询而言，似乎没有多少操作空间。

有几条很有前景的研究路线可以解决这些问题。

查询误差并不能说明全部情况。以线性回归（Linear Regression）问题为例。输入是一组形式为(x,y)的带标签数据点，其中$x \in  {\mathbb{R}}^{d}$和$y \in  \mathbb{R}$，维度$d$为任意值。目标是在假设关系为线性的情况下，给定$x$，找到能“尽可能好地”“预测”$y$的$\theta  \in  {\mathbb{R}}^{d}$。如果目标仅仅是“解释”给定的数据集，那么差分隐私很可能会引入不可接受的误差。当然，简单地计算

$$
{\operatorname{argmin}}_{\theta }{\left| \mathop{\sum }\limits_{{i = 1}}^{n}\theta  \cdot  {x}_{i} - {y}_{i}\right| }^{2}
$$

并独立地向$\theta$的每个坐标添加适当缩放的拉普拉斯噪声（Laplace Noise）的特定算法，可能会产生与$\theta$有很大差异的$\widetilde{\theta }$。但如果目标是学习一个对未来未见过的输入(x,y)表现良好的预测器，那么可以使用一种稍有不同的计算方法来避免过拟合，并且私有系数向量和非私有系数向量之间（可能很大的）差异并不会转化为分类误差的差距！在模型拟合中也观察到了类似的现象。

少即是多。许多分析要求的比实际使用的更多。利用这一原理是“报告噪声最大值（Report Noisy Max）”方法的核心，在该方法中，以一次测量的精度“代价”，我们可以得知多个测量值中的最大值之一。通过要求“更少”（即不要求发布所有带噪声的测量值，而只要求得到最大值），我们可以获得“更多”（更高的精度）。隐私领域中一个常见的原则是尽量减少数据收集和报告。在这里，我们看到这一原则在必须披露的内容方面发挥作用，而不是在计算中必须使用的内容方面。

未领先时就退出。这是“提议 - 测试 - 发布（Propose - Test - Release）”方法背后的理念，在该方法中，我们以保护隐私的方式进行测试，以确定小噪声是否足以用于对给定数据集进行特定的预期计算。

具有依赖数据的精度界限的算法。这可以看作是“未领先时就退出”方法的推广。具有依赖数据的精度界限的算法可以在“优质”数据集上取得出色的结果，就像“提议 - 测试 - 发布”方法一样，并且随着数据集“质量”的下降，精度可以逐渐降低，这比“提议 - 测试 - 发布”方法有所改进。

利用“良好”的查询集。当以批量形式呈现（可能很大的）线性查询集时，通过分析查询矩阵的几何结构，有可能获得比独立回答这些查询时更高质量的答案${}^{1}$。

差分隐私的进一步松弛 我们已经看到，$\left( {\epsilon ,\delta }\right)$ - 差分隐私是差分隐私的一种有意义的松弛方式，它可以显著改善精度界限。此外，这种松弛对于这些改进可能是必不可少的。例如，提议 - 测试 - 发布算法只能为 $\delta  > 0$ 提供 $\left( {\varepsilon ,\delta }\right)$ - 差分隐私。那么，差分隐私的其他有意义的松弛方式呢？集中差分隐私就是这样一种松弛方式，它与 $\left( {\varepsilon ,\delta }\right)$ - 差分隐私不可比，并且允许更高的精度。粗略地说，它确保了大的隐私损失以非常小的概率发生；例如，对于所有 $k$，隐私损失 ${k\varepsilon }$ 的概率在 ${k}^{2}$ 中呈指数下降。相比之下，$\left( {\varepsilon ,\delta }\right)$ - 差分隐私与以概率 $\delta$ 出现无限的隐私损失是一致的；另一方面，隐私损失 ${2\varepsilon }$ 在集中差分隐私中可以以恒定概率发生，而在 $\left( {\varepsilon ,\delta }\right)$ - 差分隐私中，它只会以由 $\delta$ 界定的概率发生，我们通常认为这个概率在密码学意义上是很小的。

为什么我们会对这种松弛方式感到满意呢？答案在于组合下的行为。当一个人的数据参与到许多数据库和许多不同的计算中时，也许真正令人担忧的是多次暴露的综合威胁。这可以通过组合下的隐私来体现。集中差分隐私在组合下的行为与 $\left( {\varepsilon ,\delta }\right)$（和 $\left( {\varepsilon ,0}\right)$）差分隐私相同，同时允许更高的精度。

---

<!-- Footnote -->

${}^{1}$ 更准确地说，分析的对象是 $K = A{B}_{1}^{k}$，其中 $A$ 是查询矩阵，${B}_{1}^{k}$ 是 $k$ 维的 ${L}_{1}$ 球；请注意，当数据库只有一个元素时，$K$ 是答案空间中的可行区域。

<!-- Footnote -->

---

差分隐私还面临着一些文化方面的挑战。其中最显著的挑战之一是非算法思维。差分隐私是算法的一种属性。然而，许多处理数据的人从根本上以非算法的术语描述他们与数据的交互，例如，“首先，我查看数据”。类似地，数据清理通常也以非算法的术语来描述。如果数据相当丰富，并且分析人员积极主动，那么示例 7.3 中描述的子采样和聚合方法的“原始数据”应用为遵循指示的可信分析人员进行非算法交互提供了一条途径。一般来说，在高维和互联网规模的数据集上，非算法交互似乎不太常见。

$\varepsilon$ 呢？在示例 3.7 中，我们应用定理 3.20 得出结论，要以概率 $1 - {e}^{-{32}}$ 将累积终身隐私损失限制在 $\varepsilon  = 1$，在参与 10000 个数据库的情况下，每个数据库具有 $\left( {1/{801},0}\right)$ - 差分隐私就足够了。虽然 $k = {10},{000}$ 可能是一个高估，但对 $k$ 的依赖相当弱 $\left( \sqrt{k}\right)$，并且在最坏的情况下，这些界限是严格的，排除了在数据库的整个生命周期内每个数据库比 ${\varepsilon }_{0} = 1/{801}$ 更宽松的界限。在实践中，这一要求过于严格。

也许我们可以换个问题：固定$\varepsilon$，比如说$\varepsilon  = 1$或$\varepsilon  = 1/{10}$；现在来问：多个$\varepsilon$该如何分配呢？允许每次查询有$\varepsilon$的隐私损失太宽松，而允许数据库整个生命周期有$\varepsilon$的损失又太严格。介于两者之间的情况，比如每项研究有$\varepsilon$的损失或每个研究人员有$\varepsilon$的损失，可能是合理的，尽管这会引出谁是“研究人员”以及什么构成“研究”的问题。与目前从数据飞地到保密合同等做法相比，这能为防止意外和故意的隐私泄露提供更充分的保护。

另一个提议的规定性没那么强。该提议借鉴了第二代减少环境退化的监管方法，特别是像有毒物质排放清单这类污染排放登记制度，这些制度已被证明能通过提高透明度来鼓励更好的做法。也许在私人数据分析中也能产生类似的效果：一个ε登记系统（Epsilon Registry），它描述数据的使用情况、隐私保护的粒度、单位时间内隐私损失的“消耗率”，以及在数据停用前允许的总隐私损失上限，再加上对无限（或非常大）损失的经济处罚，能够引发创新和竞争，让更多的研究人员和隐私专家发挥才能、投入资源来寻找差分隐私算法。

### 13.2 差分隐私视角

一本在线词源词典将“统计学”（statistics）这个词在18世纪最初的含义描述为“处理有关一个国家或社区状况数据的科学”。这与差分隐私在数据泄露方面的情况相呼应：如果少数个体数据的存在与否改变了分析结果，那么从某种意义上说，这个结果是“关于”这少数个体的，而不是在描述整个社区的状况。换句话说，数据的小扰动稳定性既是差分隐私的标志，也是“统计学”这一术语常见概念的本质。差分隐私由稳定性实现（第7节），并确保稳定性（根据定义）。从某种意义上说，它迫使所有查询本质上都是统计性的。由于稳定性也越来越被认为是可学习性的一个关键充要条件，我们发现可学习性、差分隐私和稳定性之间存在着一种诱人的道德等价关系。

考虑到这一点，差分隐私也是实现隐私之外其他目标的一种手段就不足为奇了，实际上我们在第10节的博弈论中已经看到了这一点。差分隐私的强大之处在于它易于组合。正如组合允许我们用较小的差分隐私构建块来构建复杂的差分隐私算法一样，它也为构建用于复杂分析任务的稳定算法提供了一种编程语言。例如，考虑引出一组投标人的估值，并利用这些估值为一批待售商品定价的问题。简单来说，瓦尔拉斯均衡价格（Walrasian equilibrium prices）是这样一种价格，即在给定这些价格的情况下，每个个体都能同时购买到他们最喜欢的商品组合，同时确保每种商品的需求恰好等于供给。乍一看，那么简单地计算这些价格，并根据这些价格为每个人分配他们最喜欢的商品组合，似乎会产生一种机制，在这种机制中，参与者会有动力如实说出他们的估值函数——因为任何参与者怎么可能比得到他们最喜欢的商品组合做得更好呢？然而，这个论点不成立——因为在瓦尔拉斯均衡中，参与者根据价格得到他们最喜欢的商品组合，但价格是根据报告的估值计算出来的，所以一个勤勉但不诚实的参与者可能会通过操纵计算出的价格来获利。然而，如果使用差分隐私算法来计算均衡价格，这个问题就能得到解决（并产生一种近似诚实的机制）——正是因为单个参与者对计算出的价格分布几乎没有影响。请注意，这个应用之所以成为可能是因为使用了差分隐私的工具，但它与隐私问题完全无关。更一般地说，这种联系更为根本：使用具有差分隐私所保证的稳定性属性的算法来计算各种均衡，会产生实现这些均衡结果的近似诚实机制。

差分隐私（Differential privacy）还有助于确保自适应数据分析中的泛化性。适应性是指所提出的问题和所检验的假设依赖于早期问题的结果。泛化性是指对数据集进行计算或测试的结果接近数据采样分布的真实情况。众所周知，在固定数据集上用精确的经验值回答查询的简单范式，即使在有限的自适应提问情况下也无法实现泛化。值得注意的是，使用差分隐私进行回答不仅能确保隐私性，而且在高概率下，即使对于指数级数量的自适应选择查询，它也能确保泛化性。因此，利用差分隐私技术有意引入噪声，对传统科学探究的有效性具有深远且有前景的意义。

## 附录

## A 高斯机制

设 $f : {\mathbb{N}}^{\left| \mathcal{X}\right| } \rightarrow  {\mathbb{R}}^{d}$ 为任意 $d$ 维函数，并将其 ${\ell }_{2}$ 敏感度定义为 ${\Delta }_{2}f = \mathop{\max }\limits_{{\text{adjacent }x,y}}\parallel f\left( x\right)  - f\left( y\right) {\parallel }_{2}$。参数为 $\sigma$ 的高斯机制会将按 $\mathcal{N}\left( {0,{\sigma }^{2}}\right)$ 缩放的噪声添加到输出的每个 $d$ 分量中。

定理 A.1。设 $\varepsilon  \in  \left( {0,1}\right)$ 为任意值。对于 ${c}^{2} > 2\ln \left( {{1.25}/\delta }\right)$，参数为 $\sigma  \geq  c{\Delta }_{2}f/\varepsilon$ 的高斯机制具有 $\left( {\varepsilon ,\delta }\right)$ -差分隐私性。

证明。存在一个数据库 $D$ 和一个查询 $f$，该机制将返回 $f\left( D\right)  + \eta$，其中噪声呈正态分布。我们正在添加噪声 $\mathcal{N}\left( {0,{\sigma }^{2}}\right)$。目前，假设我们讨论的是实值函数，因此

$$
{\Delta f} = {\Delta }_{1}f = {\Delta }_{2}f
$$

我们正在研究

$$
\left| {\ln \frac{{e}^{\left( {-1/2{\sigma }^{2}}\right) {x}^{2}}}{{e}^{\left( {-1/2{\sigma }^{2}}\right) {\left( x + \Delta f\right) }^{2}}}}\right| . \tag{A.1}
$$

我们正在研究在数据库为$D$的情况下，观察到一个输出的概率，该输出在$D$下的出现概率与在相邻数据库${D}^{\prime }$下的出现概率有很大不同，其中概率空间是噪声生成算法。上述比率的分子描述了数据库为$D$时看到$f\left( D\right)  + x$的概率，分母对应于数据库为${D}^{\prime }$时看到相同值的概率。这是一个概率比率，因此它始终为正，但该比率的对数可能为负。我们感兴趣的随机变量——隐私损失——是

$$
\ln \frac{{e}^{\left( {-1/2{\sigma }^{2}}\right) {x}^{2}}}{{e}^{\left( {-1/2{\sigma }^{2}}\right) {\left( x + \Delta f\right) }^{2}}}
$$

并且我们正在研究它的绝对值。

$$
\left| {\ln \frac{{e}^{\left( {-1/2{\sigma }^{2}}\right) {x}^{2}}}{{e}^{\left( {-1/2{\sigma }^{2}}\right) {\left( x + \Delta f\right) }^{2}}}}\right|  = \left| {\ln {e}^{\left( {-1/2{\sigma }^{2}}\right) \left\lbrack  {{x}^{2} - {\left( x + \Delta f\right) }^{2}}\right\rbrack  }}\right| 
$$

$$
 = \left| {-\frac{1}{2{\sigma }^{2}}\left\lbrack  {{x}^{2} - \left( {{x}^{2} + {2x\Delta f} + \Delta {f}^{2}}\right) }\right\rbrack  }\right| 
$$

$$
 = \left| {\frac{1}{2{\sigma }^{2}}\left( {{2x\Delta f} + {\left( \Delta f\right) }^{2}}\right) }\right| . \tag{A.2}
$$

只要$x < {\sigma }^{2}\varepsilon /{\Delta f} - {\Delta f}/2$，这个量就以$\varepsilon$为界。为了确保隐私损失以至少$1 - \delta$的概率被$\varepsilon$界定，我们要求

$$
\Pr \left\lbrack  {\left| x\right|  \geq  {\sigma }^{2}\varepsilon /{\Delta f} - {\Delta f}/2}\right\rbrack   < \delta ,
$$

并且因为我们关注$\left| x\right|$，我们将找到$\sigma$使得

$$
\Pr \left\lbrack  {x \geq  {\sigma }^{2}\varepsilon /{\Delta f} - {\Delta f}/2}\right\rbrack   < \delta /2.
$$

我们始终假设$\varepsilon  \leq  1 \leq  {\Delta f}$。

我们将使用尾界

$$
\Pr \left\lbrack  {x > t}\right\rbrack   \leq  \frac{\sigma }{\sqrt{2\pi }}{e}^{-{t}^{2}/2{\sigma }^{2}}.
$$

我们要求：

$$
\frac{\sigma }{\sqrt{2\pi }}\frac{1}{t}{e}^{-{t}^{2}/2{\sigma }^{2}} < \delta /2
$$

$$
 \Leftrightarrow  \sigma \frac{1}{t}{e}^{-{t}^{2}/2{\sigma }^{2}} < \sqrt{2\pi }\delta /2
$$

$$
 \Leftrightarrow  \frac{t}{\sigma }{e}^{{t}^{2}/2{\sigma }^{2}} > 2/\sqrt{2\pi }\delta 
$$

$$
 \Leftrightarrow  \ln \left( {t/\sigma }\right)  + {t}^{2}/2{\sigma }^{2} > \ln \left( {2/\sqrt{2\pi }\delta }\right) \text{.}
$$

取$t = {\sigma }^{2}\varepsilon /{\Delta f} - {\Delta f}/2$，我们得到

$$
\ln \left( {\left( {{\sigma }^{2}\varepsilon /{\Delta f} - {\Delta f}/2}\right) /\sigma }\right)  + {\left( {\sigma }^{2}\varepsilon /\Delta f - \Delta f/2\right) }^{2}/2{\sigma }^{2} > \ln \left( {2/\sqrt{2\pi }\delta }\right) 
$$

$$
 = \ln \left( {\sqrt{\frac{2}{\pi }}\frac{1}{\delta }}\right) .
$$

让我们记$\sigma  = {c\Delta f}/\varepsilon$；我们希望界定$c$。我们首先找出第一项非负的条件。

$$
\frac{1}{\sigma }\left( {{\sigma }^{2}\frac{\varepsilon }{\Delta f} - \frac{\Delta f}{2}}\right)  = \frac{1}{\sigma }\left\lbrack  {\left( {{c}^{2}\frac{{\left( \Delta f\right) }^{2}}{{\varepsilon }^{2}}}\right) \frac{\varepsilon }{\Delta f} - \frac{\Delta f}{2}}\right\rbrack  
$$

$$
 = \frac{1}{\sigma }\left\lbrack  {{c}^{2}\left( \frac{\Delta f}{\varepsilon }\right)  - \frac{\Delta f}{2}}\right\rbrack  
$$

$$
 = \frac{\varepsilon }{c\Delta f}\left\lbrack  {{c}^{2}\left( \frac{\Delta f}{\varepsilon }\right)  - \frac{\Delta f}{2}}\right\rbrack  
$$

$$
 = c - \frac{\varepsilon }{2c}.
$$

由于$\varepsilon  \leq  1$和$c \geq  1$，我们有$c - \varepsilon /\left( {2c}\right)  \geq  c - 1/2$。因此，只要$c \geq  3/2$，就有$\ln \left( {\frac{1}{\sigma }\left( {{\sigma }^{2}\frac{\varepsilon }{\Delta f} - }\right. }\right.$ $\left. \left. \frac{\Delta f}{2}\right) \right)  > 0$。因此，我们可以专注于${t}^{2}/{\sigma }^{2}$项。

$$
{\left( \frac{1}{2{\sigma }^{2}}\frac{{\sigma }^{2}\varepsilon }{\Delta f} - \frac{\Delta f}{2}\right) }^{2} = \frac{1}{2{\sigma }^{2}}{\left\lbrack  \Delta f\left( \frac{{c}^{2}}{\varepsilon } - \frac{1}{2}\right) \right\rbrack  }^{2}
$$

$$
 = {\left\lbrack  {\left( \Delta f\right) }^{2}\left( \frac{{c}^{2}}{\varepsilon } - \frac{1}{2}\right) \right\rbrack  }^{2}\left\lbrack  \frac{{\varepsilon }^{2}}{{c}^{2}{\left( \Delta f\right) }^{2}}\right\rbrack  \frac{1}{2}
$$

$$
 = \frac{1}{2}{\left( \frac{{c}^{2}}{\varepsilon } - \frac{1}{2}\right) }^{2}\frac{{\varepsilon }^{2}}{{c}^{2}}
$$

$$
 = \frac{1}{2}\left( {{c}^{2} - \varepsilon  + {\varepsilon }^{2}/4{c}^{2}}\right) .
$$

由于$\varepsilon  \leq  1$，在我们考虑的范围$\left( {c \geq  3/2}\right)$内，$\left( {{c}^{2} - \varepsilon  + {\varepsilon }^{2}/4{c}^{2}}\right)$关于$c$的导数为正，所以${c}^{2} - \varepsilon  + {\varepsilon }^{2}/4{c}^{2} \geq  {c}^{2} - 8/9$，并且确保

$$
{c}^{2} - 8/9 > 2\ln \left( {\sqrt{\frac{2}{\pi }}\frac{1}{\delta }}\right) .
$$

换句话说，我们需要

$$
{c}^{2} > 2\ln \left( \sqrt{2/\pi }\right)  + 2\ln \left( {1/\delta }\right)  + \ln \left( {e}^{8/9}\right)  = \ln \left( {2/\pi }\right)  + \ln \left( {e}^{8/9}\right)  + 2\ln \left( {1/\delta }\right) ,
$$

由于$\left( {2/\pi }\right) {e}^{8/9} < {1.55}$，只要${c}^{2} > 2\ln \left( {{1.25}/\delta }\right)$，该条件就满足。

让我们将$\mathbb{R}$划分为$\mathbb{R} = {R}_{1} \cup  {R}_{2}$，其中${R}_{1} = \{ x \in  \mathbb{R} : \left| x\right|  \leq$ ${c\Delta f}/\varepsilon \}$且${R}_{2} = \{ x \in  \mathbb{R} : \left| x\right|  > {c\Delta f}/\varepsilon \}$。固定任何子集$S \subseteq  \mathbb{R}$，并定义

$$
{S}_{1} = \left\{  {f\left( x\right)  + x \mid  x \in  {R}_{1}}\right\}  
$$

$$
{S}_{2} = \left\{  {f\left( x\right)  + x \mid  x \in  {R}_{2}}\right\}  .
$$

我们有

$$
\mathop{\Pr }\limits_{{x \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right) }}\left\lbrack  {f\left( x\right)  + x \in  S}\right\rbrack   = \mathop{\Pr }\limits_{{x \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right) }}\left\lbrack  {f\left( x\right)  + x \in  {S}_{1}}\right\rbrack  
$$

$$
 + \mathop{\Pr }\limits_{{x \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right) }}\left\lbrack  {f\left( x\right)  + x \in  {S}_{2}}\right\rbrack  
$$

$$
 \leq  \mathop{\Pr }\limits_{{x \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right) }}\left\lbrack  {f\left( x\right)  + x \in  {S}_{1}}\right\rbrack   + \delta 
$$

$$
 \leq  {e}^{\varepsilon }\left( {\mathop{\Pr }\limits_{{x \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right) }}\left\lbrack  {f\left( y\right)  + x \in  {S}_{1}}\right\rbrack  }\right)  + \delta ,
$$

为一维高斯机制产生$\left( {\varepsilon ,\delta }\right)$ -差分隐私。

高维情况。为了将其扩展到${R}^{m}$ 中的函数，定义${\Delta f} =$ ${\Delta }_{2}f$ 。现在我们可以使用欧几里得范数重复上述论证。设$v$ 是满足$\parallel v\parallel  \leq  {\Delta f}$ 的任意向量。对于固定的一对数据库$x,y$ ，我们关注$v = f\left( x\right)  - f\left( y\right)$ ，因为这是我们的噪声必须掩盖的内容。与一维情况一样，我们寻求$\sigma$ 满足的条件，使得隐私损失

$$
\left| {\ln \frac{{e}^{\left( {-1/2{\sigma }^{2}}\right) \parallel x - \mu {\parallel }^{2}}}{{e}^{\left( {-1/2{\sigma }^{2}}\right) \parallel x + v - \mu {\parallel }^{2}}}}\right| 
$$

以$\varepsilon$ 为界；这里$x$ 从$\mathcal{N}\left( {0,\sum }\right)$ 中选取，其中$\left( \sum \right)$ 是一个对角矩阵，其元素为${\sigma }^{2}$ ，因此$\mu  = \left( {0,\ldots ,0}\right)$ 。

$$
\left| {\ln \frac{{e}^{\left( {-1/2{\sigma }^{2}}\right) \parallel x - \mu {\parallel }^{2}}}{{e}^{\left( {-1/2{\sigma }^{2}}\right) \parallel x + v - \mu {\parallel }^{2}}}}\right|  = \left| {\ln {e}^{\left( {-1/2{\sigma }^{2}}\right) \left\lbrack  {\parallel x - \mu {\parallel }^{2} - \parallel x + v - \mu {\parallel }^{2}}\right\rbrack  }}\right| 
$$

$$
 = \left| {\frac{1}{2{\sigma }^{2}}\left( {\parallel x{\parallel }^{2} - \parallel x + v{\parallel }^{2}}\right) )}\right| .
$$

我们将利用球对称正态分布与其构成正态分布所选取的正交基无关这一事实，因此我们可以在与$v$ 对齐的基下进行研究。固定这样一个基${b}_{1},\ldots ,{b}_{m}$ ，通过首先抽取带符号的长度${\lambda }_{i} \sim  \mathcal{N}\left( {0,{\sigma }^{2}}\right)$ （对于$i \in  \left\lbrack  m\right\rbrack$ ），然后定义${x}^{\left\lbrack  i\right\rbrack  } = {\lambda }_{i}{b}_{i}$ ，最后令$x = \mathop{\sum }\limits_{{i = 1}}^{m}{x}^{\left\lbrack  i\right\rbrack  }$ 来抽取$x$ 。不失一般性地假设${b}_{1}$ 与$v$ 平行。我们关注$\left| {\parallel x{\parallel }^{2} - \parallel x + v{\parallel }^{2}}\right|$ 。

考虑以$v + {x}^{\left\lbrack  1\right\rbrack  }$ 为底边且边$\mathop{\sum }\limits_{{i = 2}}^{m}{x}^{\left\lbrack  i\right\rbrack  }$ 与$v$ 正交的直角三角形。该三角形的斜边为$x + v$ 。

$$
\parallel x + v{\parallel }^{2} = {\begin{Vmatrix}v + {x}^{\left\lbrack  1\right\rbrack  }\end{Vmatrix}}^{2} + \mathop{\sum }\limits_{{i = 2}}^{m}{\begin{Vmatrix}{x}^{\left\lbrack  i\right\rbrack  }\end{Vmatrix}}^{2}
$$

$$
\parallel x{\parallel }^{2} = \mathop{\sum }\limits_{{i = 1}}^{m}{\begin{Vmatrix}{x}^{\left\lbrack  i\right\rbrack  }\end{Vmatrix}}^{2}.
$$

由于$v$ 与${x}^{\left\lbrack  1\right\rbrack  }$ 平行，我们有${\begin{Vmatrix}v + {x}^{\left\lbrack  1\right\rbrack  }\end{Vmatrix}}^{2} = {\left( \parallel v\parallel  + {\lambda }_{1}\right) }^{2}$ 。因此，$\parallel x + v{\parallel }^{2} - \parallel x{\parallel }^{2} = \parallel v{\parallel }^{2} + 2{\lambda }_{1} \cdot  \parallel v\parallel$ 。回想$\parallel v\parallel  \leq  {\Delta f}$ ，并且$\lambda  \sim$ $\mathcal{N}\left( {0,\sigma }\right)$ ，所以现在我们恰好回到了一维情况，在方程A.2)中用${\lambda }_{1}$ 代替$x$ ：

$$
\left| {\frac{1}{2{\sigma }^{2}}\left( {\parallel x{\parallel }^{2} - \parallel x + v{\parallel }^{2}}\right) )}\right|  \leq  \left| {\frac{1}{2{\sigma }^{2}}\left( {2{\lambda }_{1}{\Delta f} - {\left( \Delta f\right) }^{2}}\right) }\right| 
$$

其余的论证过程如上所述。

高维情形的论证凸显了$\left( {\varepsilon ,\delta }\right)$ -差分隐私（$\left( {\varepsilon ,\delta }\right)$ -differential privacy）存在但$\left( {\varepsilon ,0}\right)$ -差分隐私（$\left( {\varepsilon ,0}\right)$ -differential privacy）不存在的一个弱点。固定一个数据库$x$ 。在$\left( {\varepsilon ,0}\right)$ -差分隐私的情形下，不可区分性保证对所有相邻数据库同时成立。在$\left( {\varepsilon ,\delta }\right)$ -差分隐私的情形下，不可区分性仅“前瞻性地”成立，即对于任何与$x$ 相邻的固定数据库$y$ ，该机制使对手能够区分$x$ 和$y$ 的概率很小。在上述证明中，这表现为我们固定了$v = f\left( x\right)  - f\left( y\right)$ 这一事实；我们不必同时论证$v$ 的所有可能方向，而且实际上我们也无法做到，因为一旦我们固定了噪声向量$x \sim  \mathcal{N}\left( {0,\sum }\right)$ ，使得在数据库$x$ 上的输出为$o = f\left( x\right)  + x$ ，可能存在一个相邻的数据库$y$ ，使得当数据库为$y$ 时输出$o = f\left( x\right)  + x$ 的可能性比在数据库$x$ 上大得多。

### A.1 参考文献注释

定理A.1是一个业内常识，最初由文献[23]的作者观察到。对非球形高斯噪声的推广出现在文献[66]中。

## B $\left( {\varepsilon ,\delta }\right)$ -差分隐私（$\left( {\varepsilon ,\delta }\right)$ -DP）的组合定理

### B.1 定理3.16的扩展

定理B.1。设${T}_{1}\left( D\right)  : D \mapsto  {T}_{1}\left( D\right)  \in  {\mathcal{C}}_{1}$ 是一个$\left( {\epsilon ,\delta }\right)$ -差分隐私（$\left( {\epsilon ,\delta }\right)$ -d.p.）函数，并且对于任何${s}_{1} \in  {\mathcal{C}}_{1},{T}_{2}\left( {D,{s}_{1}}\right)  : \left( {D,{s}_{1}}\right)  \mapsto  {T}_{2}\left( {D,{s}_{1}}\right)  \in  {\mathcal{C}}_{2}$ ，在给定第二个输入${s}_{1}$ 的情况下，${s}_{1} \in  {\mathcal{C}}_{1},{T}_{2}\left( {D,{s}_{1}}\right)  : \left( {D,{s}_{1}}\right)  \mapsto  {T}_{2}\left( {D,{s}_{1}}\right)  \in  {\mathcal{C}}_{2}$ 是一个$\left( {\epsilon ,\delta }\right)$ -差分隐私（$\left( {\epsilon ,\delta }\right)$ -d.p.）函数。然后我们证明，对于任何相邻的$D,{D}^{\prime }$ ，对于任何$S \subseteq  {\mathcal{C}}_{2} \times  {\mathcal{C}}_{1}$ ，使用我们论文中的符号，我们有

$$
P\left( {\left( {{T}_{2},{T}_{1}}\right)  \in  S}\right)  \leq  {e}^{2\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{T}_{1}}\right)  \in  S}\right)  + {2\delta }. \tag{B.1}
$$

证明。对于任何${C}_{1} \subseteq  {\mathcal{C}}_{1}$ ，定义

$$
\mu \left( {C}_{1}\right)  = {\left( P\left( {T}_{1} \in  {C}_{1}\right)  - {e}^{\epsilon }{P}^{\prime }\left( {T}_{1} \in  {C}_{1}\right) \right) }_{ + },
$$

那么$\mu$ 是${\mathcal{C}}_{1}$ 和$\mu \left( {\mathcal{C}}_{1}\right)  \leq  \delta$ 上的一个测度，因为${T}_{1}$ 是$\left( {\epsilon ,\delta }\right)$ -差分隐私（$\left( {\epsilon ,\delta }\right)$ -d.p.）的。因此，对于所有的${s}_{1} \in  {\mathcal{C}}_{1}$ ，我们有

$$
P\left( {{T}_{1} \in  d{s}_{1}}\right)  \leq  {e}^{\epsilon }{P}^{\prime }\left( {{T}_{1} \in  d{s}_{1}}\right)  + \mu \left( {d{s}_{1}}\right) . \tag{B.2}
$$

还要注意，根据$\left( {\epsilon ,\delta }\right)$ -差分隐私（$\left( {\epsilon ,\delta }\right)$ -d.p.）的定义，对于任何${s}_{1} \in  {\mathcal{C}}_{1}$ ，

$$
P\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right)  \leq  \left( {{e}^{\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right)  + \delta }\right)  \land  1
$$

$$
 \leq  \left( {{e}^{\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) }\right)  \land  1 + \delta . \tag{B.3}
$$

然后（B.2）和（B.3）推出（B.1）：

$$
P\left( {\left( {{T}_{2},{T}_{1}}\right)  \in  S}\right)  \leq  {\int }_{{S}_{1}}P\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) P\left( {{T}_{1} \in  d{s}_{1}}\right) 
$$

$$
 \leq  {\int }_{{S}_{1}}\left( {\left( {{e}^{\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) }\right)  \land  1 + \delta }\right) P\left( {{T}_{1} \in  d{s}_{1}}\right) 
$$

$$
 \leq  {\int }_{{S}_{1}}\left( {\left( {{e}^{\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) }\right)  \land  1}\right) P\left( {{T}_{1} \in  d{s}_{1}}\right)  + \delta 
$$

$$
 \leq  {\int }_{{S}_{1}}\left( {\left( {{e}^{\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) }\right)  \land  1}\right) 
$$

$$
 \times  \left( {{e}^{\epsilon }{P}^{\prime }\left( {{T}_{1} \in  d{s}_{1}}\right)  + \mu \left( {d{s}_{1}}\right) }\right)  + \delta 
$$

$$
 \leq  {e}^{2\epsilon }{\int }_{{S}_{1}}{P}^{\prime }\left( {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right) {P}^{\prime }\left( {{T}_{1} \in  d{s}_{1}}\right)  + \mu \left( {S}_{1}\right)  + \delta 
$$

$$
 \leq  {e}^{2\epsilon }{P}^{\prime }\left( {\left( {{T}_{2},{T}_{1}}\right)  \in  S}\right)  + {2\delta }. \tag{B.4}
$$

在上述方程中，${S}_{1}$ 表示 $S$ 在 ${\mathcal{C}}_{1}$ 上的投影。事件 $\left\{  {\left( {{T}_{2},{s}_{1}}\right)  \in  S}\right\}$ 指的是 $\left\{  {\left( {{T}_{2}\left( {D,{s}_{1}}\right) ,{s}_{1}}\right)  \in  S}\right\}$（或 $\left\{  \left( {{T}_{2}\left( {{D}^{\prime },{s}_{1}}\right) ,{s}_{1}) \in  S}\right) \right\}$）。

使用归纳法，我们有：

推论 B.2（$\left( {\epsilon ,\delta }\right)$ -差分隐私（$\left( {\epsilon ,\delta }\right)$ -d.p.）算法的一般组合定理）。设 ${T}_{1} : D \mapsto  {T}_{1}\left( D\right)$ 是 $\left( {\epsilon ,\delta }\right)$ -差分隐私的，并且对于 $k \geq  2,{T}_{k}$：$\left( {D,{s}_{1},\ldots ,{s}_{k - 1}}\right)  \mapsto  {T}_{k}\left( {D,{s}_{1},\ldots ,{s}_{k - 1}}\right)  \in  {\mathcal{C}}_{k}$ 是 $\left( {\epsilon ,\delta }\right)$ -差分隐私的，对于所有给定的 $\left( {{s}_{k - 1},\ldots ,{s}_{1}}\right)  \in  {\bigotimes }_{j = 1}^{k - 1}{\mathcal{C}}_{j}$。那么对于所有相邻的 $D,{D}^{\prime }$ 和所有 $S \subseteq  {\bigotimes }_{j = 1}^{k}{\mathcal{C}}_{j}$

$$
P\left( {\left( {{T}_{1},\ldots ,{T}_{k}}\right)  \in  S}\right)  \leq  {e}^{k\epsilon }{P}^{\prime }\left( {\left( {{T}_{1},\ldots ,{T}_{k}}\right)  \in  S}\right)  + {k\delta }.
$$

致谢

我们要感谢许多人对本书早期草稿提供了细致的评论和修正，其中包括维塔利·费尔德曼（Vitaly Feldman）、贾斯汀·许（Justin Hsu）、西蒙·加芬克尔（Simson Garfinkel）、卡特里娜·利格特（Katrina Ligett）、董琳（Dong Lin）、大卫·帕克斯（David Parkes）、瑞安·罗杰斯（Ryan Rogers）、盖伊·罗斯布卢姆（Guy Rothblum）、伊恩·施穆特（Ian Schmutte）、乔恩·厄尔曼（Jon Ullman）、萨利尔·瓦德汉（Salil Vadhan）、史蒂文·吴志伟（Zhiwei Steven Wu）以及匿名评审人员。本书曾用于萨利尔·瓦德汉和乔恩·厄尔曼教授的课程，他们的学生也提供了细致的反馈。本书还得益于与许多其他同事的交流，其中包括莫里茨·哈德特（Moritz Hardt）、伊利亚·米罗诺夫（Ilya Mironov）、萨肖·尼科洛夫（Sasho Nikolov）、科比·尼斯姆（Kobbi Nissim）、马勒什·派（Mallesh Pai）、本杰明·皮尔斯（Benjamin Pierce）、亚当·史密斯（Adam Smith）、阿布拉迪普·塔库尔塔（Abhradeep Thakurta）、阿比舍克·鲍米克（Abhishek Bhowmick）、库纳尔·塔尔瓦尔（Kunal Talwar）和张立（Li Zhang）。我们感谢马杜·苏丹（Madhu Sudan）提议撰写这本专著。

参考文献

[1] S. 阿罗拉（S. Arora）、E. 哈赞（E. Hazan）和 S. 卡莱（S. Kale）。乘法权重更新方法：一种元算法及其应用。《计算理论》，8(1):121 - 164，2012 年。

[2] M.-F. 巴尔坎（M.-F. Balcan）、A. 布卢姆（A. Blum）、J. D. 哈特林（J. D. Hartline）和 Y. 曼苏尔（Y. Mansour）。通过机器学习进行机制设计。见《计算机科学基础》，2005 年第 46 届 IEEE 年度研讨会（FOCS 2005），第 605 - 614 页。IEEE，2005 年。

[3] A. 贝梅尔（A. Beimel）、S. P. 卡西维斯瓦纳坦（S. P. Kasiviswanathan）和 K. 尼斯姆（K. Nissim）。隐私学习和隐私数据发布的样本复杂度界限。见《密码学理论》，第 437 - 454 页。施普林格出版社，2010 年。

[4] A. 贝梅尔（A. Beimel）、K. 尼斯姆（K. Nissim）和 U. 施泰默（U. Stemmer）。刻画隐私学习者的样本复杂度。见《理论计算机科学创新会议论文集》，第 97 - 110 页。美国计算机协会，2013 年。

[5] A. 巴斯卡拉（A. Bhaskara）、D. 达杜什（D. Dadush）、R. 克里希纳斯瓦米（R. Krishnaswamy）和 K. 塔尔瓦尔（K. Talwar）。线性查询的无条件差分隐私机制。见 H. J. 卡洛夫（H. J. Karloff）和 T. 皮塔西（T. Pitassi）编，《计算理论研讨会会议录》，计算理论研讨会，美国纽约州纽约市，2012 年 5 月 19 - 22 日，第 1269 - 1284 页。2012 年。

[6] A. 布卢姆（A. Blum）、C. 德沃克（C. Dwork）、F. 麦克谢里（F. McSherry）和 K. 尼斯姆（K. Nissim）。实用隐私：SuLQ 框架。见陈力（Chen Li）主编，《数据库系统原理》，第 128 - 138 页。美国计算机协会（ACM），2005 年。

[7] A. 布卢姆（A. Blum）、C. 德沃克（C. Dwork）、F. 麦克谢里（F. McSherry）和 K. 尼斯姆（K. Nissim）。实用隐私：sulq 框架。见《数据库系统原理》。2005 年。

[8] A. 布卢姆（A. Blum）、K. 利格特（K. Ligett）和 A. 罗斯（A. Roth）。一种用于非交互式数据库隐私的学习理论方法。见辛西娅·德沃克（Cynthia Dwork）主编，《计算理论研讨会》，第 609 - 618 页。美国计算机协会（Association for Computing Machinery），2008 年。

[9] A. 布卢姆（A. Blum）和 Y. 蒙苏尔（Y. Monsour）。学习、后悔最小化与均衡，2007 年。

[10] J. L. 卡斯蒂（J. L. Casti）。《五大黄金法则：20 世纪数学的伟大理论及其重要性》。威利出版社（Wiley），1996 年。

[11] T. H. 休伯特·陈（T. H. Hubert Chan）、E. 施（E. Shi）和 D. 宋（D. Song）。统计数据的私密且持续发布。见《自动机、语言与程序设计》，第 405 - 417 页。施普林格出版社（Springer），2010 年。

[12] K. 乔杜里（K. Chaudhuri）和 D. 许（D. Hsu）。差分隐私学习的样本复杂度界限。见《年度学习理论会议论文集（COLT 2011）》。2011 年。

[13] K. 乔杜里（K. Chaudhuri）、C. 蒙特莱奥尼（C. Monteleoni）和 A. D. 萨尔瓦特（A. D. Sarwate）。差分隐私经验风险最小化。《机器学习研究杂志：JMLR》，12:1069，2011 年。

[14] K. 乔杜里（K. Chaudhuri）、A. 萨尔瓦特（A. Sarwate）和 K. 辛哈（K. Sinha）。近乎最优的差分隐私主成分分析。见《神经信息处理系统进展 25》，第 998 - 1006 页。2012 年。

[15] Y. 陈（Y. Chen）、S. 崇（S. Chong）、I. A. 卡什（I. A. Kash）、T. 莫兰（T. Moran）和 S. P. 瓦德汉（S. P. Vadhan）。重视隐私的代理的诚实机制。美国计算机协会电子商务会议，2013 年。

[16] P. 丹德卡尔（P. Dandekar）、N. 法瓦兹（N. Fawaz）和 S. 约安尼迪斯（S. Ioannidis）。推荐系统的隐私拍卖。见《互联网与网络经济学》，第 309 - 322 页。施普林格出版社（Springer），2012 年。

[17] A. 德（A. De）。差分隐私的下界。见《密码学理论会议》，第 321 - 338 页。2012 年。

[18] I. 迪努尔（I. Dinur）和 K. 尼斯姆（K. Nissim）。在保护隐私的同时披露信息。见《美国计算机协会 SIGACT - SIGMOD - SIGART 数据库系统原理研讨会论文集》，第 202 - 210 页。2003 年。

[19] J. C. 杜奇（J. C. Duchi）、M. I. 乔丹（M. I. Jordan）和 M. J. 温赖特（M. J. Wainwright）。局部隐私与统计极小极大率。预印本 arXiv:1302.3203，2013 年。

[20] C. 德沃克（C. Dwork）。差分隐私。见《自动机、语言与程序设计国际学术讨论会论文集（ICALP）(2)》，第 1 - 12 页。2006 年。

[21] C. 德沃克（C. Dwork）、K. 肯塔帕迪（K. Kenthapadi）、F. 麦克谢里（F. McSherry）、I. 米罗诺夫（I. Mironov）和 M. 纳奥尔（M. Naor）。我们的数据，我们自己：通过分布式噪声生成实现隐私。见 EURO - ${CRYPT}$，第 ${486} - {503.2006}$ 页。

[22] C. Dwork和J. Lei。差分隐私与鲁棒统计。见《2009年美国计算机协会计算理论研讨会（STOC）会议录》。2009年。

[23] C. Dwork、F. McSherry、K. Nissim和A. Smith。在隐私数据分析中根据敏感度校准噪声。见《2006年密码学理论会议》，第265 - 284页。2006年。

[24] C. Dwork、F. McSherry和K. Talwar。隐私的代价与lp解码的极限。见《美国计算机协会计算理论研讨会会议录》，第85 - 94页。2007年。

[25] C. Dwork和M. Naor。统计数据库中防止信息泄露的困难或差分隐私的必要性。《隐私与保密期刊》，2010年。

[26] C. Dwork、M. Naor、T. Pitassi和G. N. Rothblum。持续观察下的差分隐私。见《美国计算机协会计算理论研讨会会议录》，第715 - 724页。美国计算机协会，2010年。

[27] C. Dwork、M. Naor、T. Pitassi、G. N. Rothblum和Sergey Yekhanin。泛隐私流算法。见《国际超级计算会议会议录》。2010年。

[28] C. Dwork、M. Naor、O. Reingold、G. N. Rothblum和S. P. Vadhan。差分隐私数据发布的复杂性：高效算法与困难性结果。见《2009年计算理论研讨会》，第381 - 390页。2009年。

[29] C. Dwork、M. Naor和S. Vadhan。分析师的隐私与国家的权力。见《计算机科学基础》。2012年。

[30] C. Dwork、A. Nikolov和K. Talwar。通过凸松弛实现隐私地发布边际分布的高效算法。见《年度计算几何研讨会（SoCG）会议录》。2014年。

[31] C. Dwork和K. Nissim。垂直分区数据库上的隐私保护数据挖掘。见《2004年密码学会议录》，第3152卷，第528 - 544页。2004年。

[32] C. Dwork、G. N. Rothblum和S. P. Vadhan。提升与差分隐私。见《计算机科学基础》，第51 - 60页。2010年。

[33] C. Dwork、K. Talwar、A. Thakurta和L. Zhang。分析高斯分布：隐私保护主成分分析的最优界。见《计算理论研讨会》。2014年。

[34] L. Fleischer和Y.-H. Lyu。当成本与数据相关时，出售隐私的近似最优拍卖。见《美国计算机协会电子商务会议》，第568 - 585页。2012年。

[35] A. Ghosh和K. Ligett。隐私与协调：具有内生参与的数据库计算。见《第十四届${ACM}$电子商务会议（EC）会议录》，第543 - 560页，2013年。

[36] A. Ghosh和A. Roth。通过拍卖出售隐私。见《美国计算机协会电子商务会议》，第199 - 208页。2011年。

[37] A. Groce、J. Katz和A. Yerukhimovich。客户端/服务器环境下计算差分隐私的极限。见《密码学理论会议会议录》。2011年。

[38] A. 古普塔（A. Gupta）、M. 哈特（M. Hardt）、A. 罗斯（A. Roth）和 J. 厄尔曼（J. Ullman）。私密发布合取式与统计查询障碍。见《2011 年计算理论研讨会论文集》，第 803 - 812 页。2011 年。

[39] A. 古普塔（A. Gupta）、A. 罗斯（A. Roth）和 J. 厄尔曼（J. Ullman）。迭代构造与私密数据发布。见《密码学理论会议论文集》，第 339 - 356 页。2012 年。

[40] J. 哈斯塔德（J. Håstad）、R. 因帕利亚佐（R. Impagliazzo）、L. 莱文（L. Levin）和 M. 卢比（M. Luby）。基于任意单向函数的伪随机生成器。《工业与应用数学学会计算杂志》，28 卷，1999 年。

[41] M. 哈特（M. Hardt）、K. 利格特（K. Ligett）和 F. 麦克谢里（F. McSherry）。一种简单实用的差分隐私数据发布算法。见《神经信息处理系统进展 25》，第 2348 - 2356 页。2012 年。

[42] M. 哈特（M. Hardt）和 A. 罗斯（A. Roth）。在非相干矩阵上超越随机响应。见《计算理论研讨会论文集》，第 1255 - 1268 页。美国计算机协会，2012 年。

[43] M. 哈特（M. Hardt）和 A. 罗斯（A. Roth）。私密奇异向量计算中的超越最坏情况分析。见《计算理论研讨会论文集》。2013 年。

[44] M. 哈特（M. Hardt）和 G. N. 罗斯布卢姆（G. N. Rothblum）。一种用于隐私保护数据分析的乘法权重机制。见《计算机科学基础研讨会论文集》，第 61 - 70 页。电气与电子工程师协会计算机学会，2010 年。

[45] M. 哈特（M. Hardt）和 K. 塔尔瓦尔（K. Talwar）。关于差分隐私的几何性质。见《美国计算机协会计算理论研讨会论文集》，第 705 - 714 页。美国计算机协会，2010 年。

[46] N. 霍默（N. Homer）、S. 泽林格（S. Szelinger）、M. 雷德曼（M. Redman）、D. 杜根（D. Duggan）、W. 坦贝（W. Tembe）、J. 米林（J. Muehling）、J. 皮尔逊（J. Pearson）、D. 斯蒂芬（D. Stephan）、S. 尼尔森（S. Nelson）和 D. 克雷格（D. Craig）。使用高密度单核苷酸多态性基因分型微阵列解析对高度复杂混合物贡献微量 DNA 的个体。《公共科学图书馆·遗传学》，4 卷，2008 年。

[47] J. 许（J. Hsu）、Z. 黄（Z. Huang）、A. 罗斯（A. Roth）、T. 拉夫加登（T. Roughgarden）和 Z. S. 吴（Z. S. Wu）。私密匹配与分配。预印本 arXiv:1311.2828，2013 年。

[48] J. 许（J. Hsu）、A. 罗斯（A. Roth）和 J. 厄尔曼（J. Ullman）。通过私密均衡计算为分析师实现差分隐私。见《美国计算机协会计算理论研讨会（STOC）论文集》，第 341 - 350 页，2013 年。

[49] Z. 黄（Z. Huang）和 S. 坎南（S. Kannan）。用于社会福利的指数机制：私密、真实且近乎最优。见《电气与电子工程师协会计算机科学基础年度研讨会（FOCS）论文集》，第 140 - 149 页。2012 年。

[50] P. 贾因（P. Jain）、P. 科塔里（P. Kothari）和 A. 塔库尔塔（A. Thakurta）。差分隐私在线学习。《机器学习研究杂志 - 会议录专刊》，23:24.1 - 24.34，2012 年。

[51] M. 卡普拉洛夫（M. Kapralov）和 K. 塔尔瓦尔（K. Talwar）。关于差分隐私低秩逼近。见桑吉夫·坎纳（Sanjeev Khanna）主编的《离散算法研讨会论文集》，第 1395 - 1414 页。工业与应用数学学会，2013 年。

[52] S. P. 卡西维斯瓦纳坦（S. P. Kasiviswanathan）、H. K. 李（H. K. Lee）、科比·尼斯姆（Kobbi Nissim）、S. 拉斯霍德尼科娃（S. Raskhodnikova）和 A. 史密斯（A. Smith）。我们能私密地学习什么？《工业与应用数学学会计算杂志》，40(3):793 - 826，2011 年。

[53] M. 凯恩斯（M. Kearns）。基于统计查询的高效容错学习。《美国计算机协会杂志》（JAssociation for Computing Machinery），45(6):983 - 1006，1998 年。

[54] M. 卡恩斯（M. Kearns）、M. 派伊（M. Pai）、A. 罗斯（A. Roth）和 J. 厄尔曼（J. Ullman）。大型博弈中的机制设计：激励与隐私。见第五届理论计算机科学创新会议（ITCS）论文集，2014 年。

[55] D. 基弗（D. Kifer）、A. 史密斯（A. Smith）和 A. 塔库尔塔（A. Thakurta）。私有凸经验风险最小化与高维回归。《机器学习研究杂志》，1:41，2012 年。

[56] K. 利格特（K. Ligett）和 A. 罗斯（A. Roth）。接受或放弃：当隐私需要付出代价时进行调查。见《互联网与网络经济学》，第 378 - 391 页。施普林格出版社，2012 年。

[57] N. 利特尔斯特恩（N. Littlestone）和 M. K. 瓦尔穆斯（M. K. Warmuth）。加权多数算法。见年度计算机科学基础研讨会，1989 年，第 256 - 261 页。电气与电子工程师协会（IEEE），1989 年。

[58] A. 麦格雷戈（A. McGregor）、I. 米罗诺夫（I. Mironov）、T. 皮塔西（T. Pitassi）、O. 赖因戈尔德（O. Reingold）、K. 塔尔瓦尔（K. Talwar）和 S. P. 瓦德汉（S. P. Vadhan）。双方差分隐私的局限性。见《计算机科学基础》，第 81 - 90 页。IEEE 计算机协会，2010 年。

[59] F. 麦克谢里（F. McSherry）。隐私集成查询（代码库）。可在微软研究院下载网站获取。另见 2009 年 SIG - MOD 会议论文集。

[60] F. 麦克谢里（F. McSherry）和 K. 塔尔瓦尔（K. Talwar）。通过差分隐私进行机制设计。见《计算机科学基础》，第 94 - 103 页。2007 年。

[61] F. 麦克谢里（F. McSherry）和 K. 塔尔瓦尔（K. Talwar）。通过差分隐私进行机制设计。见《计算机科学基础》，第 94 - 103 页。2007 年。

[62] D. 米尔（D. Mir）、S. 穆图克里什南（S. Muthukrishnan）、A. 尼科洛夫（A. Nikolov）和 R. N. 赖特（R. N. Wright）。通过草图统计实现泛隐私算法。见美国计算机协会 SIGMOD - SIGACT - SIGART 数据库系统原理研讨会论文集，第 37 - 48 页。美国计算机协会，2011 年。

[63] I. 米罗诺夫（I. Mironov）。最低有效位对差分隐私的重要性。见 T. 于（T. Yu）、G. 达内齐斯（G. Danezis）和 V. D. 格里戈尔（V. D. Gligor）编，美国计算机协会计算机与通信安全会议论文集，第 650 - 661 页。美国计算机协会，2012 年。

[64] I. 米罗诺夫（I. Mironov）、O. 潘迪（O. Pandey）、O. 赖因戈尔德（O. Reingold）和 S. P. 瓦德汉（S. P. Vadhan）。计算差分隐私。见《密码学会议论文集》，第 126 - 142 页。2009 年。

[65] A. 纳拉亚南（A. Narayanan）和 V. 什马蒂科夫（V. Shmatikov）。大型稀疏数据集的鲁棒去匿名化（如何破解网飞奖数据集的匿名性）。见 IEEE 安全与隐私研讨会论文集。2008 年。

[66] A. 尼科洛夫（A. Nikolov）、K. 塔尔瓦尔（K. Talwar）和 L. 张（L. Zhang）。差分隐私的几何：稀疏和近似情况。计算理论研讨会，2013 年。

[67] K. 尼斯姆（K. Nissim）、C. 奥兰迪（C. Orlandi）和 R. 斯莫罗金斯基（R. Smorodinsky）。隐私感知机制设计。见美国计算机协会电子商务会议论文集，第 774 - 789 页。2012 年。

[68] K. 尼斯姆（K. Nissim）、S. 拉斯霍德尼科娃（S. Raskhodnikova）和 A. 史密斯（A. Smith）。私有数据分析中的平滑敏感度和采样。见美国计算机协会计算理论研讨会论文集，第 75 - 84 页。2007 年。

[69] K. 尼斯姆（K. Nissim）、R. 斯莫罗金斯基（R. Smorodinsky）和 M. 滕内霍尔茨（M. Tennenholtz）。通过差分隐私实现近似最优机制设计。见《理论计算机科学创新》，第 203 - 213 页。2012 年。

[70] M. 派（M. Pai）和 A. 罗斯（A. Roth）。隐私与机制设计。SIGecom 交流，2013 年。

[71] R. 罗杰斯（R. Rogers）和 A. 罗斯（A. Roth）。大型拥塞博弈中的渐近真实均衡选择。预印本 arXiv:1311.2625，2013 年。

[72] A. 罗斯（A. Roth）。差分隐私与线性查询的胖粉碎维度。见《近似、随机化与组合优化：算法与技术》，第 683 - 695 页。施普林格出版社，2010 年。

[73] A. 罗斯（A. Roth）。通过拍卖购买私有数据：敏感调查者问题。美国计算机协会 SIGecom 交流，11(1):1 - 8，2012 年。

[74] A. 罗斯（Roth）和 T. 拉夫加登（Roughgarden）。通过中位数机制实现交互式隐私保护。收录于《2010 年计算理论研讨会论文集》，第 765 - 774 页。2010 年。

[75] A. 罗斯（Roth）和 G. 舍内贝克（Schoenebeck）。低成本开展真实调查。收录于《ACM 电子商务会议论文集》，第 826 - 843 页。2012 年。

[76] B. I. P. 鲁宾斯坦（Rubinstein）、P. L. 巴特利特（Bartlett）、L. 黄（Huang）和 N. 塔夫特（Taft）。在大型函数空间中学习：支持向量机学习的隐私保护机制。预印本 arXiv:0911.5708，2009 年。

[77] R. 沙皮尔（Schapire）。机器学习的提升方法概述。收录于 D. D. 丹尼森（Denison）、M. H. 汉森（Hansen）、C. 霍姆斯（Holmes）、B. 马利克（Mallick）和 B. 于（Yu）主编的《非线性估计与分类》。施普林格出版社，2003 年。

[78] R. 沙皮尔（Schapire）和 Y. 辛格（Singer）。使用置信度评级预测改进提升算法。《机器学习》，39 卷：297 - 336 页，1999 年。

[79] R. E. 沙皮尔（Schapire）和 Y. 弗罗因德（Freund）。《提升方法：基础与算法》。麻省理工学院出版社，2012 年。

[80] A. 史密斯（Smith）和 A. G. 塔库尔塔（Thakurta）。通过稳定性论证实现差分隐私特征选择以及套索回归的鲁棒性。收录于《学习理论会议论文集》。2013 年。

[81] L. 斯威尼（Sweeney）。将技术与政策相结合以维护保密性。《法律、医学与伦理学杂志》，25 卷：98 - 110 页，1997 年。

[82] J. 厄尔曼（Ullman）。用差分隐私回答 ${\mathrm{n}}^{\{ 2 + o\left( 1\right) \} }$ 计数查询是困难的。收录于 D. 博内（Boneh）、T. 拉夫加登（Roughgarden）和 J. 费根鲍姆（Feigenbaum）主编的《计算理论研讨会论文集》，第 361 - 370 页。美国计算机协会，2013 年。

[83] L. G. 瓦利安特（Valiant）。可学习性理论。《美国计算机协会通讯》，27(11) 卷：1134 - 1142 页，1984 年。

[84] S. L. 华纳（Warner）。随机化回答：一种消除回避性回答偏差的调查技术。《美国统计协会杂志》，60(309) 卷：63 - 69 页，1965 年。

[85] D. 肖（Xiao）。隐私与真实性是否兼容？收录于《理论计算机科学创新会议论文集》，第 67 - 86 页。2013 年。