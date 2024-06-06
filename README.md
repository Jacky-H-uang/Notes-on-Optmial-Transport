# A Study of Aligned Textual Visual Features by Optimal Transport

## Optimal Transport theory 
This theory is designed to minimize the difference between two distributions. His main process is as follows:

First the two distributions to be compared:

$$
U = \sum_{m=1}^Mu_m\delta_{f_m} \quad \quad and \quad \quad V = \sum_{n=1}^{N}v_n\delta_{g_n}
$$

where $u,v$ are discrete probability vectors with sum 1; $\delta_f$ is the Dirac delta function placed at the embedding space support $f$ out.

Then there is the distance function:

$$
<{T,C}> = \sum_{m=1}^M\sum_{n=1}^N T_{m,n}C_{m,n}.
$$

Where $C$ is a cost matrix where each point represents the cost between $f_m$ and $g_n$, e.g., $C_{m,n} = 1-sim(f_m,g_n)$; and $T$ is known as the transport plan, which is learned to minimize the total distance.

Finally it is the optimization problem of transport theory as:

$$
d_{OT}(u,v|C) = \underset{T}{minimize}<T,C> \\
subject \ to \ \ \ \ T1_N = u, T^\top 1_M = v, \ T \in \mathbb{R}_+^{M\times N}.
$$

> The subject is understood as $T1_N = u$ which ensures that the sum of masses at the beginning of the transfer is constant and equal to the pre-transfer distribution, and $T^\top1_M = v$ which ensures that the sum of masses after the transfer is equal to the post-transfer distribution. So what he's trying to say is that I want to control that both the entrance and exit of this probability are error-free, which is commonly understood to mean that what departs from one place correctly follows the intended distribution to the other place. Such a design ensures the completeness and consistency of the probability, which in math we call keeping the marginal distribution.
> 

In addition are some variants of OT optimization which include fast optimization using Sinkhorn distance:

$$
d_{OT}(u,v|C) = \underset{T}{minimize}<T,C> - \lambda h(T) \\
subject \ to \ \ \ \ T1_N = u, T^\top1_M = v, \ T \in \mathbb{R}_+^{M\times N}.
$$

Here $h(\cdot)$ is an entropy and $\lambda \ge 0$ is a hyperparameter; a fast optimization solution can be obtained from the above expression, requiring only a few iterations:

$$
T^* = diag(u^{(t)})exp(-C/\lambda)diag(v^{(t)}) \\
u^{(t)} = u / ((exp(-C/\lambda)v^{(t-1)})) \quad \ and \quad \ v^{(t)} = v / ((exp(-C/\lambda)^\top u^{(t-1)}))
$$

Here $t$ denotes the number of iterations, initialized $v^{(0)}=1$;

> $T^*$ This is designed to reduce the amount of transmission while satisfying the edge distribution.
> 

> The specific form of $h(T)$ should be $h(T) = -\sum_{i,j}T_{i,j}logT_{i,j}$, and this form of entropy measures the dispersion and smoothness of $T$, with higher entropy values indicating a uniform distribution of $T$ elements, and lower entropy indicating a distribution by a larger number of 0 elements or an uneven distribution.
> 

> Since $u^{(t)}, v^{(t)}$ are alternately updated and both have analytic expressions, their optimization process is indeed fast and converges in only a few iterations.
>

## Acknowledgements
This code is partly based on the open-source implementations from [PLOT](https://github.com/CHENGY12/PLOT).
