<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# مرسییی. ببین دقیقا میخوام همین کارو کنی. این فایل یه تمرین هست باید انجام بدم و میخوام بدم به یه هوش مصنوعی راه نماییم کنه ولی عکس رو نمی تونه بخونه یا فایل pdf رو. ازت میخوام این فایل بطور بصورت متنی توضیح بدی ساختار فرمول ها قابل کپی باشن همه چیز رو بیار روی متن که بتونم کپی کنم. مواردی که فرمول دارن و نمیشه کپی کرد رو توضیح بده بصورت حرف بجاش. هیچ چیزی رو جا نزار

متن های پیشفرض رو هم که راحت عین pdf کپی کن برام بفرست. همه چیز رو به ترتیب درستش بفرست

حتما؛ در ادامه، کل محتوای PDF به‌صورت متنی و قابل کپی (با فرمول‌ها در قالب LaTeX) به همان ترتیب اصلی آورده می‌شود و هیچ موردی جا نیفتاده است.[^1][^2]

# Deep Learning-Based Synchronization for Uplink NB-IoT

Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer, Matthijs Van Keirsbilck, and Alexander Keller, NVIDIA; Contact: faitaoudia@nvidia.com.[^2][^1]

## Abstract

We propose a neural network (NN)-based algorithm for device detection and time of arrival (ToA) and carrier frequency offset (CFO) estimation for the narrowband physical random-access channel (NPRACH) of narrowband internet of things (NB-IoT). The introduced NN architecture leverages residual convolutional networks as well as knowledge of the preamble structure of the 5G New Radio (5G NR) specifications. Benchmarking on a 3rd Generation Partnership Project (3GPP) urban microcell (UMi) channel model with random drops of users against a state-of-the-art baseline shows that the proposed method enables up to 8 dB gains in false negative rate (FNR) as well as significant gains in false positive rate (FPR) and ToA and CFO estimation accuracy. Moreover, our simulations indicate that the proposed algorithm enables gains over a wide range of channel conditions, CFOs, and transmission probabilities. The introduced synchronization method operates at the base station (BS) and, therefore, introduces no additional complexity on the user devices. It could lead to an extension of battery lifetime by reducing the preamble length or the transmit power; our code is available on GitHub at NVlabs/nprach_synch (URL ذکرشده در متن مقاله).[^1][^2]

## I. INTRODUCTION

Narrowband internet of things (NB-IoT) is a radio technology standard developed by the 3rd Generation Partnership Project (3GPP) to enable a wide range of IoT services and user equipments (UEs). It mostly reuses 5G New Radio (5G NR) specifications, and aims to support massive numbers of connected UEs and to provide very good outdoor-to-indoor coverage, very low power consumption to enable long battery lifetime, and low cost connectivity. In the uplink, many UEs can simultaneously contend to access the channel in a random access manner. The narrowband physical random-access channel (NPRACH) relates to the first message sent by a UE to a base station (BS) to request access to the channel, and is used by the BS to identify the UE and estimate its time of arrival (ToA) and carrier frequency offset (CFO).[^2][^1]

The NPRACH waveform  is specified as a single-tone frequency hopping preamble, and different configurations are available to adapt to various channel conditions and cell sizes. Frequency hopping is performed according to a pseudo-random pattern to mitigate inter-cell interference as neighboring cells use different hopping patterns. Within a cell, up to 48 orthogonal hopping patterns are available for the UEs to choose from, and the UEs that simultaneously request access to the channel must use different patterns to avoid collisions. The problem of NPRACH detection consists in jointly detecting the UEs that simultaneously attempt to access the channel and estimating their respective ToAs and CFOs.[^1][^2]

To make the problem tractable, many existing algorithms – assume that (i) the channel frequency response is flat, which is reasonable considering the narrow bandwidth of NB-IoT (180 kHz), (ii) interference between UEs can be neglected, which only holds true assuming low CFO and low mobility, and (iii) the channel is time-invariant over the duration of the preamble, which is only valid under low mobility. Moreover, most algorithms require the configuration of a detection power threshold that depends on the noise-plus-interference level, and which controls the trade-off between the occurrence of false positives and false negatives.[^3][^4][^2][^1]

We propose a deep learning-based synchronization algorithm that requires none of these assumptions to hold, as well as no configuration of a detection threshold. The proposed algorithm is standard-compliant and operates at the BS, leading to no additional complexity for the UEs. To the best of our knowledge, the only related work is , which uses a convolutional neural network to predict the active UEs and corresponding ToAs and CFOs. In comparison, the method we propose uses a different neural network (NN) architecture which exploits knowledge of the preamble structure to achieve increased performance, and it uses a different loss function for training which is key to enable accurate ToA and CFO estimates.[^5][^2][^1]

Benchmarking using the Sionna link-level simulator  on a realistic 3GPP urban microcell (UMi) channel model  and against a state-of-the-art baseline  shows that the proposed algorithm enables gains of up to 8 dB in FNR, significant reduction in FPR, and more accurate ToA and CFO estimation. Moreover, these results hold over a wide range of CFOs and transmission probabilities, and as these gains were obtained for a short preamble length, such an algorithm may remove the need for longer preambles under many channel conditions or reduce the required transmit power, leading to battery lifetime extension.[^4][^6][^7][^2][^1]

## II. SYSTEM MODEL

The NPRACH waveform  consists of a sequence of symbol groups (SGs), as shown in Fig. 1; each SG is made of five identical single-tone orthogonal frequency division multiplexing (OFDM) symbols that share a single cyclic prefix (CP) to reduce overhead, and occupies one tone of 3.75 kHz bandwidth. Frequency hopping is performed between the SGs, and four consecutive SGs are treated as the basic unit of the preamble, referred to as a repetition; a preamble can consist of up to 128 repetitions for coverage extension.[^2][^1]

Let K denote the maximum number of devices that can simultaneously access the channel, assuming UEs that simultaneously request access do not collide (i.e., they use different hopping patterns). Under this assumption, the highest value allowed for K is 48, as this is the highest number of hopping patterns available. The samples transmitted by the kth user for the ith symbol of the mth SG are given by the single-tone OFDM exponential on subcarrier index of the hopping pattern as follows :[^1][^2]

$$
s_{k,m,i}[n] \;=\; \beta_k \, e^{j 2\pi \, \phi_k[m] \, \frac{n}{N}} \tag{1}
$$

where $\beta_k$ is the transmission power used by the kth user, $N$ is the number of subcarriers, and $\phi_k[m]$ is the subcarrier index used by the kth user for the mth SG determined by the hopping pattern; note that $N$ is typically greater than $K$ as NPRACH is only part of the BS-processed spectrum.[^2][^1]

We consider a multi-user time-invariant multipath channel, where the continuous-time channel response of the kth user is a tapped-delay-line of weighted Dirac impulses as follows :[^1][^2]

$$
h_k(\tau) \;=\; \sum_{p=0}^{P_k-1} a_{k,p}\, \delta(\tau - \tau_{k,p}) \tag{2}
$$

where $P_k$ is the number of paths, and $a_{k,p}$ and $\tau_{k,p}$ are the baseband coefficient and delay of the pth path of the kth user, respectively; the time of arrival (ToA) of user k is the minimum path delay defined by :[^2][^1]

$$
D_k \;:=\; \min_{p} \, \tau_{k,p} \tag{3}
$$

The received discrete-time signal at the BS, after sampling, is the sum over users and channel taps with user-specific CFO and AWGN given by :[^1][^2]

$$
y_{m,i}[n] \;=\; \sum_{k=0}^{K-1} \sum_{\ell=-\infty}^{\infty} A_k \, h_{k,\ell} \, s_{k,m,i}[n - \ell] \, e^{j 2\pi f_{\text{off},k}(n-\ell)} \;+\; w_{m,i}[n] \tag{4}
$$

where $A_k \in \{0,1\}$ indicates if user k is active, $h_{k,\ell}$ is the $\ell$th channel tap of user k, $f_{\text{off},k}$ is the CFO normalized by the sampling frequency, and $w_{m,i}[n]\sim \mathcal{CN}(0,\sigma^2)$ is AWGN; summation over taps is implemented over finite lags $L_{\min}\le \ell \le L_{\max}$ in simulation.[^2][^1]

Assuming sinc pulse shaping at the transmitter and matched filtering at the receiver, the discrete-time channel taps are samples of the bandlimited impulse response given by :[^1][^2]

$$
h_{k,\ell} \;=\; \sum_{p=0}^{P_k-1} a_{k,p}\, \mathrm{sinc}\big(\ell - W \tau_{k,p}\big) \;=\; \mathcal{F}^{-1}\{ \mathrm{rect}(f) \, H_k(f)\}(\ell) \tag{5–6}
$$

where $W = N \Delta f$ is the bandwidth, $\mathcal{F}^{-1}$ denotes the inverse Fourier transform, and the channel frequency response is :[^2][^1]

$$
H_k(f) \;=\; \sum_{p=0}^{P_k-1} a_{k,p} \, e^{-j 2\pi \tau_{k,p} W f} \tag{7}
$$

At the receiver, the CP of each SG is removed, a DFT is applied, and a resource grid (RG) $Y$ of size $N \times 5S$ is obtained where $S$ is the number of SGs in the preamble. Considering user k, the received RG sample on the active subcarrier for the ith OFDM symbol of the mth SG is the desired term plus ICI due to other users’ CFO and noise as follows :[^1][^2]

$$
\begin{aligned}
Y[\phi_k[m],\, 5m+i] \;=&\; A_k\, H_k\!\left(\frac{\phi_k[m]}{N}\right) \beta_k \cdot \frac{1}{N} \sum_{n=N_{m,i}}^{N_{m,i}+N-1} e^{j 2\pi f_{\text{off},k} n} \\
&+ \sum_{k'\ne k} A_{k'}\, H_{k'}\!\left(\frac{\phi_k[m]}{N}\right)\, \beta_{k'} \cdot \frac{1}{N} \sum_{n=N_{m,i}}^{N_{m,i}+N-1} e^{j 2\pi\left(\frac{\phi_{k'}[m]-\phi_k[m]}{N} + f_{\text{off},k'}\right)n} \\
&+ W_{k,m,i}[\phi_k[m],\, 5m+i]
\end{aligned} \tag{8}
$$

where $W[\phi_k[m],5m+i]\sim \mathcal{CN}(0,\sigma^2)$ is AWGN, and $N_{m,i}:=m N_{\text{SG}} + iN$ with $N_{\text{SG}}$ the number of time-domain samples in an SG; hopping patterns are orthogonal in time and frequency so $\phi_k[m]\ne \phi_{k'}[m]$ when $k\ne k'$. NPRACH synchronization consists in jointly detecting the active users and estimating their ToA and CFO from $Y$ in (8).[^2][^1]

## III. DEEP LEARNING-BASED SYNCHRONIZATION

Fig. 2 shows the proposed algorithm for NPRACH synchronization; it takes the received RG $Y \in \mathbb{C}^{N\times 5S}$ and preprocesses it into a real tensor $\bar{Y}\in \mathbb{R}^{K\times S\times 3}$ that feeds two NNs: one outputs transmission probabilities $\widehat{\Pr}(A_k\mid \bar{Y})$ and the other outputs ToA and CFO estimates $\widehat{D}_k$ and $\hat f_{\text{off},k}$.[^1][^2]

### A. Preprocessing

First, average the five resource elements (REs) forming each SG separately for each subcarrier, reducing input size by factor five to $\tilde{Y}\in \mathbb{C}^{N\times S}$. Second, gather and separately normalize the $K$ sequences of SGs corresponding to the possible hopping patterns per 3GPP, yielding $\tilde{y}_k\in \mathbb{C}^S$ for $0\le k\le K-1$, and convert complex sequences to real tensors by stacking real and imaginary parts to shape $S\times 2$. Since normalization removes power information, compute average received power per sequence in log-scale and concatenate it along the inner dimension, then scatter back to time-frequency RG order to form $\bar{Y}\in \mathbb{R}^{K\times S\times 3}$, allowing NNs to operate over time-frequency to better mitigate CFO-induced ICI.[^2][^1]

### B. Neural Network Architecture

The tensor $\bar{Y}$ is fed to two similar NNs: the first estimates $\widehat{\Pr}(A_k\mid \bar{Y})$ and the second estimates $\widehat{D}_k$ and $\hat f_{\text{off},k}$ per user. The first stage of both NNs uses 1D depth-wise separable convolutions (128 kernels, size 3) along frequency with skip connections and zero-padding to handle CFO-induced ICI efficiently, producing $Z^{(1)}$ and $Z^{(2)}$ of shape $K\times S\times 128$. Next, sequences per hopping pattern are gathered to $Z^{(1)}_k, Z^{(2)}_k\in \mathbb{R}^{S\times 128}$ and processed with MLPs: the first NN outputs $\widehat{\Pr}(A_k\mid \bar{Y})$, while the second NN uses two MLP heads for $\widehat{D}_k$ and $\hat f_{\text{off},k}$ with weight sharing across patterns. MLPs are feasible due to short preambles (e.g., one repetition $S=4$); longer preambles may require different architectures, though the proposed method’s gains could obviate longer preambles in many cases.[^1][^2]

### C. Loss Function

The user-activity detector NN is trained with binary cross-entropy :[^2][^1]

$$
L_1 \;:=\; - \sum_{k=0}^{K-1} \mathbb{E}\!\left[ \ln \widehat{\Pr}(A_k \mid \bar{Y}) \right] \tag{9}
$$

estimated via Monte Carlo over batch size $B$ as :[^1][^2]

$$
\begin{aligned}
L_1 \;\approx\;& -\frac{1}{B} \sum_{b=0}^{B-1} \sum_{k=0}^{K-1} \Big( A_k^{[b]} \ln \widehat{\Pr}(A_k^{[b]} \mid \bar{Y}^{[b]}) \\
&+ (1-A_k^{[b]}) \ln \big(1 - \widehat{\Pr}(A_k^{[b]} \mid \bar{Y}^{[b]})\big) \Big)
\end{aligned} \tag{10}
$$

The ToA/CFO regressor NN is trained with SNR-weighted MSE focusing loss on active users and emphasizing low-SNR errors :[^2][^1]

$$
L_2 \;:=\; \sum_{k=0}^{K-1} \mathbb{E}\!\left[ A_k\, \mathrm{SNR}_k \, (D_k - \widehat{D}_k)^2 \right] \;+\; \sum_{k=0}^{K-1} \mathbb{E}\!\left[ A_k\, \mathrm{SNR}_k \, (f_{\text{off},k} - \hat f_{\text{off},k})^2 \right] \tag{11}
$$

where the average SNR of user k is :[^1][^2]

$$
\mathrm{SNR}_k \;:=\; \frac{\beta_k}{\sigma^2}\,\frac{1}{S}\,\sum_{m=0}^{S-1} \left| H_k\!\left(\frac{\phi_k[m]}{N}\right) \right|^2 \tag{12}
$$

An empirical estimate of $L_2$ over batch is :[^2][^1]

$$
\begin{aligned}
L_2 \;\approx\;& \frac{1}{B} \sum_{k=0}^{K-1} \sum_{b=0}^{B-1} A_k^{[b]} \mathrm{SNR}_k^{[b]} \left(D_k^{[b]} - \widehat{D}_k^{[b]}\right)^2 \\
&+ \frac{1}{B} \sum_{k=0}^{K-1} \sum_{b=0}^{B-1} A_k^{[b]} \mathrm{SNR}_k^{[b]} \left(f_{\text{off},k}^{[b]} - \hat f_{\text{off},k}^{[b]}\right)^2
\end{aligned} \tag{13}
$$

The total loss jointly trains both NNs without extra weighting because $L_1$ and $L_2$ act on different heads :[^1][^2]

$$
L \;:=\; L_1 + L_2 \tag{14}
$$

## IV. SIMULATIONS RESULTS

The proposed algorithm is benchmarked against a state-of-the-art baseline using the Sionna link-level simulator  with the 3GPP UMi channel model , carrier frequency 3.4 GHz, and sampling frequency 50 MHz. NPRACH preamble format 0 is used with one repetition $S=4$, subcarrier spacing $\Delta f = 3.75$ kHz, and 48 subcarriers allocated to NPRACH; the maximum number of simultaneous access attempts is $K=48$ with collisionless access. While the method could be trained to detect and resolve collisions, no procedure exists to acknowledge colliding UEs, so such a solution cannot be exploited here.[^6][^7][^2][^1]

Training uses Adam  with batch size 64 and learning rate $10^{-3}$; for each batch example, user access probability $P(A_k)\sim \mathcal{U}(0,1)$, CFO per user in ppm $\sim \mathcal{U}(-25,25)$, and ToA $\sim \mathcal{U}(0,66.7\,\mu s)$ corresponding to CP length; ToAs are added to path delays from the 3GPP UMi channel model. Each batch example consists of a random drop of $K$ users with random large-scale parameters to avoid overfitting; a single NN is trained over a wide range of transmission probabilities, CFOs, and channel conditions, and then evaluated under specific conditions.[^8][^2][^1]

The baseline synchronization algorithm from , building on , relies on a detection power threshold $\gamma$ that trades FNR and FPR; FFT size is 256 and two thresholds correspond to false alarm probabilities 99.9% and 99%. Results: FNR vs SNR shows up to 8 dB gains at FNR $10^{-3}$ with zero CFO; gains decrease with CFO but remain significant up to 20 ppm, and baseline’s lower threshold improves FNR slightly but at the cost of higher FPR due to CFO-induced ICI leaking energy into adjacent subcarriers, whereas the NN mitigates false detections better. ToA and CFO RMSE vs SNR indicate the NN achieves more accurate estimates for all CFO values and ToA gains persist across transmission probabilities; similar CFO results are obtained but omitted for space.[^9][^3][^4][^2][^1]

## V. CONCLUSION

A neural network-based solution for NPRACH synchronization in NB-IoT is developed, showing significant gains in FNR, FPR, and ToA/CFO estimation accuracy across a wide range of CFOs and access probabilities, with standard compliance and no additional UE complexity. These gains suggest potential for shorter preambles or lower transmit power, improving battery lifetime for NB-IoT applications.[^2][^1]

## References

- M. Kanj, V. Savaux, and M. Le Guen, “A Tutorial on NB-IoT Physical Layer Design,” IEEE Commun. Surveys Tuts., vol. 22, no. 4, pp. 2408–2446, 2020.[^1][^2]
- 3GPP, “E-UTRA; Physical channels and modulation,” TS 36.211, V17.0.0, 2022.[^2][^1]
- X. Lin, A. Adhikary, and Y.-P. Eric Wang, “Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems,” IEEE Wireless Commun. Lett., 5(6):640–643, 2016.[^3][^1][^2]
- A. Chakrapani, “NB-IoT Uplink Receiver Design and Performance Study,” IEEE Internet Things J., 7(3):2469–2482, 2020.[^9][^1][^2]
- H. Chougrani, S. Kisseleff, and S. Chatzinotas, “Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT,” IEEE Internet Things J., 8(9):7437–7449, 2021.[^4][^1][^2]
- M. H. Jespersen, M. Pajovic, T. Koike-Akino, Y. Wang, P. Popovski, and P. V. Orlik, “Deep Learning for Synchronization and Channel Estimation in NB-IoT Random Access Channel,” in IEEE GLOBECOM, 2019, pp. 1–7.[^5][^1][^2]
- J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, “Sionna: An Open-Source Library for Next-Generation Physical Layer Research,” arXiv:2203.11854, 2022.[^6][^1][^2]
- 3GPP, “Study on channel model for frequencies from 0.5 to 100 GHz,” TS 38.901, V17.0.0, 2022.[^7][^1][^2]
- H. S. Jang, H. Lee, T. Q. S. Quek, and H. Shin, “Deep Learning-Based Cellular Random Access Framework,” IEEE Trans. Wireless Commun., 20(11):7503–7518, 2021.[^10][^1][^2]
- F. Chollet, “Xception: Deep Learning With Depthwise Separable Convolutions,” in CVPR, July 2017.[^11][^1][^2]
- D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” arXiv:1412.6980, 2014.[^8][^1][^2]

<div style="text-align: center">⁂</div>

[^1]: 2205.10805v2.pdf

[^2]: https://arxiv.org/pdf/2205.10805.pdf

[^3]: https://ui.adsabs.harvard.edu/abs/2022arXiv220510805A/abstract

[^4]: http://arxiv.org/list/cs/2022-05?skip=2725\&show=2000

[^5]: https://link.aps.org/doi/10.1103/PhysRevLett.122.011805

[^6]: https://vbn.aau.dk/ws/files/307172130/Deep_Learning_for_Synchronization_and_Channel_Estimation_in_NB_IoT_Random_Access_Channel.pdf

[^7]: https://github.com/NVlabs/nprach_synch

[^8]: https://discovery.ucl.ac.uk/10211593/1/Deep_Learning_ISAC_Review_Accepted.pdf

[^9]: http://arxiv.org/list/math/2022-05?skip=2090\&show=250

[^10]: https://daeckel.com/lander/daeckel.com/index.php?searchtype=author\&query=Van+keirsbilck%2C+M\&_=%2Fsearch%2Fcs%23XwKUwjkJUt2C%2BeRjLcsDVWE%3D

[^11]: https://arxiv.org/pdf/2501.11574.pdf

