Deep Learning-Based Synchronization for
Uplink NB-IoT
Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer,
Matthijs Van Keirsbilck, and Alexander Keller
NVIDIA
Contact: faitaoudia@nvidia.com
Abstract—We propose a neural network (NN)-based
algorithm for device detection and time of arrival (ToA)
and carrier frequency offset (CFO) estimation for the
narrowband physical random-access channel (NPRACH)
of narrowband internet of things (NB-IoT). The introduced
NN architecture leverages residual convolutional networks
as well as knowledge of the preamble structure of the 5G
New Radio (5G NR) specifications. Benchmarking on a 3rd
Generation Partnership Project (3GPP) urban microcell
(UMi) channel model with random drops of users against
a state-of-the-art baseline shows that the proposed method
enables up to 8 dB gains in false negative rate (FNR)
as well as significant gains in false positive rate (FPR)
and ToA and CFO estimation accuracy. Moreover, our
simulations indicate that the proposed algorithm enables
gains over a wide range of channel conditions, CFOs, and
transmission probabilities. The introduced synchronization
method operates at the base station (BS) and, therefore,
introduces no additional complexity on the user devices. It
could lead to an extension of battery lifetime by reducing
the preamble length or the transmit power. Our code is
available at: https://github.com/NVlabs/nprach_synch/.
I. INTRODUCTION
Narrowband internet of things (NB-IoT) is a radio
technology standard developed by the 3rd Generation
Partnership Project (3GPP) to enable a wide range of
IoT services and user equipments (UEs) [1]. It mostly
reuses 5G New Radio (5G NR) specifications, and aims
to support massive numbers of connected UEs and to
provide very good outdoor-to-indoor coverage, very low
power consumption to enable long battery lifetime, and
low cost connectivity. In the uplink, many UEs can simultaneously contend to access the channel in a random
access manner. The narrowband physical random-access
channel (NPRACH) relates to the first message sent by
a UE to a base station (BS) to request access to the
channel, and is used by the BS to identify the UE and
estimate its time of arrival (ToA) and carrier frequency
offset (CFO).
The NPRACH waveform [2] is specified as a singletone frequency hopping preamble, and different configurations are available to adapt to various channel
conditions and cell sizes. Frequency hopping is performed according to a pseudo-random pattern to mitigate
inter-cell interference as neighboring cells use different
hopping patterns. Within a cell, up to 48 orthogonal
hopping patterns are available for the UEs to choose
from, and the UEs that simultaneously request access
to the channel must use different patterns to avoid
collisions. The problem of NPRACH detection consists
in jointly detecting the UEs that simultaneously attempt
to access the channel and estimating their respective
ToAs and CFOs.
To make the problem tractable, many existing algorithms [3]–[5] assume that (i) the channel frequency
response is flat, which is reasonable considering the
narrow bandwidth of NB-IoT (180 kHz), (ii) interference
between UEs can be neglected, which only holds true
assuming low CFO and low mobility, and (iii) the channel is time-invariant over the duration of the preamble,
which is only valid under low mobility. Moreover, most
algorithms require the configuration of a detection power
threshold that depends on the noise-plus-interference
level, and which controls the trade-off between the
occurrence of false positives and false negatives.
We propose a deep learning-based synchronization
algorithm that requires none of these assumptions to
hold, as well as no configuration of a detection threshold. The proposed algorithm is standard-compliant and
operates at the BS, leading to no additional complexity
for the UEs. To the best of our knowledge, the only
related work is [6], which uses a convolutional neural
network to predict the active UEs and corresponding
ToAs and CFOs. In comparison, the method we propose
uses a different neural network (NN) architecture which
exploits knowledge of the preamble structure to achieve
increased performance. Moreover, it uses a different loss
function for training which is key to enable accurate ToA
and CFO estimates.
arXiv:2205.10805v2 [cs.IT] 29 Jul 2022
OFDM symbols
1st Repetition
SG SG SG SG
subcarriers
subcarriers
allocated to NPRACH
SG SG SG SG
Repetition
Fig. 1: NPRACH structure. Each color correspond to a preamble transmitted by a user.
Benchmarking using the Sionna link-level simulator [7] on a realistic 3GPP urban microcell (UMi) channel model [8] and against a state-of-the-art baseline [5]
shows that the proposed algorithm enables gains of up to
8 dB in false negative rate (FNR), significant reduction
in false positive rate (FPR), and more accurate ToA and
CFO estimation. Moreover, these results hold over a
wide range of CFOs and transmission probabilities. As
these gains were obtained for a short preamble length,
such an algorithm may remove the need for longer
preambles under many channel conditions or reduce
the required transmit power, leading to battery lifetime
extension which is critical for NB-IoT applications.
II. SYSTEM MODEL
The NPRACH waveform [2] consists of a sequence
of symbol groups (SGs), as shown in Fig. 1. Each SG is
made of five identical single-tone orthogonal frequency
division multiplexing (OFDM) symbols that share a single cyclic prefix (CP) to reduce overhead, and occupies
one tone of 3.75 kHz bandwidth. Frequency hopping is
performed between the SGs, and four consecutive SGs
are treated as the basic unit of the preamble, referred
to as a repetition. A preamble can consist of up to 128
repetitions for coverage extension.
Let us denote by K the maximum number of devices
that can simultaneously access the channel. It is assumed
that UEs that simultaneously request access do not
collide, i.e., that they use different hopping patterns.
Some methods in the literature address the detection of
colliding UEs, e.g., [9]. However, there is currently no
procedure to acknowledge access requests from colliding
UEs. Under this assumption, the highest value allowed
for K is 48, as this is the highest number of hopping
patterns available [2]. The samples transmitted by the k
th
user for the i
th symbol of the mth SG is
sk,m,i[n] = βke
j2πφk[m]
n
N (1)
where βk is the transmission power used by the k
th
user, N is the number of subcarriers, and φk[m] is the
subcarrier index used by the k
th user for the mth SG and
is determined by the hopping pattern. Note that N is
typically greater than K as the NPRACH is only part of
the radio spectrum processed by the BS.
We consider a multi-user time-invariant multipath
channel, where the channel response of the k
th user is
hk(τ ) =
P
Xk−1
p=0
ak,pδ (τ − τk,p) (2)
where Pk is the number of paths for the k
th user, and
ak,p and τk,p are the baseband coefficient and delay of
the p
th path of the k
th user, respectively. The ToA of the
k
th user is defined as
Dk := min
p
τk,p. (3)
The received signal at the BS is
ym,i[n] =
K
X−1
k=0
X∞
`=−∞
Akhk,`sk,m,i [n − `] e
j2πfoff,k(n−`)
+ wm,i[n] (4)
where Ak indicates if user k is active, i.e., it equals 1
if the user is transmitting and 0 otherwise, hk,` is the
channel coefficient of the `
th tap and of the k
th user,
foff,k is the CFO of the k
th user normalized by the
sampling frequency, and wm,i[n] is the additive white
Gaussian noise (AWGN) with variance σ
2
. To simulate
the channel, summation over taps is performed for a
finite number of time-lags Lmin ≤ ` ≤ Lmax. Assuming
sinc pulse shaping on the transmitter side and matched
filtering on the receiver side, the channel taps are
hk,` =
P
Xk−1
p=0
ak,psinc (` − W τk,p) (5)
= F
−1
{rect(f) Hk(f)} (`) (6)
2
where W = N∆f is the bandwidth, F
−1 {X(f)} (t) the
inverse Fourier transform of X(f) evaluated at t, and
Hk(f) =
P
Xk−1
p=0
ak,pe
−j2πτk,pW f (7)
the frequency response of the channel.
On the receiver side, the CP of each SG is removed
and discrete Fourier transform (DFT) is performed. This
leads to a resource grid (RG) Y of size N ×5S where S
is the number of SGs forming the preamble. Considering
the k
th user, the received signal for the for the i
th symbol
of the mth SG is
Y [φk[m], 5m + i] = AkHk

φk[m]
N

βk·
1
N
Nm,iX
+N−1
n=Nm,i
e
j2πfoff,kn +
X
k06=k
(
Ak0Hk0

φk[m]
N

βk0 ·
1
N
Nm,iX
+N−1
n=Nm,i
e
j2π
 φk0 [m]−φk[m]
N +foff,k0

n
)
+ Wk,m,i [φk[m], 5m + i] (8)
where W [φk[m], 5m + i] is AWGN with variance σ
2
,
and Nm,i := mNSG + iN where NSG is the number
of samples forming an SG. The hopping patterns are
orthogonal in time and frequency, i.e., φk[m] 6= φk0 [m]
when k 6= k
0
. The first term on the right-hand side of (8)
is the signal received from the k
th user, and the second
term corresponds to inter-carrier interference (ICI) from
the other users due to CFO. NPRACH synchronization
consists in jointly detecting the active users and estimating their ToA and CFO from the received signal Y (8).
The next section introduces a NN-based detector that
aims to achieve this goal by exploiting the NPRACH
preamble structure.
III. DEEP LEARNING-BASED SYNCHRONIZATION
Fig. 2 shows the algorithm that we propose for
NPRACH synchronization. It takes as input the received
RG Y ∈ C
N×5S, which is first preprocessed into a realvalued tensor Y¯ ∈ R
K×S×3
. Y¯ then serves as input to
two NNs, one to compute transmission probabilities, denoted by Pr c

Ak|Y¯

, and the other to compute ToA and
CFO estimates, denoted by Dbk and ˆfoff,k, respectively.
The rest of this section details the preprocessing, the NN
architectures, and the loss function used for training.
A. Preprocessing
Preprocessing consists of first averaging the five resource elements (REs) forming each SG. This is done
separately for each subcarrier. This preprocessing step
reduces by a factor of five the input size, resulting in a
matrix Ye ∈ C
N×S.
The second preprocessing step consists of separately
normalizing the sequences of SGs corresponding to the
K possible hopping patterns. This is key to enable
accurate detection despite the users having signal-tonoise ratios (SNRs) that differ by orders of magnitude
due to path loss. This is achieved by gathering the
K sequences corresponding to every hopping pattern
y˜k ∈ C
S, 0 ≤ k ≤ K−1, according to the 3GPP specifications [2], and normalizing each sequence individually.
The resulting normalized sequences are then converted
from complex-valued tensors to real-valued ones by
stacking the real and imaginary components along an
additional dimension, resulting in a tensor of shape
S × 2. As normalization erases the information about
the received power, the average received power of each
sequence is computed prior to normalization in log-scale
and concatenated to the normalized sequence along the
inner dimension. The received power takes values over
a range of several orders of magnitude due to path loss.
Finally, the K resulting tensors are scattered according
to the hopping patterns, reverting the previous gathering
operation, to form a normalized tensor Y¯ ∈ R
K×S×3
.
This allows the NNs to operate over the time-frequency
RG and hence to better mitigate the ICI due to CFO,
as opposed to conventional approaches which typically
ignore ICI for tractability.
B. Neural Network Architecture
The tensor Y¯ serves as input to two NNs that
share a similar architecture. For each user, the first NN
computes a probability of the user to request channel
access, and the second NN computes estimates of the
ToA and the CFO of the user. The first stage of both
NNs is motivated by the ICI caused by the CFO, and
consists of one-dimensional (1D) depth-wise separable
convolutional layers [10] with 128 kernels of size 3
and that operate along the frequency dimension. Depthwise separable convolutional layers are used as they significantly decrease computational complexity compared
to conventional convolutional layers without reducing
accuracy. Skip connections are used to avoid gradient
vanishing, and zero-padding ensures that the output has
the same length as the input. The resulting tensors are
denoted by Z
(1) and Z
(2) for the first and second NN,
respectively, and have shape K × S × 128, where the
last dimension corresponds to the “channels” of the
convolution. Intuitively, the convolutional NNs compute
for each subcarrier and for each SG a vector of features.

Next, the sequences of SGs corresponding to the hopping patterns are gathered, leading to K tensors denoted
by Z
(1)
k ∈ R
S×128 and Z
(2)
k ∈ R
S×128
, 0 ≤ k ≤ K − 1,
for the first and second NN, respectively. This operation
is similar to the one performed for normalizing the
received NN at preprocessing. The first NN computes
for every user k a probability that the user is requesting
channel access, denoted by Pr c

Ak|Y¯

, by processing
Z
(1)
k with a multilayer perceptron (MLP). Similarly, the
second NN computes for every user k estimates of
the ToA and CFO, denoted by Dbk and ˆfoff,k, from
Z
(2)
k ∈ R
S×128 using two separate MLPs. For both
NNs, weights sharing is performed across the K hopping
patterns for the MLPs.
The use of MLPs is made possible as short preambles
are assumed, leading to small values of S. For example,
one preamble repetition corresponds to S = 4. Long
preambles could prohibit the use of MLPs due to scalability and require a different architecture. However, as
we will see in the next section, the proposed algorithm
enables significant gains that could remove the need for
longer preambles in many environments.
C. Loss Function
Training of the NN that detects the transmitting users
is done on the binary cross-entropy
L1 := −
K
X−1
k=0
E
h
ln 
Pr c

Ak|Y¯

i (9)
which is estimated through Monte Carlo sampling by
L1 ≈ −
1
B
B
X−1
b=1
K
X−1
k=0 "
A
[b]
k
ln 
Pr c

A
[b]
k
|Y¯ [b]

+

1 − A
[b]
k

ln 
1 − Pr c

A
[b]
k
|Y¯ [b]
 #
(10)
where B is the batch size and the superscript [b] refers
to the b
th batch example.
Training of the NN that estimates ToA and CFO is
done on the weighted mean squared error
L2 :=
K
X−1
k=0
E

AkSNRk

Dk − Dbk
2

+
K
X−1
k=0
E

AkSNRk

foff,k − ˆfoff,k2

(11)
where
SNRk :=
βk
σ
2
1
S
S
X−1
m=0




Hk

φk[m]
N




2
(12)
Average over
SGs
Stack
Extract REs
for each pattern
Layer norm.
Scatter REs
over RG
Preprocessing
Applied to all
Sep. Conv1D
128/3
Sep. Conv1D
128/3 Sep. Conv1D
128/3
Sep. Conv1D
128/3
ResNet Block
ResNet Block
ResNet Block
Extract REs
for each pattern
ResNet Block
ResNet Block
ResNet Block
Extract REs
for each pattern
Applied to all
Dense
1024/ReLU
Dense
1024/ReLU
Dense
1024/ReLU
Dense
256/ReLU
Dense
256/ReLU
Dense
256/ReLU
Dense
1/sigmoid
Dense
512/ReLU
Dense
512/ReLU
Dense
512/ReLU
Flatten Flatten Flatten
ReLU
ReLU
Layer norm.
Layer norm.
+
Dense
1/linear
Dense
1/linear
Fig. 2: NN-based synchronization algorithm. When labeling the output of a layer is not required, only the
shape is indicated. For dense layers, the number of
units/activation function is indicated. For separable convolution layers, the number of kernels/kernel size is
indicated.

is the average SNR of user k. Weighting by Ak ensures
that only the active users are considered. Weighting
by the SNR is motivated by the observation that the
errors measured for high SNRs are negligible compared
to the ones measured for low SNRs. Training on the
unweighted MSE therefore results in poor accuracy for
high SNR. The loss L2 is estimated by
L2 ≈
1
B
K
X−1
k=0
B
X−1
b=0
A
[b]
k
SNR[b]
k

D
[b]
k − Db[b]
k
2
+
1
B
K
X−1
k=0
B
X−1
b=0
A
[b]
k
SNR[b]
k

f
[b]
off,k − ˆf
[b]
off,k2
. (13)
Training of the two NNs is jointly performed on the total
loss
L := L1 + L2 (14)
where no weighting is required as L1 and L2 act on
different NNs.
IV. SIMULATIONS RESULTS
We have benchmarked the previously described algorithm against a state-of-the-art baseline using the
Sionna link-level simulator [7]. The 3GPP UMi channel
model [8] was used, with the carrier frequency set to
3.4 GHz and the sampling frequency set to 50 MHz. The
NPRACH preamble format 0 was implemented, and the
number of repetitions was set to 1, leading to S = 4 SGs.
Following the NB-IoT specifications [2], the subcarrier
spacing ∆f was set to 3.75 kHz and the number of
subcarriers allocated to the NPRACH to 48. The maximum number of users that can simultaneously request
channel access was set to K = 48 with collisionless
access. Note that the proposed method could be trained
to detect and resolve collisions. However, as there is no
procedure for handling collisions, such a solution could
not be exploited.
Training of the NNs was done using the Adam optimizer [11], with the batch size set to 64 and the
learning rate set to 10−3
. At training, the probability
for a user to request access P(Ak) was independently
and uniformly sampled from the range (0, 1) for each
batch example. The CFO, in parts-per-million (ppm),
of each user and for each batch example was independently and uniformly sampled from the range (−25, 25).
Similarly, the ToA of each user and for each batch
example was independently and uniformly sampled from
the range (0, 66.7 µs), where 66.7 µs corresponds to the
CP length [2]. The ToAs were added to the path delays
generated by the 3GPP UMi channel model. At both
evaluation and training, each batch example consisted
of a random drop of K users with randomly chosen
large scale parameters to avoid over-fitting to specific
channel conditions. Note that a single NN was trained
over a wide range of transmission probabilities, CFOs,
and channel conditions, and then evaluated under specific
conditions.
To benchmark the proposed method, the synchronization algorithm from [5] was implemented, which
builds on previous work [3], [4]. This algorithm relies
on the configuration of a detection power threshold
denoted by γ, that controls a trade-off between the FNR
and the FPR. As in [5], the size of the fast Fourier
transform (FFT) performed by the baseline was set to
256. Moreover, two values for the detection threshold
were used, corresponding to false alarm probabilities of
99.9 % (as in [5]) and 99 %.
Fig. 3a shows FNR versus SNR (12). In all figures,
the baseline and our approach are referred to as “BL”
and “NN”, respectively. The CFO was randomly and
uniformly sampled for each user, and the maximum CFO
value in ppm is indicated in the legend. The probability
for a user to transmit was set to 0.5 for Figures 3a to 3c
One can see that the NN-based method enables gains of
up to 8 dB at an FNR of 10−3 under no CFO. The gains
decrease as the CFO increases, but remain significant
even under high CFOs of up to 20 ppm. As expected,
the FNRs achieved by the baseline are slightly better for
the low power detection threshold γ. However, this is
at the cost of higher FPRs, as shown in Fig. 3e. Note
that Fig. 3e shows results averaged over all channel
realizations, and therefore over all SNRs. The FPR is
significantly higher with the baseline than with the NN
approach, except for the lowest CFOs. The steep FPR
increase observed for the baseline can be explained by
the ICI caused by the CFO (8), as the energy leaked
in adjacent subcarriers erroneously triggers the detection
threshold. The more advanced processing performed by
the NN-based partially prevents from false detection.
In addition to better detection performance, the NNbased algorithm also enables more accurate ToA and
CFO estimation, as shown in Fig. 3b and Fig. 3c. For the
baseline, the detection threshold γ does not impact the
ToA and CFO estimation accuracy. Both the baseline
and the NN-based approach are negatively impacted
by increasing CFOs, however, the NN-based algorithm
outperforms the baseline for all CFO values. Moreover,
as shown in Fig 3d, the NN-based algorithm outperforms
the baseline in ToA estimation for all transmission
probabilities. Similar results were obtained for CFO
estimation, but are not shown due to space limitation.
5
CFO: 0 ppm
BL - =99%
BL - =99.9%
NN
CFO: 10 ppm
BL - =99%
BL - =99.9%
NN
CFO: 20 ppm
BL - =99%
BL - =99.9%
NN
20 10 0 10 20
SNR [dB]
10
4
10
3
10
2
10
1
10
0
False negative rate
(a) FNR against SNR.
20 10 0 10 20
SNR [dB]
10
2
10
1
10
0
Normalized ToA RMSE
(b) ToA RMSE against SNR.
20 10 0 10 20
SNR [dB]
10
10
10
9
10
8
Normalized CFO RMSE
(c) CFO RMSE against SNR.
0.0 0.2 0.4 0.6 0.8 1.0
P(Ak)
10
1
10
0
Normalized ToA RMSE
(d) ToA RMSE against P(Ak).
0 5 10 15 20
CFO [ppm]
10
1
False positive rate
BL - =99%
BL - =99.9%
NN
(e) FPR against CFO.
Fig. 3: Simulation results. γ is the baseline detection threshold. P(Ak) is the probability for a user to transmit.
V. CONCLUSION
We have developed a NN based solution for NPRACH
synchronization in NB-IoT. We have shown that it enables significant gains in FNR, FPR, as well as ToA and
CFO estimation accuracy. These gains are observed for
a wide range of CFOs and transmission probabilities,
showing the robustness of the such an approach. Our
solution is standard-compliant and incurs no additional
complexity at the user devices. It could hence increase
battery lifetime by enabling shorter preambles or lower
transmit power, which is critical for NB-IoT applications.
REFERENCES
[1] M. Kanj, V. Savaux, and M. Le Guen, “A Tutorial on NB-IoT
Physical Layer Design,” IEEE Commun. Surveys Tuts., vol. 22,
no. 4, pp. 2408–2446, 2020.
[2] 3rd Generation Partnership Project (3GPP), “Evolved Universal
Terrestrial Radio Access (E-UTRA); Physical channels and modulation,” TS 36.211, V17.0.0, 2022.
[5] H. Chougrani, S. Kisseleff, and S. Chatzinotas, “Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone
Frequency Hopping Random Access in NB-IoT,” IEEE Internet
Things J., vol. 8, no. 9, pp. 7437–7449, 2021.
[3] X. Lin, A. Adhikary, and Y.-P. Eric Wang, “Random Access
Preamble Design and Detection for 3GPP Narrowband IoT
Systems,” IEEE Wireless Commun. Lett., vol. 5, no. 6, pp. 640–
643, 2016.
[4] A. Chakrapani, “NB-IoT Uplink Receiver Design and Performance Study,” IEEE Internet Things J., vol. 7, no. 3, pp. 2469–
2482, 2020.
[6] M. H. Jespersen, M. Pajovic, T. Koike-Akino, Y. Wang,
P. Popovski, and P. V. Orlik, “Deep Learning for Synchronization
and Channel Estimation in NB-IoT Random Access Channel,” in
IEEE Global Telecommun. Conf. (GLOBECOM), 2019, pp. 1–7.
[7] J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder,
G. Marcus, and A. Keller, “Sionna: An Open-Source Library for Next-Generation Physical Layer Research,” preprint
arXiv:2203.11854, 2022.
[8] 3rd Generation Partnership Project (3GPP), “Study on channel
model for frequencies from 0.5 to 100 GHz,” TS 38.901, V17.0.0,
2022.
[9] H. S. Jang, H. Lee, T. Q. S. Quek, and H. Shin, “Deep LearningBased Cellular Random Access Framework,” IEEE Trans. Wireless Commun., vol. 20, no. 11, pp. 7503–7518, 2021.
[10] F. Chollet, “Xception: Deep Learning With Depthwise Separable Convolutions,” in IEEE Conf. Comput. Vis. Pattern Recog.
(CVPR), July 2017.
[11] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic
Optimization,” preprint arXiv:1412.6980, 2014.