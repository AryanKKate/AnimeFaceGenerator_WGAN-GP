# AnimeFaceGenerator_WGAN-GP
1. Project Overview
This project implements an anime face generator using a 
Wasserstein GAN with Gradient Penalty (WGAN-GP). The model learns the distribution of anime faces from the Kaggle Anime Face Dataset and can synthesise novel, photorealistic-looking anime face images from random noise vectors.

Key design choices:
•	WGAN-GP training objective for stable gradient flow and no mode collapse
•	Deep convolutional Generator and Critic (DCGAN backbone)
•	InstanceNorm in the Critic to avoid batch-statistics interference with the GP
•	Evaluated with Inception Score (IS) and Frechet Inception Distance (FID)

2. Repository Structure
.
├── gan_anime_faces.py    
├── README.md             
├── generator.pth         
├── critic.pth            
└── outputs/
    ├── epoch_01.png …      
    ├── loss_curves.png     
    └── metrics.txt         

3. Setup & Installation
3.1 Requirements
Package	Version / Notes
Python	>= 3.9
PyTorch	>= 2.0 (CUDA recommended)
torchvision	>= 0.15
pytorch-fid	For FID computation
scipy	Matrix square-root in FID
numpy	>= 1.24
tqdm	Progress bars
matplotlib	Loss curve plots
kaggle	Dataset download (Colab)

3.2 Install Dependencies
pip install torch torchvision pytorch-fid scipy numpy tqdm matplotlib kaggle

3.3 Dataset Download
The training data is the Anime Face Dataset from Kaggle (splcher/animefacedataset).

Option A — Google Colab (automatic):
Uncomment the setup_kaggle_and_download() call in main(). Upload your kaggle.json when prompted.

Option B — Manual:
1.	Download the dataset from https://www.kaggle.com/datasets/splcher/animefacedataset
2.	Unzip into a directory so that images are at data/images/*.jpg
3.	Run the script normally

3.4 Running the Training
python gan_anime_faces.py
Outputs are written to the outputs/ directory. Checkpoints are saved as generator.pth and critic.pth after training completes.

4. GAN Architecture
4.1 Generator
The Generator maps a random latent vector z ∈ ℝ^{128×1×1} to a 64×64 RGB image. It uses five transposed convolution blocks to progressively upsample from a 1×1 spatial map to the target resolution.

Block	Operation	Kernel/Stride	Output Shape	Activation
Input	Latent z	—	128 × 1 × 1	—
Block 1	ConvTranspose2d + BN	4 / 1	512 × 4 × 4	ReLU
Block 2	ConvTranspose2d + BN	4 / 2	256 × 8 × 8	ReLU
Block 3	ConvTranspose2d + BN	4 / 2	128 × 16 × 16	ReLU
Block 4	ConvTranspose2d + BN	4 / 2	64 × 32 × 32	ReLU
Block 5	ConvTranspose2d	4 / 2	3 × 64 × 64	Tanh

Weights are initialised from N(0, 0.02) for convolutional layers and N(1, 0.02) for BatchNorm scale parameters (standard DCGAN initialisation).

4.2 Critic
The Critic maps an image to a scalar Wasserstein score. It mirrors the Generator with five convolutional blocks but uses InstanceNorm instead of BatchNorm.

Why InstanceNorm? The WGAN-GP gradient penalty is computed on interpolated samples. BatchNorm couples examples within a batch, causing the gradient of the penalty to depend on other samples and biasing the penalty estimate. InstanceNorm normalises each sample independently, avoiding this issue.

4.3 WGAN-GP Objective
The training objectives are:

# Critic (maximise Wasserstein distance):
L_C = E[C(fake)] - E[C(real)] + λ * GP

# Generator (fool the critic):
L_G = -E[C(G(z))]

# Gradient penalty:
GP = E[(||grad C(x_hat)||_2 - 1)^2]
x_hat = eps * real + (1 - eps) * fake,   eps ~ U[0,1]

The penalty enforces a 1-Lipschitz constraint on the Critic, which is required for the Wasserstein distance to be well-defined.

5. Training Process
5.1 Hyperparameters
Parameter	Value	Rationale
Image size	64 × 64	Good quality / speed trade-off
Batch size	128	Stable gradient estimates
Latent dim Z	128	Sufficient capacity
Epochs	20	Convergence on 15k subset
Critic iterations	3	Critic leads the generator
λ (GP weight)	10	Standard WGAN-GP value
LR (both)	1e-4	Adam recommended in paper
β1, β2 (Adam)	0.0, 0.9	Per WGAN-GP paper

5.2 Training Schedule
For each mini-batch:
4.	Sample a batch of real images from the dataset.
5.	Update the Critic 3 times using real and freshly generated fakes.
6.	Update the Generator once using a new batch of fakes.
7.	Log average losses; save a 8×4 image grid at the end of each epoch.

5.3 Convergence Indicators
•	Critic loss L_C should trend towards small negative or near-zero values.
•	Generator loss L_G should decrease (the generator improves).
•	Visually inspect outputs/epoch_XX.png: early epochs show noise / blobs; later epochs should show recognisable anime faces.

6. Evaluation Metrics
6.1 Inception Score (IS)
IS measures both the quality and diversity of generated images using a pretrained InceptionV3 classifier:

IS = exp( E_z[ KL( p(y|x) || p(y) ) ] )

•	High p(y|x) entropy (low confidence) → blurry / unrecognisable images → low IS
•	Low p(y) entropy (all generated images look the same) → mode collapse → low IS
•	

6.2 Frechet Inception Distance (FID)
FID compares the distribution of InceptionV3 pool3 features between real and generated images using the Wasserstein-2 distance between two Gaussians:

FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*(Sigma_r*Sigma_f)^(1/2))

•	FID captures both quality (feature proximity) and diversity (covariance matching).
•	Lower FID is better. State-of-the-art models on 64×64 anime faces achieve FID < 20.
•	Our model targets FID < 60 after 20 epochs on 15k training images.

6.3 Interpreting Results
Metric	Good Range	Issue if Outside Range
IS	> 2.5	Mode collapse or blurry images

7. Challenges & Design Decisions
7.1 Training Stability
Vanilla GANs are notoriously hard to train due to vanishing gradients and mode collapse. WGAN-GP directly addresses these by:
•	Using the Wasserstein-1 distance as the loss, which always provides a usable gradient.
•	Enforcing 1-Lipschitz via a soft gradient penalty rather than weight clipping.
•	Allowing more Critic updates per Generator step so the critic is a good estimator.

7.2 InstanceNorm vs BatchNorm in the Critic
As detailed in Section 4.2, BatchNorm in the Critic creates a dependency between samples in a mini-batch that corrupts the gradient penalty. InstanceNorm removes this at the cost of slightly slower convergence but produces more reliable Lipschitz enforcement.

7.3 Memory / Speed
Training on 15,000 images for 25 epochs on a T4 GPU takes approximately 20-25 minutes. Increasing the subset size or resolution will require proportionally more VRAM and time.

7.4 Potential Improvements
•	Self-Attention GAN (SAGAN) — add self-attention layers to capture global face structure.
•	Progressive Growing — start at 4×4 and grow to 64×64 for higher-quality outputs.
•	Spectral normalisation — additional Lipschitz control on the Critic.
•	Differentiable Augmentation (DiffAugment) — effective when training data is limited.
•	Train on the full dataset (~63k images) for improved FID.
•	Higher Number of epochs with checkpointing.



8. Script Quick Reference
Function / Class	Purpose
setup_kaggle_and_download()	Download dataset via Kaggle API (Colab)
get_dataloader()	Build DataLoader from anime face images
Generator	5-block transposed conv network (z -> image)
Critic	5-block conv network (image -> score)
gradient_penalty()	Compute WGAN-GP penalty term
weights_init()	DCGAN weight initialisation
train()	Main WGAN-GP training loop
inception_score()	Compute IS on generated samples
compute_fid()	Compute FID between real & generated
plot_loss_curves()	Save matplotlib loss curve figure
main()	Entry point: data -> train -> evaluate

PS : Comments in the Code file are curated by AI (ChatGPT)
