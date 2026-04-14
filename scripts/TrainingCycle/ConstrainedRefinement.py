import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from vocos import Vocos
from vocos.discriminators import MultiResolutionDiscriminator
from vocos.loss import GeneratorLoss, DiscriminatorLoss
from transformers import WavLMModel
import torchaudio
import json, random, os, csv, gc
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# --- CONFIG ---
DEVICE = torch.device("cuda")
BATCH_SIZE = 32
LR = 5e-6            # Low LR for fine-grained refinement
CROP_SAMPLES = 48000
DATA_ROOT = "/workspace/data/"
SAVE_INTERVAL = 1

# --- PHASE II CONFIGURATION ---
W_WAVLM = 5.0        # INCREASED: Stricter phonetic control
W_ENERGY = 100.0     # DECREASED: Allows return of natural acoustic peaks
W_ADV = 0.005        # DECREASED: Prevents GAN from generating artifacts

# --- SETUP DIRECTORIES ---
OLD_CHECKPOINT_PATH = "checkpoints_phase1/checkpoint_e3.pt" # Rollback to stable epoch 3 checkpoint
CHECKPOINT_DIR = "checkpoints_phase2"
SAMPLES_DIR = "val_samples_phase2"
LOG_FILE_NAME = "training_phase2.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# --- MODELS ---
torch.cuda.empty_cache()
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE)
vocos.requires_grad_(True)

disc = MultiResolutionDiscriminator().to(DEVICE)
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE)
wavlm.eval(); wavlm.requires_grad_(False)

gen_loss_func = GeneratorLoss().to(DEVICE)
disc_loss_func = DiscriminatorLoss().to(DEVICE)
resampler_16k = torchaudio.transforms.Resample(24000, 16000).to(DEVICE)

# --- OPTIMIZERS ---
opt_g = torch.optim.AdamW(vocos.parameters(), lr=LR)
opt_d = torch.optim.AdamW(disc.parameters(), lr=LR)
scaler_d = GradScaler('cuda')

# --- LOAD STABLE CHECKPOINT (ROLLBACK TO PHASE I E3) ---
if os.path.exists(OLD_CHECKPOINT_PATH):
    print(f"♻️ Loading stable foundation: {OLD_CHECKPOINT_PATH}")
    checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location=DEVICE)
    vocos.load_state_dict(checkpoint['vocos_state_dict'])
    disc.load_state_dict(checkpoint['disc_state_dict'])
    opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
    opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
    scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
    start_epoch = 4 # Start new branch from Epoch 4
else:
    start_epoch = 1

# --- DATASET ---
class DSRDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(l) for l in f if l.strip()]
            
    def __len__(self): 
        return len(self.data)
        
    def load_wav(self, path):
        full_path = os.path.join(DATA_ROOT, path.replace('\\', '/'))
        wav, sr = torchaudio.load(full_path)
        if sr != 24000: 
            wav = torchaudio.transforms.Resample(sr, 24000)(wav)
        return wav.squeeze(0)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        sel = random.choice([m for m in ["cosy", "f5", "mask", "fish"] if m in item])
        d, c = self.load_wav(item[sel]["path"]), self.load_wav(item["orig"]["path"])
        ml = min(len(d), len(c))
        
        if ml > CROP_SAMPLES:
            s = random.randint(0, ml - CROP_SAMPLES)
            d, c = d[s:s+CROP_SAMPLES], c[s:s+CROP_SAMPLES]
        else:
            d, c = F.pad(d, (0, CROP_SAMPLES - len(d))), F.pad(c, (0, CROP_SAMPLES - len(c)))
        return d, c

train_loader = DataLoader(DSRDataset("train.jsonl"), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(DSRDataset("val.jsonl"), batch_size=4, shuffle=False)

# --- LOGGING INIT ---
if not os.path.exists(LOG_FILE_NAME):
    with open(LOG_FILE_NAME, "w", newline='') as f:
        csv.writer(f).writerow(["epoch", "batch", "l_wavlm_raw", "l_wavlm_scaled", "l_energy_raw", "l_energy_scaled", "l_adv", "total_g"])

print(f"🚀 Starting Run 7.2 (Refined). Starting from epoch: {start_epoch}")
print(f"New weight balance: WavLM={W_WAVLM}, Energy={W_ENERGY}, Adv={W_ADV}")

for epoch in range(start_epoch, 21):
    vocos.train(); disc.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for i, (dirty, clean) in enumerate(loop):
        dirty, clean = dirty.to(DEVICE), clean.to(DEVICE)
        
        # --- STEP 1: DISCRIMINATOR ---
        opt_d.zero_grad()
        with autocast('cuda'):
            with torch.no_grad():
                mel = vocos.feature_extractor(dirty)
                x = vocos.backbone(mel)
                gv = vocos.head(x)
                if isinstance(gv, tuple): gv = gv[0]
                gv = torch.clamp(gv, min=-0.99, max=0.99).squeeze(1)[..., :clean.shape[-1]]
                c_d = clean[..., :gv.shape[-1]]
                
            out_d = disc(c_d, gv.detach())
            loss_d = sum([(disc_loss_func(r, f)[0] if isinstance(disc_loss_func(r, f), tuple) else disc_loss_func(r, f)) 
                          for r, f in zip(out_d[0], out_d[1])])
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- STEP 2: GENERATOR ---
        opt_g.zero_grad()
        dirty_safe = dirty + torch.randn_like(dirty) * 1e-7
        clean_safe = clean + torch.randn_like(clean) * 1e-7
        
        mel = vocos.feature_extractor(dirty_safe)
        x = vocos.backbone(mel)
        gv = vocos.head(x)
        if isinstance(gv, tuple): gv = gv[0]
        gv = torch.clamp(gv, min=-0.99, max=0.99).squeeze(1)[..., :clean_safe.shape[-1]]
        d_g = dirty_safe[..., :gv.shape[-1]]
        c_g = clean_safe[..., :gv.shape[-1]]
        
        # 1. SEMANTIC ANCHOR (SOTA layers 10-20)
        g16, d16 = resampler_16k(gv), resampler_16k(d_g)
        gf = wavlm(g16, output_hidden_states=True).hidden_states
        with torch.no_grad(): 
            df = wavlm(d16, output_hidden_states=True).hidden_states
        
        # Safe cosine calculation with eps
        l_wavlm_raw = sum([(1 - F.cosine_similarity(g + 1e-8, d + 1e-8, dim=-1)).mean() for g, d in zip(gf[10:20], df[10:20])])
        l_wavlm_scaled = l_wavlm_raw * W_WAVLM
        
        # 2. ACOUSTIC PURITY (GAN)
        out_g = disc(c_g, gv)
        l_adv_raw = sum([gen_loss_func(f)[0] if isinstance(gen_loss_func(f), tuple) else gen_loss_func(f) for f in out_g[1]])
        l_adv_scaled = l_adv_raw * W_ADV

        # 3. LOCAL ENERGY ENVELOPE (Energy)
        env_g = F.avg_pool1d(gv.unsqueeze(1).abs(), kernel_size=256, stride=256)
        env_d = F.avg_pool1d(d_g.unsqueeze(1).abs(), kernel_size=256, stride=256)
        l_energy_raw = F.l1_loss(env_g, env_d)
        l_energy_scaled = l_energy_raw * W_ENERGY 

        total_g = l_wavlm_scaled + l_adv_scaled + l_energy_scaled
        
        if not (torch.isnan(total_g) or torch.isinf(total_g)):
            total_g.backward()
            torch.nn.utils.clip_grad_norm_(vocos.parameters(), max_norm=1.0)
            opt_g.step()
        
        # LOGGING
        if i % 10 == 0:
            with open(LOG_FILE_NAME, "a", newline='') as f:
                csv.writer(f).writerow([epoch, i, l_wavlm_raw.item(), l_wavlm_scaled.item(), l_energy_raw.item(), l_energy_scaled.item(), l_adv_raw.item(), total_g.item()])

        loop.set_postfix(w=f"{l_wavlm_scaled.item():.2f}", adv=f"{l_adv_scaled.item():.2f}", tot=f"{total_g.item():.1f}")

    # --- VALIDATION (DETERMINISTIC 4-MODEL EXTRACTION) ---
    vocos.eval()
    with torch.no_grad():
        val_data = val_loader.dataset.data
        models_to_test = ["cosy", "f5", "mask", "fish"]
        
        for idx, model_name in enumerate(models_to_test):
            if idx >= len(val_data): break
            item = val_data[idx]
            
            if model_name not in item:
                continue
                
            vd_tensor = val_loader.dataset.load_wav(item[model_name]["path"]).to(DEVICE).unsqueeze(0)
            vc_tensor = val_loader.dataset.load_wav(item["orig"]["path"]).to(DEVICE).unsqueeze(0)
            
            # Crop/pad for memory safety during validation
            ml = min(vd_tensor.shape[-1], vc_tensor.shape[-1])
            if ml > CROP_SAMPLES:
                vd_tensor, vc_tensor = vd_tensor[..., :CROP_SAMPLES], vc_tensor[..., :CROP_SAMPLES]
            else:
                vd_tensor = F.pad(vd_tensor, (0, CROP_SAMPLES - vd_tensor.shape[-1]))
                vc_tensor = F.pad(vc_tensor, (0, CROP_SAMPLES - vc_tensor.shape[-1]))

            vg_tensor = vocos.head(vocos.backbone(vocos.feature_extractor(vd_tensor)))
            if isinstance(vg_tensor, tuple): vg_tensor = vg_tensor[0]
            vg_tensor = torch.clamp(vg_tensor, min=-0.99, max=0.99).squeeze(1)[..., :vc_tensor.shape[-1]]
            
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_1_dirty.wav", vd_tensor[0].cpu().unsqueeze(0), 24000)
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_2_gen.wav", vg_tensor[0].cpu().unsqueeze(0), 24000)
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_3_clean.wav", vc_tensor[0].cpu().unsqueeze(0), 24000)
    
    # --- SAVE ---
    checkpoint = {
        'epoch': epoch, 
        'vocos_state_dict': vocos.state_dict(), 
        'disc_state_dict': disc.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(), 
        'opt_d_state_dict': opt_d.state_dict(), 
        'scaler_d_state_dict': scaler_d.state_dict()
    }
    torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_e{epoch}.pt")
    gc.collect(); torch.cuda.empty_cache()