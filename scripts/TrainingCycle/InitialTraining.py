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
LR = 1e-5
CROP_SAMPLES = 48000
DATA_ROOT = "/workspace/data/"
SAVE_INTERVAL = 1

# --- PHASE I CONFIGURATION ---
W_WAVLM = 2.5        # Strong semantic anchor
W_ENERGY = 200.0     # Strict energy adherence
W_ADV = 0.01         # Weakened GAN penalty for stability

# --- SETUP DIRECTORIES ---
CHECKPOINT_DIR = "checkpoints_phase1"
SAMPLES_DIR = "val_samples_phase1"
LOG_FILE_NAME = "training_phase1.csv"

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
            d = F.pad(d, (0, CROP_SAMPLES - len(d)))
            c = F.pad(c, (0, CROP_SAMPLES - len(c)))
        return d, c

train_loader = DataLoader(DSRDataset("train.jsonl"), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(DSRDataset("val.jsonl"), batch_size=4, shuffle=False)

# --- OPTIMIZERS ---
opt_g = torch.optim.AdamW(vocos.parameters(), lr=LR)
opt_d = torch.optim.AdamW(disc.parameters(), lr=LR)
scaler_d = GradScaler('cuda')

# --- LOGGING SETUP ---
log_file = open(LOG_FILE_NAME, "w", newline='')
log_writer = csv.writer(log_file)
# Logging only the effective metrics
log_writer.writerow([
    "epoch", "batch", 
    "l_wavlm_raw", "l_wavlm_scaled", 
    "l_energy_raw", "l_energy_scaled", 
    "l_adv", "total_g"
])

print(f"Weights: WavLM(Cos)={W_WAVLM}, Energy={W_ENERGY}, Adv={W_ADV}")

for epoch in range(1, 101):
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
                
                gv = torch.clamp(gv, min=-0.99, max=0.99)
                gv = gv.squeeze(1)[..., :clean.shape[-1]] 
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
        
        gv = torch.clamp(gv, min=-0.99, max=0.99)
        gv = gv.squeeze(1)[..., :clean_safe.shape[-1]]
        
        c_g = clean_safe[..., :gv.shape[-1]]
        d_g = dirty_safe[..., :gv.shape[-1]] 
        
        # 1. SEMANTIC ANCHOR (Target: Dirty, Method: Cosine Similarity)
        # Protects deepfake words and prosody from degradation
        g16, d16 = resampler_16k(gv), resampler_16k(d_g)
        gf = wavlm(g16, output_hidden_states=True).hidden_states
        with torch.no_grad(): 
            df = wavlm(d16, output_hidden_states=True).hidden_states
        
        l_wavlm_raw = sum([(1 - F.cosine_similarity(g + 1e-8, d + 1e-8, dim=-1)).mean() for g, d in zip(gf[10:20], df[10:20])])
        l_wavlm_scaled = l_wavlm_raw * W_WAVLM
        
        # 2. ACOUSTIC PURITY (GAN, Target: Clean)
        # The sole source of studio-quality acoustic knowledge
        out_g = disc(c_g, gv)
        l_adv_raw = sum([gen_loss_func(f)[0] if isinstance(gen_loss_func(f), tuple) else gen_loss_func(f) for f in out_g[1]])
        l_adv_scaled = l_adv_raw * W_ADV

        # 3. LOCAL ENERGY ENVELOPE (Target: Dirty)
        # Protects against volume explosions and hallucinations during pauses
        env_g = F.avg_pool1d(gv.unsqueeze(1).abs(), kernel_size=256, stride=256)
        env_d = F.avg_pool1d(d_g.unsqueeze(1).abs(), kernel_size=256, stride=256)
        l_energy_raw = F.l1_loss(env_g, env_d)
        l_energy_scaled = l_energy_raw * W_ENERGY 

        # TOTAL GENERATOR LOSS (Without conflicting spectral penalties)
        total_g = l_wavlm_scaled + l_adv_scaled + l_energy_scaled
        
        if torch.isnan(total_g) or torch.isinf(total_g):
            opt_g.zero_grad()
        else:
            total_g.backward()
            torch.nn.utils.clip_grad_norm_(vocos.parameters(), max_norm=1.0)
            opt_g.step()
        
        if i % 10 == 0:
            log_writer.writerow([
                epoch, i, 
                l_wavlm_raw.item(), l_wavlm_scaled.item(), 
                l_energy_raw.item(), l_energy_scaled.item(), 
                l_adv_raw.item(), l_adv_scaled.item(), total_g.item()
            ])
            log_file.flush()

        loop.set_postfix(
            w=f"{l_wavlm_scaled.item():.2f}", 
            eng=f"{l_energy_scaled.item():.2f}", 
            tot=f"{total_g.item():.1f}"
        )

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

            mel_v = vocos.feature_extractor(vd_tensor)
            vg_tensor = vocos.head(vocos.backbone(mel_v))
            if isinstance(vg_tensor, tuple): vg_tensor = vg_tensor[0]
            
            vg_tensor = torch.clamp(vg_tensor, min=-0.99, max=0.99).squeeze(1)[..., :vc_tensor.shape[-1]]
            
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_1_dirty.wav", vd_tensor[0].cpu().unsqueeze(0), 24000)
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_2_gen.wav", vg_tensor[0].cpu().unsqueeze(0), 24000)
            torchaudio.save(f"{SAMPLES_DIR}/e{epoch}_{model_name}_3_clean.wav", vc_tensor[0].cpu().unsqueeze(0), 24000)
    
    # --- SAVING FULL CHECKPOINT ---
    if epoch % SAVE_INTERVAL == 0:
        checkpoint = {
            'epoch': epoch,
            'vocos_state_dict': vocos.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict(),
            'scaler_d_state_dict': scaler_d.state_dict()
        }
        torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_e{epoch}.pt")
    
    gc.collect()
    torch.cuda.empty_cache()

log_file.close()