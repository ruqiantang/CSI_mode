
import torch, time
from wifo_upa.config import ModelConfig
from wifo_upa.data import SyntheticCSIDataset
from wifo_upa.model import UPAMAE
from wifo_upa.train import Trainer
from torch.utils.data import DataLoader
cfg = ModelConfig(embed_dim=256, decoder_embed_dim=256, depth=6, decoder_depth=4,
                  num_heads=8, decoder_num_heads=8, pt=4, pf=4, pe_mode="4d",
                  use_upa_bias=True, allow_spatial_mask=True)
model = UPAMAE(cfg).cuda()
nparam = sum(p.numel() for p in model.parameters())
ds = SyntheticCSIDataset("D14", num_samples=64, seed=0)
loader = DataLoader(ds, batch_size=8)
trainer = Trainer(model, task_schedule="sample", device="cuda")
for b in loader: trainer.train_epoch([b], epoch=0); break
torch.cuda.synchronize()
t0 = time.time()
for b in loader:
    trainer.train_epoch([b], epoch=0)
torch.cuda.synchronize()
t1 = time.time()
steps = len(loader)
sec_per_step = (t1-t0)/steps
print(f"params={nparam/1e6:.1f}M batch=8 sec/step={sec_per_step:.3f} samples/s={8/sec_per_step:.1f}")
h_per_epoch = 160000/(8/sec_per_step)/3600
print(f"1 epoch(160K样本)={h_per_epoch:.2f}h  20epoch={h_per_epoch*20:.0f}h  50epoch={h_per_epoch*50:.0f}h  200epoch={h_per_epoch*200:.0f}h")
