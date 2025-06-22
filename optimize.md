Add optimizations to code to make it faster

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = torch.compile(model, mode="default")

scaler = torch.amp.GradScaler("cuda", enabled=True)


with torch.amp.autocast(device_type="cuda", dtype=torch.float16)
  
```
