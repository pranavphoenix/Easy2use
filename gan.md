
GAN training

```python

optimizer_gen.zero_grad(set_to_none=True)
optimizer_disc.zero_grad(set_to_none=True)

gen.train()
disc.train()

output = gen(input)

if step % 2 == 0 and step > 0:
  fake_gen_pred = disc(output)
  gan_loss = - fake_gen_pred.mean()
else:
  gan_loss = torch.zero(1).to(output)

scaler.scale(gan_loss).backward()

scaler.unscale_(optimizer_gen)
torch.nn.utils.clip_grad_norm_(gen.parameters(), <max_grad_norm>)
scaler.step(optimizer_gen)
scaler.update()
optimizer_gen.zero_grad(set_to_none=True)

optimizer_disc.zero_grad(set_to_none=True)

real_disc_pred = disc(target)
disc_real_loss = F.relu(1. - real_disc_pred).mean() * 0.5

fake_disc_pred = disc(output.detach())
disc_fake_loss = F.relu(1. + fake_disc_pred).mean() * 0.5

disc_loss = (disc_real_loss + disc_fake_loss)

scaler_d.scale(disc_loss).backward()

scaler_d.unscale_(optimizer_disc)
torch.nn.utils.clip_grad_norm_(disc.parameters(), <max_grad_norm>)
scaler_d.step(optimizer_disc)
scaler_d.update()
optimizer_disc.zero_grad(set_to_none=True)


```
