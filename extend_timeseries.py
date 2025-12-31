import numpy as np
import torch

@torch.no_grad()
def extend_series_slide1(model, series_np, window_size, extra_steps, device):
    """
    model: trained TimeSeriesTransformer
    series_np: [T, ROI] original real sequence
    window_size: L
    extra_steps: how many points to generate (length increase)
    return: [T + extra_steps, ROI]
    """
    model.eval()

    out = series_np.astype(np.float32).copy()

    for _ in range(extra_steps):
        src = out[-window_size:, :]                       # [L, ROI]
        src = torch.from_numpy(src).unsqueeze(0).to(device)  # [1, L, ROI]

        # 推理时只解码 1 步：tgt_in = last observed point
        tgt_in = src[:, -1:, :]                           # [1, 1, ROI]

        y1 = model(src=src, tgt=tgt_in, tgt_mask=None)    # 期望输出 [1, 1, ROI]
        y1 = y1.squeeze(0).squeeze(0).cpu().numpy()       # [ROI]

        out = np.vstack([out, y1[None, :]])               # append 1 point

    return out

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("your_model.pth", map_location=device).to(device)

x = np.load("some_subject.npy")          # [T, ROI]
x_aug = extend_series_slide1(model, x, window_size=30, extra_steps=300, device=device)

np.save("some_subject_extended.npy", x_aug)
print("Original length:", x.shape[0], "Extended length:", x_aug.shape[0])